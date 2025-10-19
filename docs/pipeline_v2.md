# Imagen Genetics Lab — Modular Pipeline Implementation

This document describes the refactored pipeline that replaces the legacy monolith. The new
architecture fulfils the macro/meso/micro hierarchy requirements, JSON-driven styles and
characters, and deterministic orchestration for local testing.

## Module layout

```
imagen_lab/
├── io/json_loader.py        # Style/character ingestion + bias rules
├── scene/builder.py         # Pre-prompt assembly with macro/meso blending
├── caption/ollama.py        # Two-sentence caption synthesis
├── image/imagen.py          # Placeholder Imagen renderer
├── scoring/core.py          # Micro metric synthesis + meso aggregates
├── weights/manager.py       # Macro/meso profile blending + conflict handling
├── db/repository.py         # Normalised SQLite persistence
├── ga/engine.py             # Aggregate-driven GA helpers
└── pipeline/
    ├── configuration.py     # Comment-friendly config loader
    └── orchestrator.py      # Run controller + dry-run support
```

## JSON assets

* `assets/styles/*.json` describe style presets, bias rules, and gene pools.
* `assets/characters/*.json` enumerate character overrides and wardrobe options.

The loader validates required keys, converts gene weights to structured objects, and exposes
`StyleProfile` and `CharacterProfile` dataclasses consumed by the scene builder.

## Weight hierarchy

`weights.manager.WeightManager` blends:

1. Style defaults
2. Character overrides
3. User config macro overrides
4. Adaptive profile feedback

Conflicts such as high `sfw_level` combined with low `coverage_target` are clamped with a
recorded `Conflict` object returned in the pipeline result.

## Orchestration flow

1. `SceneBuilder` constructs a `PrePrompt` from JSON assets and macro/meso vectors.
2. `CaptionService` reduces the signal to macro + meso summaries and emits ≤40 word captions.
3. `ImagenService` generates deterministic placeholder JPEG files for integration tests.
4. `ScoringEngine` projects macro controls into the full set of micro metrics and configured
   meso aggregates. `fitness_visual`, `fitness_alignment`, and `fitness_sfw_balance` are derived
   from these aggregates.
5. `Repository` persists `runs`, `prompts`, and `scores` tables alongside optional profile vectors.
6. `WeightManager.update_from_scores` feeds meso aggregates back into the profile for future runs.

The orchestrator honours `dry_run_no_scoring` by skipping the scoring and feedback steps while
still recording artefacts. Profiles are saved when scoring executes, enabling resume semantics.

## Configuration format

`configs/pipeline.yaml` (or any YAML-like file parsed by the loader) contains:

* `style_preset` / `character_preset`
* `macro_weights` and `meso_overrides`
* `meso_aggregates` with component coefficients and optional targets
* `runtime` toggles (dry run, resume, population/generation sizes, Ollama model)
* `storage` paths for SQLite database and artefact directory
* Optional profile definitions for initial macro/meso seeds

The loader accepts genuine YAML (if PyYAML is installed) or a simple indentation-based syntax
compatible with the bundled parser for offline environments.

## Database schema

| Table     | Purpose |
|-----------|---------|
| `runs`    | Session-level metadata and macro/meso snapshots |
| `prompts` | Captions, gene selections, macro/meso JSON blobs, image path, fitness |
| `scores`  | Flattened micro + meso metric key/value pairs |
| `profiles`| Persistent macro/meso vectors with stability annotations |

Indices and schema mirror the plan outlined in the refactor document, enabling analytics and
GA warm starts without coupling orchestration to SQL internals.

## Dry run behaviour

When `dry_run_no_scoring` is enabled, the orchestrator bypasses scoring and profile updates while
still emitting captions and placeholder images. This matches the requirement for a fast render mode
that leans on the latest stable weights.

## Genetic helpers

`ga.engine.GAEngine` exposes deterministic selection and mutation helpers that operate on macro/meso
vectors only. They can seed follow-up runs by mutating the `novelty_preference` bias without touching
micro metrics directly.

---

The refactored modules, configuration loader, and tests live entirely within the repository without
external dependencies, delivering the requested modular pipeline.
