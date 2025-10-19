# Imagen Genetics Lab — Refactor & Weight Architecture Plan

## 1. Complexity Assessment of the Current Codebase

### 1.1 Monolithic pipeline orchestration
The current `pipeline.py` module instantiates almost every service (catalogs, characters, scoring, embeddings, feedback, logging), manages prompt workflow logic, controls Ollama/Imagen calls, and performs GA seeding loops in one file. Responsibilities such as dependency wiring, runtime loops, API calls, scoring orchestration, persistence, and feedback all live together, which makes reasoning about the flow and testing individual components difficult.【F:imagen_lab/pipeline.py†L14-L520】

### 1.2 Configuration sprawl without domain abstractions
`config.py` exposes nested dataclasses that mirror historical behavior (e.g., GA parameters, scoring internals, prompt enforcement, embedding caches). Many knobs are tuned for the existing pipeline but do not reflect the macro/meso/micro hierarchy required by the new system. There is no separation between user-facing configuration (style/character presets, macro sliders) and low-level runtime toggles (device, batch, embedding cache).【F:imagen_lab/config.py†L1-L145】

### 1.3 Genetic logic coupled to SQLite schemas
`ga.py` reads fitness data directly from the SQLite database, reaches into JSON blobs, and mutates gene IDs using ad-hoc selectors. This tight coupling of evolution logic to persistence makes it hard to evolve toward the requested macro/meso weight-driven system and introduces hidden dependencies on specific table layouts.【F:imagen_lab/ga.py†L1-L98】

### 1.4 Feedback and scoring intertwined with prompt execution
Inside the main loops, scoring results immediately trigger feedback updates and prompt-level persistence. Because scoring, feedback, and artifact management share the same procedural flow, replacing the scorer or skipping evaluation (“dry runs”) requires branching across large blocks of code instead of swapping strategy objects.【F:imagen_lab/pipeline.py†L330-L489】

## 2. Target Architecture Overview

We propose a layered architecture that cleanly separates domain data (styles, characters, weights), orchestration (pre-prompt → caption → image → scoring), learning (feedback, profiles), and infrastructure (storage, external APIs). Core packages:

```
imagen_lab/
├── io/json_loader.py         # Schema validation + loading for style/character JSON
├── scene/builder.py          # Pre-prompt assembly and bias application
├── caption/ollama.py         # Caption API client (macro/meso inputs only)
├── image/imagen.py           # Imagen invocation + response normalization
├── scoring/core.py           # Micro metric calculators + meso aggregations
├── db/repository.py          # Persistence layer (runs, prompts, scores, profiles)
├── weights/manager.py        # Macro/meso weight blending, profile evolution
├── ga/engine.py              # Optional evolutionary search operating on aggregates
└── pipeline/orchestrator.py  # High-level run control + mode switches
```

Each module exposes a narrow interface to enable unit testing and easier substitutions (e.g., switch to a different caption model or scoring backend) without altering other layers.

## 3. JSON-driven Style & Character Definitions

* `styles/*.json` — declarative files containing stylistic tags, palette hints, macro biases, and meso defaults. Example keys: `id`, `description`, `macro_defaults`, `meso_defaults`, `bias_rules`.
* `characters/*.json` — persona descriptors with demographic info, wardrobe seeds, personality cues, and allowable macro overrides.

`json_loader` is responsible for:

1. Validating JSON against JSON Schema definitions.
2. Merging defaults (e.g., applying style macro defaults unless overridden in the main config).
3. Producing typed `StyleProfile` and `CharacterProfile` objects consumed by the scene builder.

## 4. Pre-prompt Assembly Pipeline

`scene.builder` composes a structured `PrePrompt` object:

* Base content: selected style profile, optional character profile, environment/pose “genes.”
* Macro bias: apply weights (`sfw_level`, `era_target`, etc.) to adjust selection probabilities for pose, lighting, palette, wardrobe.
* Meso influence: use aggregated scores (e.g., `fitness_style`, `fitness_body_focus`) from profiles to prioritize gene bundles.
* Output structure: JSON-ready object with `scene_summary`, `visual_axes`, `style_tokens`, `macro_controls`, and `selected_genes`.

`PrePrompt` → `CaptionPayload` transformation strips micro metrics and retains only macro/meso descriptors and factual scene notes before calling the caption service.

## 5. Caption → Image → Scoring Flow

1. **Caption (Ollama)** — `caption.ollama` receives the macro/meso subset, pre-prompt summary, and deterministic seed. It returns ≤2 sentences. Enforcement of required terms is handled locally without exposing micro metrics.
2. **Image (Imagen)** — `image.imagen` wraps Imagen requests, tracks guidance parameters, and stores JPEGs via an artifact sink.
3. **Scoring** — `scoring.core` computes all micro metrics and derives meso aggregates. It outputs `ScoreResult` objects referencing the generated assets.
4. **Persistence** — `db.repository` stores `Run`, `Prompt`, `Score`, and optional `ProfileSnapshot` records with normalized schemas described in §6.

## 6. Database Schema Redesign

Tables (SQLite or Postgres):

* `runs(run_id TEXT PK, timestamp, config_json JSONB, profile_id TEXT NULL)`
* `prompts(prompt_id TEXT PK, run_id, caption TEXT, preprompt JSONB, genes JSONB, macro_weights JSONB, meso_weights JSONB, image_path TEXT, fitness REAL, parent_ids JSONB, operator TEXT, generation INTEGER, individual INTEGER)`
* `scores(prompt_id FK, metric TEXT, micro_value REAL, meso_value REAL NULL, PRIMARY KEY(prompt_id, metric))`
* `profiles(profile_id TEXT PK, vector_json JSONB, last_run_id TEXT, stability_score REAL)`
* `bias_stats(metric TEXT PK, success_weight REAL, failure_weight REAL, updated_at TIMESTAMP)` (optional rolling statistics)

The repository layer translates between domain models and SQL, so GA and feedback modules do not need SQL knowledge.

## 7. Weight Hierarchy Implementation

* **Macro weights** — stored in the main config / profile vector; control high-level biasing of gene selectors and macro hints passed to Ollama.
* **Meso aggregates** — derived from recent scoring sessions, stored per profile; used to weight gene pools (pose, lighting, palette). Example formula maintained as configuration (e.g., `fitness_style = 0.5*style_core + ...`).
* **Micro metrics** — computed during scoring, persisted for analytics only.

`weights/manager.py` responsibilities:

1. Load macro defaults from config + selected style/character presets.
2. Blend with active profile vectors and runtime overrides.
3. Provide probability adjustment factors to the scene builder (`select_gene(category) -> WeightedChoice`).
4. Update profile vectors after scoring (EMA-based) using meso feedback, keeping micro metrics internal.

## 8. Feedback & Learning Loop

1. After each scored run, `weights.manager` ingests meso aggregates (`fitness_*`) and updates profile components (macro-level vector) plus bias statistics for gene categories.
2. GA engine (if enabled) operates on macro/meso vectors and gene bundles, not micro metrics. It retrieves top-performing prompts via repository queries filtered by fitness scores.
3. Profiles remain compact (10–12 macro/meso dimensions) and are versioned. Micro metrics remain in analytics for dashboards.

## 9. Configuration Design

Adopt a comment-friendly format (YAML or JSONC). Suggested structure:

```yaml
style_preset: watercolor_poster
character_preset: default_muse
macro_weights:
  sfw_level: 0.6
  era_target: "80s"
  focus_mode: "style_first"
  gloss_priority: 0.7
  coverage_target: 0.4
  illustration_strength: 0.8
  novelty_preference: 0.5
  lighting_softness: 0.6
  retro_authenticity: 0.75
meso_aggregates:
  fitness_style:
    formula: 0.5*style_core + 0.3*gloss_intensity + 0.2*softness_blur
    target: 75
  fitness_body_focus:
    formula: 0.4*chest_focus + 0.4*thigh_focus + 0.2*pose_suggestiveness
bias_rules:
  - when: {sfw_level: "< 0.3"}
    adjust: {coverage_gene: -0.4, pose_gene: +0.2}
  - when: {era_target: "80s"}
    adjust: {palette_gene: +0.3, lighting_gene: +0.2}
runtime:
  dry_run_no_scoring: false
  resume_best: true
  population: 12
  generations: 6
```

The configuration loader resolves references to the selected style/character JSON files, merges their defaults, and optionally saves the blended macro vector as a reusable profile snapshot.

## 10. Run Modes & Dry Run Support

* **Full mode** — Executes the entire pipeline (pre-prompt → caption → image → scoring). Results update weights, bias statistics, and profiles.
* **Dry run (`dry_run_no_scoring: true`)** — Skips the scoring stage, reuses the latest stable profile snapshot from the repository, and records generated prompts/images with a flag indicating no scoring was performed.
* **Resume-best** — Pulls top-performing prompts from persistence, seeds GA population with their macro/meso vectors, and resumes search.

Mode switching is handled in `pipeline/orchestrator.py` via strategy objects rather than branching inside a monolithic loop.

## 11. Evolutionary Search over Aggregates

`ga.engine` operates on `GenePlan` structures containing:

* Selected gene IDs (pose, lighting, palette, wardrobe) with attached macro/meso modifiers.
* Fitness evaluations from the scoring layer (when available).

Mutation/crossover adjust macro bias values or swap gene bundles rather than toggling micro metrics. The engine interacts only with the repository API (`load_top_prompts`, `store_generation_result`) to remain persistence-agnostic.

## 12. Contradiction Handling

Potential conflict: `sfw_level` high (approaching 1.0) while `coverage_target` low (near 0.0). The system should detect this during config validation and:

* Warn the user that the combination is risky for SFW policies.
* Automatically clamp `coverage_target` to the minimum allowed for the declared `sfw_level` (e.g., enforce `coverage_target >= 0.35` when `sfw_level >= 0.8`).
* Document the applied clamp in the run metadata so downstream analytics understand the adjustment.

## 13. Migration & Implementation Plan

1. **Domain extraction** — Introduce new packages (`io`, `scene`, `caption`, `image`, `scoring`, `db`, `weights`, `ga`, `pipeline`). Migrate existing logic incrementally, ensuring old modules delegate to the new services before full removal.
2. **JSON schema & loaders** — Define JSON schemas for style/character files, migrate current catalog data to the new format, and update loaders to output structured profiles.
3. **Pre-prompt builder** — Reimplement scene assembly around macro/meso biases with deterministic selection hooks for GA and feedback.
4. **API clients** — Wrap Ollama/Imagen interactions in isolated modules with retries, logging, and payload contracts.
5. **Scoring refactor** — Port micro-metric computation to `scoring.core`, ensure meso formulas are configurable, and emit normalized score objects.
6. **Persistence layer** — Build repository API with migrations for the new schema, including run/prompt/score/profile tables and indices supporting fitness queries.
7. **Weight manager & profiles** — Implement macro/meso blending, feedback updates, and dry-run snapshot loading.
8. **Pipeline orchestrator** — Compose services, implement mode switches, integrate GA engine, and expose CLI flags aligned with the new config.
9. **QA & reporting** — Provide analytics scripts demonstrating option shifts when macro weights change (e.g., coverage↓, gloss↑, era=80s), ensuring database snapshots verify the bias behavior.
10. **Deprecation cleanup** — Remove legacy modules once parity is confirmed; update documentation and examples to reference the new architecture.

