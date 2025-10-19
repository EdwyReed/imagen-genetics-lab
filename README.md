# Imagen Genetics Lab

This repository orchestrates a glossy pin-up illustration workflow that combines:

* **Scene assembly** driven by a rich style catalog (defaults to `catalogs/all-together.json`).
* **Prompting** via Ollama to turn scene metadata into natural captions.
* **Image generation** with Google Imagen 3.
* **Dual scoring** (style + NSFW) for automatic quality tracking.
* **Optional genetic evolution** to iteratively explore promising prompt configurations.

All runtime parameters are managed through [`config.yaml`](./config.yaml). The new modular layout makes it easy to customize a single component without touching the others.

## Project layout

```
imagen-genetics-lab/
├── config.yaml                 # Global configuration (paths, defaults, GA tuning)
├── catalogs/                   # Candy Danger catalog family
│   ├── all-together.json       # Aggregated options across every variant
│   ├── sugar-trouble.json      # Core sticky-sweet mix
│   ├── cherry-pop-idol.json    # Neon pop-art escalation
│   ├── lick-me-softly.json     # NSFW-leaning syrup overload
│   ├── cotton-kiss.json        # Soft SFW daytime variant
│   └── jelly-pin-up.json       # Legacy jelly-gloss preset (optional)
├── process.py                  # Image pre-processing utility
├── scorer.py                   # DualScorer (style + NSFW)
├── smart.py                    # CLI entrypoint
├── imagen_lab/                 # Modular pipeline package
│   ├── __init__.py
│   ├── catalog.py              # JSON catalog helpers
│   ├── config.py               # Dataclasses + YAML loader
│   ├── ga.py                   # Genetic operators and DB helpers
│   ├── pipeline.py             # Plain + GA orchestration logic
│   ├── prompting.py            # Ollama + Imagen utilities
│   ├── randomization.py        # Weighted selectors & RNG helpers
│   ├── scene_builder.py        # Dataclasses + scene assembly logic
│   └── storage.py              # Artifact writing & prompt logging
└── README.md
```

## Prerequisites

* Python 3.10+
* Required Python packages: `google-genai`, `requests`, `python-dotenv`, `open-clip-torch`, `opennsfw2`, `numpy`, `torch`, `Pillow`, `scikit-image`, `ultralytics`, and `piexif` (optional, for EXIF comments).
* Access to a running Ollama instance with the desired caption model.
* Access to Google Imagen 3 via the `google-genai` SDK.

Install dependencies with `pip`:

```bash
pip install -r requirements.txt
```

> **Note**: The repository does not ship a frozen requirements file. Install the versions compatible with your environment or export them yourself after a successful setup.

## Configuration (`config.yaml`)

Key sections:

| Section      | Purpose |
|--------------|---------|
| `presets`    | Default `profile_id`, `style_preset`, and `character_preset` used when the pipeline boots. |
| `storage`    | Paths for the SQLite database, JSONL score log, artefact output, catalogs, and the bias profile directory (`profiles/<id>.json`). |
| `runtime`    | Groups the prompting templates, Ollama/Imagen parameters, GA tuning knobs, default SFW/temperature settings, and feedback/history switches. |
| `scoring`    | DualScorer device, batch size, temperature schedule, calibration ranges, inline weights, and dynamic profile options. |
| `macro_weights` | Baseline macro regulators (6 entries) that seed the bias engine. |
| `meso_aggregates` | Baseline meso regulators (6 entries) for the bias engine. |
| `bias_rules` | Declarative DSL rules applied by the bias engine before sampling scene genes. |

Adjust `config.yaml` whenever you need to change long-lived settings. The CLI exposes overrides for quick experiments without editing the file.

## Running the pipeline

### Plain batch mode

Generates `cycles × per_cycle` images without evolution.

```bash
python smart.py --config config.yaml --cycles 4 --per-cycle 2
```

Useful overrides:

* `--outdir /path/to/output`
* `--sfw 0.4`
* `--temperature 0.6`
* `--ollama-model my-model`

### Genetic evolution mode

Evolves scene genes (palette, lighting, camera, etc.) using fitness feedback.

```bash
python smart.py --config config.yaml --evolve --gens 6 --pop 12
```

Resume from previous high-scoring prompts stored in the SQLite log:

```bash
python smart.py --config config.yaml --evolve --resume-best --resume-k 8
```

### Example: custom Imagen model

```bash
python smart.py --config config.yaml --model imagen-3.0-generate-003 --person-mode allow_all
```

All CLI options default to the values declared in `config.yaml`. See `python smart.py --help` for the full list.

## Data flow

1. **Scene construction** (`imagen_lab.scene_builder`)
   * Samples palette, lighting, background, camera setup, wardrobe, props, and mood from the configured catalog (`catalogs/all-together.json` by default) using the `WeightedSelector`.
   * Returns typed dataclasses containing metadata and Ollama-ready payloads.

2. **Prompting** (`imagen_lab.prompting`)
   * Builds a system prompt tuned to the requested SFW level.
   * Calls Ollama to generate captions, enforcing required terms and word-count bounds.

3. **Imagen generation**
   * Sends the caption to Imagen 3 with the configured aspect ratio and person settings.

4. **Scoring & storage** (`scorer.DualScorer`, `imagen_lab.storage`)
   * Calculates NSFW/style scores, composition metrics (cropping tightness, rule-of-thirds alignment, negative space), writes them to SQLite/JSONL, logs metadata, and persists per-component style breakdowns plus aggregated batch metrics.

## Pose & segmentation benchmarks

For downstream body-tracking integrations we maintain GPU benchmarks for the
most likely pose-estimation and human-segmentation candidates. Refer to
[`wiki/pose_benchmark.md`](./wiki/pose_benchmark.md) for GTX 1060 guidance,
including VRAM usage, throughput and recommended FP16/CPU fallback settings.
  * Calculates NSFW/style scores, writes them to SQLite/JSONL, logs metadata, and persists per-component style breakdowns plus aggregated batch metrics.

5. **Genetic evolution** (`imagen_lab.ga`, `imagen_lab.pipeline`)
   * Maintains populations of gene IDs, applies crossover/mutation, and optionally seeds from the highest-fitness prompts found in prior runs.

## Extending the system

* Add new wardrobe items, lighting presets, or camera options by editing any file under `catalogs/`; the scene builder reads the catalog dynamically.
* Switch the entire art direction by pointing `paths.catalog` to a different file under `catalogs/`. The prompting layer now derives its system prompt and required terms from the selected catalog's metadata, so swapping catalogs fully rewires the generated captions without additional tweaks.
* Customize selection logic or introduce new genes inside `imagen_lab/scene_builder.py` and `imagen_lab/ga/genes.py`.
* Swap or reconfigure scoring components by editing `scorer.py`—no changes to the pipeline layer are required.

## Troubleshooting

* Ensure Ollama and Imagen endpoints are reachable before launching the pipeline.
* If you change GPU/CPU availability, adjust `scoring.device` and `scoring.batch_size` in `config.yaml`.
* The SQLite database (`scores.sqlite`) and JSONL log (`scores.jsonl`) grow over time; archive or prune them periodically.

### Log schema versioning

Every score record now includes an explicit schema version to make downstream parsing safer:

* **SQLite (`scores.sqlite`, table `scores`)** — new column `schema_version` (integer, defaults to `1` for historical rows, new
  writes use `2`).
* **JSONL (`scores.jsonl`)** — each object adds the `schema_version` field with the same integer value.

Pipelines consuming these logs should branch on `schema_version` when reading older exports and can fail fast if an unexpected
version appears.

Happy experimenting!

## Adaptive style weighting

The `scoring.auto_weights` section enables an exponential moving-average controller that keeps the CLIP, specular and illustration
signals in balance. When `enabled` is `true`, the scorer monitors recent batches and nudges the component weights toward the targ
et value while respecting the configured bounds (`min_weight`, `max_weight`, etc.). This produces stable, incremental corrections
instead of abrupt jumps and keeps the overall style score aligned with the desired glossy watercolor look.

```yaml
scoring:
  weights:
  auto_weights:
    enabled: true
    target: 0.88
    ema_alpha: 0.25
    momentum: 0.35
```

Profiles are normalised automatically and persisted in `weight_profiles.yaml`. Use the bundled `notebooks/style_weight_profiles.ipynb` report to compare how each profile influences the weighted style score for representative component values.

Tune the controller parameters to match your dataset; higher `momentum` reacts faster, while a smaller `ema_alpha` favours long-term stability.

## Calibrating or resetting weights

Use `weights_tool.py` to manage the active profile and inline fallback weights:

```bash
# Reset to factory defaults
python weights_tool.py --reset

# Estimate weights from a directory of reference images (dry run)
python weights_tool.py --reference-dir ./golden_samples --dry-run

# Solve weights and update the selected profile + config.yaml
python weights_tool.py --reference-dir ./golden_samples --target 0.92
```

The calibration pass scores every image in the directory, solves a least-squares problem so the references approach the desired style score, and writes the normalised weights back to both the configuration file and the selected profile entry. Combine this with the adaptive controller for a feedback loop that converges quickly and remains steady across long runs.
