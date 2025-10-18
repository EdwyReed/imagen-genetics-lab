# Imagen Genetics Lab

This repository orchestrates a glossy pin-up illustration workflow that combines:

* **Scene assembly** driven by a rich style catalog (`jelly-pin-up.json`).
* **Prompting** via Ollama to turn scene metadata into natural captions.
* **Image generation** with Google Imagen 3.
* **Dual scoring** (style + NSFW) for automatic quality tracking.
* **Optional genetic evolution** to iteratively explore promising prompt configurations.

All runtime parameters are managed through [`config.yaml`](./config.yaml). The new modular layout makes it easy to customize a single component without touching the others.

## Project layout

```
imagen-genetics-lab/
├── config.yaml                 # Global configuration (paths, defaults, GA tuning)
├── jelly-pin-up.json           # Style catalog for scene construction
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
* Required Python packages: `google-genai`, `requests`, `python-dotenv`, `open-clip-torch`, `opennsfw2`, `numpy`, `torch`, `Pillow`, `scikit-image`, and `piexif` (optional, for EXIF comments).
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
| `paths`      | File locations for the style catalog, SQLite database, JSONL score log, and output directory. Paths are resolved relative to the repository root. |
| `prompting`  | Required style terms and template IDs used by the scene builder when crafting Ollama payloads. |
| `ollama`     | Endpoint, model name, default temperature, and `top_p` for caption generation. |
| `imagen`     | Imagen model identifier, person-generation mode, and guidance scale. |
| `scoring`    | DualScorer device, batch size, tau, calibration ranges, component weights, and optional auto-weight tuning. |
| `fitness`    | Weights applied to style and NSFW scores when computing fitness. |
| `defaults`   | Baseline values for SFW level, caption temperature, number of cycles, etc. |
| `ga`         | Population size, number of generations, elite fraction, mutation/crossover rates, and resume strategy for genetic runs. |

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
   * Samples palette, lighting, background, camera setup, wardrobe, props, and mood from `jelly-pin-up.json` using the `WeightedSelector`.
   * Returns typed dataclasses containing metadata and Ollama-ready payloads.

2. **Prompting** (`imagen_lab.prompting`)
   * Builds a system prompt tuned to the requested SFW level.
   * Calls Ollama to generate captions, enforcing required terms and word-count bounds.

3. **Imagen generation**
   * Sends the caption to Imagen 3 with the configured aspect ratio and person settings.

4. **Scoring & storage** (`scorer.DualScorer`, `imagen_lab.storage`)
   * Saves JPEG + JSON + TXT sidecars (with EXIF notes when `piexif` is installed).

## Pose & segmentation benchmarks

For downstream body-tracking integrations we maintain GPU benchmarks for the
most likely pose-estimation and human-segmentation candidates. Refer to
[`wiki/pose_benchmark.md`](./wiki/pose_benchmark.md) for GTX 1060 guidance,
including VRAM usage, throughput and recommended FP16/CPU fallback settings.
   * Calculates NSFW/style scores, writes them to SQLite/JSONL, and logs metadata for later analysis.

5. **Genetic evolution** (`imagen_lab.ga`, `imagen_lab.pipeline`)
   * Maintains populations of gene IDs, applies crossover/mutation, and optionally seeds from the highest-fitness prompts found in prior runs.

## Extending the system

* Add new wardrobe items, lighting presets, or camera options by editing `jelly-pin-up.json`; the scene builder reads the catalog dynamically.
* Customize selection logic or introduce new genes inside `imagen_lab/scene_builder.py` and `imagen_lab/ga.py`.
* Swap or reconfigure scoring components by editing `scorer.py`—no changes to the pipeline layer are required.

## Troubleshooting

* Ensure Ollama and Imagen endpoints are reachable before launching the pipeline.
* If you change GPU/CPU availability, adjust `scoring.device` and `scoring.batch_size` in `config.yaml`.
* The SQLite database (`scores.sqlite`) and JSONL log (`scores.jsonl`) grow over time; archive or prune them periodically.

Happy experimenting!

## Adaptive style weighting

The `scoring.auto_weights` section enables an exponential moving-average controller that keeps the CLIP, specular and illustration
signals in balance. When `enabled` is `true`, the scorer monitors recent batches and nudges the component weights toward the targ
et value while respecting the configured bounds (`min_weight`, `max_weight`, etc.). This produces stable, incremental corrections
instead of abrupt jumps and keeps the overall style score aligned with the desired glossy watercolor look.

```yaml
scoring:
  weights:
    clip: 0.55
    spec: 0.35
    illu: 0.10
  auto_weights:
    enabled: true
    target: 0.88
    ema_alpha: 0.25
    momentum: 0.35
```

Tune the parameters to match your dataset; higher `momentum` reacts faster, while a smaller `ema_alpha` favours long-term stabilit
y.

## Calibrating or resetting weights

Use `weights_tool.py` to manage the persisted weights in `config.yaml`:

```bash
# Reset to factory defaults
python weights_tool.py --reset

# Estimate weights from a directory of reference images (dry run)
python weights_tool.py --reference-dir ./golden_samples --dry-run

# Solve weights and write them back to config.yaml
python weights_tool.py --reference-dir ./golden_samples --target 0.92
```

The calibration pass scores every image in the directory, solves a least-squares problem so the references approach the desired st
yle score, and writes the normalised weights back to the YAML file. Combine this with the adaptive controller for a feedback loop t
hat converges quickly and remains steady across long runs.
