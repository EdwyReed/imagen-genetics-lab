# Imagen Genetics Lab – Pipeline v0.9

The project provides a data-driven prompt assembly and evolution pipeline. Version 0.9 performs a hard reset of the repository structure: the new CLI, asset schemas, and genome are **not** compatible with earlier releases.

## Quick start

1. Prepare assets and packs using the unified format under `data/catalog/` and optional overlays in `data/packs/`.
2. Create a bias file containing weight overrides and locks:

```json
{
  "weights": {
    "palettes:neutral_warm": 1.0
  },
  "locks": {}
}
```

3. Author a scenario JSON/YAML following `schemas/scenario.schema.json`.
4. Run generation:

```bash
python main.py run plain \
  --style-profile retro_glossy \
  --asset-pack data/packs/examples \
  --bias bias.json \
  --scenario scenarios/demo.json \
  --out out/plain_run
```

To request captions from a local Ollama instance and hand them off to Google Imagen, add `--generate-images` and the desired
options:

```bash
python main.py run plain \
  --style-profile retro_glossy \
  --asset-pack data/packs/examples \
  --bias bias.json \
  --scenario scenarios/demo.json \
  --out out/plain_run \
  --generate-images \
  --ollama-model qwen2.5:3b \
  --imagen-model imagen-3.0-generate-002 \
  --imagen-variants 2
```

The workflow automatically ensures that the Ollama model is available (pulling it on demand and removing it afterwards unless
`--ollama-keep-model` is provided), forwards the assembled scene payload together with weights and the system prompt, and embeds the
resulting final prompt plus all generation options in every image’s EXIF metadata. Each variant also receives adjacent `.json` and
`.txt` companions storing the metadata and prompt. The CLI writes JSON artefacts per stage/cycle and appends metadata to
`runs.jsonl`. For GA mode:

```bash
python main.py run evolve \
  --style-profile retro_glossy \
  --asset-pack data/packs/examples \
  --bias bias.json \
  --scenario scenarios/demo.json \
  --ga-pop 8 --ga-gen 4 \
  --out out/evolve_run
```

## Components

- `imagen_pipeline/core/assets.py` – asset loading across packs with last-wins merge and schema validation.
- `imagen_pipeline/core/preferences.py` – bias weights and locks.
- `imagen_pipeline/core/selector.py` – weighted selector honouring locks and compatibility checks.
- `imagen_pipeline/core/system_prompt.py` – system prompt assembly from style profiles/tokens and rules.
- `imagen_pipeline/core/scenarios.py` – scenario loader with stage iteration.
- `imagen_pipeline/core/build.py` – scene assembly, gene extraction, metadata capture.
- `imagen_pipeline/core/evolve.py` – minimal GA operators over the new genome.

## Schemas

Schemas live under `schemas/`. All assets, style profiles/tokens, scenarios, and locks are validated at load time.

## Offline learning

Use `tools/offline_update_bias.py` to adjust weights based on recorded runs:

```bash
python tools/offline_update_bias.py --runs out/plain_run/runs.jsonl --bias bias.json
```

The utility applies a bounded EMA to palette and pose weights while leaving locks untouched.

## Tests

Run smoke tests with:

```bash
pytest
```

The suite covers selectors, asset merging, prompt assembly, scenario execution, genome handling, and GA constraints.
