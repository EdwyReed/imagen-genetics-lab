# Imagen Genetics Lab — Prompt-to-Image Pipeline

This repository provides a stripped-down workflow for assembling randomized
preprompts, enriching them via Ollama, and finally generating imagery with an
Imagen-compatible endpoint. The goal is to keep the orchestration minimal while
retaining a reproducible record of every decision made along the way.

## Template catalogue

Prompt fragments are stored under [`templates/`](./templates/):

- `poses/` — posture descriptions.
- `items/` — props or accessories.
- `characters/` — persona descriptors.
- `environment/` — setting or ambience cues.
- `styles/` — rendering and stylistic hints.

Each template file contains:

```json
{
  "name": "identifier",
  "prompt": "fragment text",
  "nsfw_weight": 0.35
}
```

The `nsfw_weight` indicates how daring a fragment is on a 0–1 scale. When a run
requests a specific NSFW level, the sampler biases towards fragments whose
weights are closest to the requested intensity.

## Runtime requirements

The orchestration script expects two services to be available:

1. **Ollama** — used to expand the structured preprompt into a natural-language
   caption. The defaults target `http://localhost:11434` with the `llama3`
   model, but you can override both values via command-line flags.
2. **Imagen-compatible generator** — any HTTP endpoint that accepts a JSON
   payload with `model`, `prompt`, `variants`, `aspect_ratio`, and
   `guidance_scale` fields and returns base64-encoded images. By default the
   script targets `http://localhost:9090` with a generic `imagen` model name.

Both endpoints are contacted over HTTP using the Python standard library. If an
endpoint is unreachable or returns an invalid payload, the script will abort
with an informative error message.

## Running the pipeline

Launch [`prompt_generator.py`](./prompt_generator.py) to run the full
preprompt → caption → image flow:

```bash
python prompt_generator.py \
  --nsfw 0.6 \            # desired NSFW intensity (0–1)
  --seed 42 \             # optional seed for deterministic sampling
  --output-dir generated  # where to store metadata and images
```

Additional knobs are available, all with sensible defaults:

- `--ollama-url`, `--ollama-model`, `--ollama-timeout`
- `--caption-temperature`, `--caption-top-p`
- `--imagen-url`, `--imagen-model`, `--imagen-timeout`
- `--variants`, `--aspect-ratio`, `--guidance-scale`

The script prints the full metadata JSON to stdout and also writes it to a
timestamped file inside `--output-dir`. Image variants are saved alongside the
metadata file using the same timestamped stem.

## Output structure

Each run produces a JSON document with:

- the sampled templates and assembled preprompt string;
- the exact payload sent to Ollama and the resulting caption;
- Imagen request parameters and a list of saved variants, including the relative
  path to each decoded image and any metadata echoed by the service.

This makes every downstream asset fully reproducible—rerun the pipeline with the
same seed and service configuration to recreate both the caption and the
resulting imagery.
