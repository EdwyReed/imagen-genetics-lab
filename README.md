# Imagen Genetics Lab (Simplified)

This repository now focuses on a single task: assembling a lightweight preprompt
object from small template libraries and saving the metadata for downstream
image generation experiments.

## Template layout

All prompt fragments are defined under [`templates/`](./templates/):

- `poses/` — posture snippets.
- `items/` — props or accessories.
- `characters/` — persona descriptors.
- `environment/` — surrounding location or mood.
- `styles/` — rendering guidance.

Each JSON file contains:

```json
{
  "name": "identifier",
  "prompt": "fragment text",
  "nsfw_weight": 0.35
}
```

The `nsfw_weight` expresses how spicy the fragment is on a scale from 0 to 1. The
generator biases its random choice toward fragments whose `nsfw_weight` is close
to the requested NSFW level.

## Prompt generator

Use [`prompt_generator.py`](./prompt_generator.py) to build a randomized
preprompt object. The script accepts only three optional arguments:

```bash
python prompt_generator.py \
  --nsfw 0.6 \   # desired NSFW level (0 to 1, default 0.0)
  --output-dir generated \   # where to write the JSON metadata (default ./generated)
  --seed 42       # optional seed for deterministic sampling
```

The script prints the resulting JSON to stdout and writes the same payload to the
output directory (files are timestamped). Each run returns:

- the combined `prompt` string;
- the selected fragment from every category, including its template source file;
- the NSFW level and optional seed used during sampling;
- a `generated_at` timestamp.

Tweak or extend the templates to control the vocabulary. Nothing else is
required—there are no additional dependencies or background services.
