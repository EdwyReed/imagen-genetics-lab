# Imagen Genetics Lab Offline Dataset Builder

This project provides a minimal orchestrator that automates dataset creation for LoRA training runs.  
It stitches together prompt construction, caption generation via Ollama, Imagen 3 image synthesis, and
structured artifact storage while maintaining deterministic seeds, reproducible logs, and complete metadata.

## Project Layout

```
project/
  data/dictionaries/    # reusable JSON dictionaries with structured traits
  conf/                 # TOML configuration and immutable Ollama system prompt
  out/                  # generated sessions (created at runtime)
  src/                  # orchestrator and helper modules
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.orchestrator --config conf/config.toml --count 2 --dry-run
```

The `--dry-run` flag skips Imagen calls so that the pipeline can be validated offline.  Remove the flag
once the `IMAGEN_API_KEY` environment variable is configured and Imagen access is available.

## Configuration

The default configuration is stored in `conf/config.toml`.  Runtime flags can override the run `count`,
`seed`, output directory, and dry-run behavior.  The orchestrator records every step to both stdout (with
color via Rich) and `log.txt` inside the session folder.

## Environment variables

* `IMAGEN_API_KEY` – API key for Google Imagen 3.  Required unless `--dry-run` is supplied.
* `IMAGEN_ENDPOINT` – optional override for the Imagen endpoint base URL.

## Related scripts

The repository also keeps the legacy `smart.py` orchestration script.  `src/adapters/smart_adapt.py`
provides thin wrappers around its Ollama and Imagen helpers so they can be reused in the new modular design.
