# Migrating to v0.9

Version 0.9 intentionally breaks compatibility with all previous releases. Follow the checklist below when upgrading an existing workspace.

1. **Remove legacy code paths.** Delete the old CLI, prompt builder, and schema files. Only the modules under `imagen_pipeline/core/` are supported.
2. **Rebuild style profiles.** Create dedicated JSON files in `data/catalog/style/` with `kind="profile"` (e.g. `retro_glossy.json`) containing the `system_prompt` and `required_terms` arrays.
3. **Restructure assets.** Move assets into per-group directories under `data/catalog/`. Optional overrides live in `data/packs/base/` or additional pack folders.
4. **Create a new bias file.** Start with empty locks and weights of `1.0`:
   ```json
   {
     "weights": {},
     "locks": {}
   }
   ```
5. **Regenerate runs.** Old genomes cannot be loaded. Rerun scenarios using the new CLI and record outputs to fresh directories.
6. **Optional migration aid.** Implement project-specific content migration inside `tools/migrate_v0_9.py` if you need to translate bespoke formats.

Consult `README.md` for command examples and schema references.
