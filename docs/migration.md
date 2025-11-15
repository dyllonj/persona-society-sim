# Steering pipeline migration (v2)

The latest steering release replaces ad-hoc layer lists and legacy prompt schemas with an A/B dataset format, normalized extraction pipeline, and metadata-aware loaders. Follow this checklist before launching a new study so every run references the same artifacts.

## 1. Convert prompt files to the A/B schema
- Run `python -m data.prompts.schema convert old_prompts.jsonl new_prompts.jsonl` for every trait file that still uses the `situation`/`positive`/`negative` keys.
- Validate the result with `python -m data.prompts.schema validate data/prompts/*.jsonl` to ensure each record contains a single `*_is_high = true` flag.

## 2. Declare metadata in `configs/steering.layers.yaml`
- Set `vector_root` to the directory where `.meta.json` + `.npy` bundles will be saved.
- Under `traits`, provide `vector_store_id`, `prompt_file`, and the decoder `layers` you want for each trait. This file is now the single source of truth for extraction and runtime loading, so the removed `[12, 16, 20]` fallback cannot surprise you.
- Optional defaults (prompt_dir, notes, masking guidance) keep the file succinct while still documenting intent for reviewers.

## 3. Regenerate normalized vectors
- Run `./scripts/compute_vectors.sh` (optionally pass `TRAITS=A,C` to limit the run). The script reads the metadata file, resolves prompt paths, and calls `steering.compute_caa` with the declared layers so vectors are saved with the correct `vector_store_id`.
- Inspect `data/vectors/<trait>.meta.json` to confirm the stored `preferred_layers`, SHA256 hash of the prompt file, and recorded norms match expectations.

## 4. Evaluate vectors before launching sims
- Execute `./scripts/eval_vectors.sh` after setting `STEERING_ALPHA` to the same value as `steering.strength` in your run config. The harness regenerates vectors, runs `steering.eval` on the held-out `_eval.jsonl` prompts, and emits JSON + Markdown reports.
- Use `DELTA_THRESHOLD` and `SIGN_THRESHOLD` environment variables to require minimum accuracy/log-prob improvements. Treat failures as a blocker until prompts or layer selections are updated.

## 5. Update run configs
- Ensure `steering.strength` captures your intended alpha scaling and that `metadata_files.vectors` points to `configs/steering.layers.yaml`.
- Keep `metadata_files.personas` in sync so `_persona_sampling_jitter` matches the new persona sampling plan.
- Re-run `python3 -m orchestrator.cli ... --mock-model` once to verify trait vectors load from the metadata-aware loader before moving to full HF runs.

## 6. Re-run extraction/evaluation when prompts or layers change
- Any edits to the A/B prompt files or `configs/steering.layers.yaml` should be followed by Steps 3–4 to regenerate normalized vectors and refresh the evaluation artifact.
- Check the git diff of `data/vectors/*.meta.json` into your experiment logs so future analysts can confirm which vector-store IDs and norms were used.
