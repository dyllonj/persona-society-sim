# Jacobian Lens toolchain

This directory is a separate Python project. It pins Anthropic's Jacobian Lens
at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` and deliberately does not
share the simulator's Transformers 4.45 environment.

## Fit a lens

From the repository root:

```bash
uv run --project interpretability python -m interpretability.fit_lens \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --wikitext-prompts 100 \
  --source-layers 12,16,20,28,36,40,48,58,60,62 \
  --output-dir artifacts/jacobian_lens/qwen32b-pilot
```

The output directory contains the fitted `lens.pt`, a resumable checkpoint,
the exact fitting texts, and `manifest.json` with model, corpus, dependency,
and artifact hashes.

## Export agent traces

Production events come from the simulator's `inference_events` output. For a
standalone end-to-end GPU validation, capture one structured, CAA-steered agent
decision with the exact model revision and checked-in vectors:

```bash
uv run --project interpretability python -m interpretability.capture_pilot_event \
  --model-revision <immutable-hf-commit> \
  --vector-metadata configs/steering.layers.yaml \
  --output artifacts/jacobian_traces/pilot-event.jsonl
```

The command performs the same per-hook residual smoke test as the simulator,
then writes exact tokens, effective alphas, vector hashes, and a hashed sidecar
manifest. It is a validation fixture, not a replacement for simulation capture.

```bash
uv run --project interpretability python -m interpretability.export_traces \
  --events storage/dumps/fast_test/inference_events \
  --lens artifacts/jacobian_lens/qwen32b-pilot/lens.pt \
  --lens-manifest artifacts/jacobian_lens/qwen32b-pilot/manifest.json \
  --vector-metadata configs/steering.layers.yaml \
  --include-neutral \
  --top-k 10 \
  --output artifacts/jacobian_traces/fast_test.parquet
```

The exporter consumes recorded token IDs directly, reapplies the recorded CAA
vectors only to continuation positions, validates the lens/model/vector hashes,
checks final-layer replay logits, and writes long-form top-k Parquet plus a
hashed trace manifest.
