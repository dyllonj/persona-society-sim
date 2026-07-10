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

## Merge layer-sharded lenses

Fit only missing source layers against the exact same model revision, tokenizer,
corpus, target layer, dtype, and fit settings, then combine the old and new
artifacts without refitting the existing matrices:

```bash
uv run --project interpretability python -m interpretability.merge_lenses \
  --input artifacts/jacobian_lens/qwen32-alltraits-n100 \
  --input artifacts/jacobian_lens/qwen32-missing-layers-n100 \
  --output-dir artifacts/jacobian_lens/qwen32-seven-layers-n100
```

The merger verifies both parent lens hashes and their recorded corpora, rejects
any model, revision, config, dependency, dtype, corpus, target, or fit-setting
mismatch, and only accepts an overlapping layer when its stored tensor is
exactly equal. It atomically publishes a union `lens.pt`, the verified fitting
corpus, and a provenance manifest containing both parent manifest and lens
hashes. The output directory must not already exist.

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

## Report cross-layer trait interactions

Once the lens covers the configured intervention layers, transport every
steering component into the common target-layer basis and measure amplification,
within-trait coherence, and cross-trait overlap:

```bash
uv run --project interpretability python -m interpretability.trait_report \
  --lens artifacts/jacobian_lens/qwen32-seven-layers-n100/lens.pt \
  --lens-manifest artifacts/jacobian_lens/qwen32-seven-layers-n100/manifest.json \
  --vector-metadata configs/steering.layers.yaml \
  --alpha E=0.8 --alpha A=0.5 --alpha C=0.6 \
  --require-complete \
  --output-prefix artifacts/jacobian_reports/qwen32-seven-trait-space
```

The command atomically writes JSON and Markdown reports with artifact hashes,
coverage, `J_layer @ vector_trait`, directional versus isotropic RMS gain, the
full transported-component Gram/cosine matrices, per-trait sums, and the
alpha-weighted combined effect. Omit `--skip-vocab` on a machine capable of
loading the exact manifest-pinned Hugging Face checkpoint to add positive and
negative vocabulary projections for every component and aggregate direction.

## Run the paired live-generation factorial

Build the checked-in condition-blind 60-prompt bundle from the held-out E/A/C
scenarios. Only the scenario enters model-visible text; trait labels and
forced-choice options remain analysis metadata:

```bash
uv run python -m interpretability.build_factorial_prompts \
  --output experiments/factorial_prompts.jsonl
```

The runner rejects explicit personality labels so prompt wording cannot reveal
the intervention arm. Then run all six prespecified conditions:

```bash
uv run --project interpretability python -m interpretability.run_factorial \
  --model-revision <immutable-hf-commit> \
  --prompts experiments/factorial_prompts.jsonl \
  --vector-metadata configs/steering.layers.yaml \
  --output artifacts/jacobian_factorial/live-generations.jsonl
```

The model is loaded once. For each prompt, neutral, E-only, A-only, C-only,
E+A+C, and coordinate-shuffled placebo arms receive the same sampling seed but
generate independent token paths. The placebo preserves each vector's layer,
polarity, coefficient, values, and norm while deterministically permuting its
coordinates. The JSONL records exact prompt/generated token IDs, raw text,
alphas, source and applied vector hashes, and permutation provenance. Its
sidecar manifest hashes the complete output, model configuration, input prompt
file, metadata, vector index, and runner source.

The runner refuses an existing output, progress sidecar, or final manifest by
default. After each prompt's complete six-arm block it atomically replaces the
canonical JSONL, then advances `<output-stem>.progress.json`. If a process is
interrupted, rerun the identical command with `--resume`:

```bash
uv run --project interpretability python -m interpretability.run_factorial \
  --model-revision <immutable-hf-commit> \
  --prompts experiments/factorial_prompts.jsonl \
  --vector-metadata configs/steering.layers.yaml \
  --output artifacts/jacobian_factorial/live-generations.jsonl \
  --resume
```

Resume verifies the complete run specification, prompt/model/vector/code/input
hashes, and every durable event. It accepts only a contiguous prefix of unique,
complete six-arm blocks; corrupt, partial, reordered, or incompatible data
fails closed. Because output is committed before progress, recovery can also
reconcile the single completed block written immediately before a crash. At
most the currently generating prompt block is lost.

Validate the six complete arms for every prompt/seed and render descriptive
paired contrasts:

```bash
uv run python -m interpretability.analyze_factorial \
  --events artifacts/jacobian_factorial/live-generations.jsonl \
  --prompt-metadata experiments/factorial_prompts.jsonl \
  --output-prefix artifacts/jacobian_factorial/analysis
```

The analyzer reports action distributions, structured-action validity, token
path divergence, output lengths, and E/A/C origin-stratum slices. It does not
invent a personality score or call a model judge. A prespecified human or
external rubric can be joined with `--rubric-scores`.
