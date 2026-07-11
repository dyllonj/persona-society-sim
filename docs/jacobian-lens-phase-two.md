# Jacobian Lens phase-two runbook

This runbook turns the three-layer Qwen32 pilot into two distinct experiments:

1. a model-level seven-component Jacobian-space analysis; and
2. behavioral interventions with independently generated token paths.

The seven components are E@36, A@16/40/58, and C@20/44/62. All work uses the
immutable `Qwen/Qwen2.5-32B-Instruct` revision
`5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`.

The fit, exact seven-layer merge, trait report, replay, held-out evaluation,
and live factorial are complete. See
[jacobian-lens-phase-two-results.md](jacobian-lens-phase-two-results.md) for
artifact hashes and findings. The society study in section 6 is the remaining
confirmatory stage.

## Implemented controls

- E/A/C each have 20 held-out evaluation items, disjoint from the eight vector
  extraction items. `scripts/split_eval_prompts.py --verify-existing` checks
  IDs, hashes, normalized content, and near-duplicate stems.
- `interpretability.merge_lenses` combines disjoint source-layer artifacts
  only after matching model, tokenizer, corpus, target, fit configuration,
  dtype, dependency, and parent hashes.
- `interpretability.trait_report` computes every `J_layer @ vector_trait`,
  transported gain, the full seven-component Gram/cosine matrices, per-trait
  sums, the alpha-weighted combined direction, and exact-model vocabulary
  projections.
- `interpretability.run_factorial` pairs prompts and sampling seeds across
  neutral, E-only, A-only, C-only, E+A+C, and coordinate-permuted placebo arms
  while allowing each arm to generate its own continuation.
- `scripts/run_society_study.py` materializes immutable configs and refuses
  real execution without explicit GPU-hour and cost caps.
- `scripts/analyze_society_study.py` aggregates ticks, agents, and actions
  within each world before calculating uncertainty across simulation runs.

These controls establish provenance and internal validity. The authored
held-out items are not an independently normed psychometric instrument, and
token-path divergence is not itself a personality-effect measurement.

## 1. Verify held-out data

```bash
uv run python scripts/split_eval_prompts.py --verify-existing
```

This must report 8 train and 20 evaluation items for each E/A/C trait with no
overlap.

## 2. Fit only the four missing matrices

Use the same fitting corpus recovered with the pilot. Do not sample a new
corpus, change prompt order, or change numerical settings.

Do not let `uv` sync or upgrade this run. The current interpretability lockfile
can resolve a newer Torch build, while the pilot manifest records and the
merger requires `torch==2.7.1+cu128`, `transformers==5.13.0`, and
`jlens==0.1.0` from commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. Point `JACOBIAN_PYTHON` at the
preserved pilot-compatible virtual environment and verify it before fitting:

```bash
export JACOBIAN_PYTHON=/path/to/pilot-compatible-venv/bin/python

"$JACOBIAN_PYTHON" - <<'PY'
import importlib.metadata
import torch
import transformers

assert torch.__version__ == "2.7.1+cu128"
assert transformers.__version__ == "5.13.0"
assert importlib.metadata.version("jlens") == "0.1.0"
print(torch.__version__, transformers.__version__, importlib.metadata.version("jlens"))
PY
```

The exact dependency versions are part of lens identity, not incidental
environment details. Never relax the merger's compatibility check to get past
a mismatch.

```bash
"$JACOBIAN_PYTHON" -m interpretability.fit_lens \
  --model-id Qwen/Qwen2.5-32B-Instruct \
  --model-revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
  --prompts artifacts/jacobian_lens/qwen32-alltraits-n100/fit_prompts.jsonl \
  --corpus-name wikitext-103-v1 \
  --output-dir artifacts/jacobian_lens/qwen32-missing-layers-n100-refit \
  --source-layers 16,20,40,44 \
  --target-layer 63 \
  --max-prompts 100 \
  --max-seq-len 128 \
  --skip-first 16 \
  --dim-batch 4 \
  --checkpoint-every 5 \
  --dtype bf16
```

The existing E@36, A@58, and C@62 matrices are not recomputed.

## 3. Merge and report the seven components

```bash
"$JACOBIAN_PYTHON" -m interpretability.merge_lenses \
  --input artifacts/jacobian_lens/qwen32-alltraits-n100 \
  --input artifacts/jacobian_lens/qwen32-missing-layers-n100-refit \
  --output-dir artifacts/jacobian_lens/qwen32-seven-layers-n100

"$JACOBIAN_PYTHON" -m interpretability.trait_report \
  --lens artifacts/jacobian_lens/qwen32-seven-layers-n100/lens.pt \
  --lens-manifest artifacts/jacobian_lens/qwen32-seven-layers-n100/manifest.json \
  --vector-metadata configs/steering.layers.yaml \
  --alpha E=0.8 --alpha A=0.5 --alpha C=0.6 \
  --require-complete \
  --output-prefix artifacts/jacobian_reports/qwen32-seven-trait-space
```

Do not use `--skip-vocab` for the final report. Vocabulary projection must use
the exact manifest-pinned model checkpoint.

## 4. Score the held-out forced-choice sets

Use the already checked-in Qwen32 vectors. Do not regenerate them during the
evaluation run:

```bash
SKIP_VECTOR_REGEN=1 \
MODEL_REVISION=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
TOKENIZER_REVISION=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
TRAIT_ALPHAS=E=0.8,A=0.5,C=0.6 \
INFERENCE_DTYPE=bf16 \
MEASURE_BLEED=1 \
ARTIFACT_DIR=artifacts/steering_eval/qwen32-heldout-v1 \
./scripts/eval_vectors.sh
```

Archive prompt-level gaps as well as aggregate accuracy, directional
improvement, sign consistency, anti-steerable fraction, and per-sample
variance. Mean conditional log-probability per continuation token is primary;
summed log-probability is secondary. The report's provenance block and content
hash bind the result to the BF16 runtime and every prompt/vector/config input.
This is a forced-choice intervention check, not the behavioral factorial.

## 5. Run and analyze the live factorial

The checked-in `experiments/factorial_prompts.jsonl` contains 60 neutral
structured-action scenarios, 20 from each held-out origin stratum.

```bash
"$JACOBIAN_PYTHON" -m interpretability.run_factorial \
  --model-revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
  --tokenizer-revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd \
  --prompts experiments/factorial_prompts.jsonl \
  --alphas E=0.8,A=0.5,C=0.6 \
  --base-seed 1701 \
  --placebo-seed 2909 \
  --max-new-tokens 96 \
  --output artifacts/jacobian_factorial/live-generations.jsonl

uv run python -m interpretability.analyze_factorial \
  --events artifacts/jacobian_factorial/live-generations.jsonl \
  --prompt-metadata experiments/factorial_prompts.jsonl \
  --output-prefix artifacts/jacobian_factorial/analysis
```

The factorial runner checkpoints each complete six-arm prompt block. It refuses
pre-existing artifacts unless `--resume` is explicit; a resume invocation must
otherwise use the identical command and environment. The progress sidecar
hashes the complete run specification and durable JSONL prefix, and recovery
fails closed on partial blocks or any prompt, model, vector, code, input, or
sampling mismatch.

Primary diagnostic outputs are structured-action validity, action choice,
length, and exact token-path divergence relative to neutral. Confirmatory
trait claims require a rubric specified before inspecting generations; provide
its JSONL/CSV scores with `--rubric-scores`.

## 6. Materialize the replicated society study

The matrix contains six arms, five paired world seeds, 30 agents, and 100
ticks: 30 independent simulation runs in total.

```bash
uv run python scripts/run_society_study.py \
  --matrix experiments/society_study/matrix.yaml \
  --output-root artifacts/society_study/caa-factorial-v1 \
  --dry-run \
  --hourly-rate-usd <quoted-fixed-rate>
```

The current transparent planning assumption is 90,000 generations and 50
GPU-hours. It is not a benchmark. Replace `seconds_per_generation` in the
matrix with measured throughput from the exact runtime before authorizing the
full study.

Real execution requires both an explicit execution gate and a sufficient
budget cap:

```bash
uv run python scripts/run_society_study.py \
  --matrix experiments/society_study/matrix.yaml \
  --output-root artifacts/society_study/caa-factorial-v1 \
  --execute --resume \
  --max-estimated-gpu-hours <authorized-hours> \
  --hourly-rate-usd <quoted-fixed-rate> \
  --max-estimated-cost-usd <authorized-dollars>

uv run python scripts/analyze_society_study.py \
  --matrix experiments/society_study/matrix.yaml \
  --output-root artifacts/society_study/caa-factorial-v1
```

The placebo society arm is a seeded trait-label derangement: every trait gets
a different real trait vector. It is an active control for label specificity,
not the coordinate-permuted null used by the live factorial.

## Compute policy

- Use fixed-price on-demand instances only. Interruptible/preemptible/bid
  instances are prohibited.
- Query the current fixed hourly rate immediately before creation.
- Never place provider credentials in commands, files, manifests, or logs.
- Persist checkpoints and logs off-instance throughout fitting.
- Enforce a wall-clock timeout and destroy the instance in a `finally` path.
- Recover and hash every artifact before teardown.
- The full society matrix needs separate funding approval when its measured
  estimate exceeds the current experiment balance.

## Interpretation order

1. Treat the seven-component cosine/Gram matrix as model-level mechanism
   evidence.
2. Use the live factorial to test whether those directions change independent
   generated paths.
3. Use replicated worlds, not action rows, for society-level uncertainty.
4. If A/C overlap survives all three levels, characterize it as a candidate
   shared downstream careful/cooperative task direction, not proof that the
   psychological traits are identical.
