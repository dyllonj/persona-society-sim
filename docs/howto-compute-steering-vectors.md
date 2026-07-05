# How to compute and evaluate steering vectors

For the reasoning behind this pipeline, see
[explanation-steering.md](explanation-steering.md). This doc is purely the
steps to go from prompt files to vectors your simulation runs can load.

## Prerequisites

- A trait's forced-choice prompt file under `data/prompts/<trait>.jsonl` (schema: `data/prompts/schema.py`)
- A GPU-capable environment if extracting from a real model (CPU works but is slow)
- Know which model you're extracting from — see the warning below before trusting `configs/steering.layers.yaml` as-is

## 1. Validate or convert your prompt files

```bash
# If you have legacy situation/positive/negative-style files:
python -m data.prompts.schema convert old_prompts.jsonl data/prompts/newtrait.jsonl

# Always validate before extracting:
python -m data.prompts.schema validate data/prompts/*.jsonl
```

Validation enforces that every record has exactly one `option_a_is_high` /
`option_b_is_high` flag set to `true`.

## 2. Check `configs/steering.layers.yaml` matches the model you'll actually use

**Read this before running the next step.** The checked-in
`configs/steering.layers.yaml` currently documents a Qwen2.5-32B-Instruct
layout (64 layers, indices like 60) that the extraction scripts do **not**
default to — they default to `meta-llama/Llama-3.1-8B-Instruct` (32 layers)
unless you override `MODEL_NAME`. If you extract against the file as-is
without an explicit model override, you risk asking a 32-layer model for
layer 60, which doesn't exist. Either:

- Pass `MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct` explicitly and use layer
  indices that exist in that model (the vectors already in `data/vectors/`
  were extracted this way), or
- Update the YAML's layer indices to match whatever model you're actually
  targeting before running extraction.

See [explanation-known-gaps.md](explanation-known-gaps.md#steering-config-describes-a-model-that-doesnt-match-the-checked-in-vectors)
for the full detail.

## 3. Extract vectors

```bash
# All traits declared in configs/steering.layers.yaml
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct ./scripts/compute_vectors.sh

# Just specific traits
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct TRAITS=A,C ./scripts/compute_vectors.sh
```

This writes `data/vectors/<vector_store_id>_layer<N>.npy` per layer and
`data/vectors/<trait>.meta.json` (SHA256 of the prompt file, model name,
layers used, per-layer norms). Inspect the `.meta.json` to confirm it matches
what you expect.

## 4. Evaluate before trusting the vectors in a simulation run

```bash
STEERING_ALPHA=1.0 DELTA_THRESHOLD=0.1 SIGN_THRESHOLD=0.55 ./scripts/eval_vectors.sh
```

This regenerates vectors (same as step 3, so it will also hit the model/layer
mismatch above if unset) and runs `steering.eval`, writing
`artifacts/steering_eval/report.json` and `.md`. Set `STEERING_ALPHA` to the
same value as `steering.strength` in the run config you plan to use, so the
evaluated dose matches what agents will actually receive.

**Caveat**: no `<trait>_eval.jsonl` held-out files exist in this repo today,
so the harness silently falls back to evaluating on the training prompts
(logged as a warning). Treat current accuracy/sign-consistency numbers as
optimistic until you add real held-out files — see
[explanation-known-gaps.md](explanation-known-gaps.md#steeringeval-currently-evaluates-on-training-data-not-held-out-data).

`steering.eval`'s `--traits` currently only supports `extraversion`,
`agreeableness`, `conscientiousness` by default (its `TRAIT_ALIASES` table
doesn't cover Openness/Neuroticism yet) — extending it is a small code change
in `steering/eval.py`, not a config change.

## 5. (Optional) Sweep layers empirically

```bash
python -m steering.layer_sweep <trait> <held_out_prompt_file> data/vectors --model <model_name>
```

Evaluates the trait's existing per-layer vectors against a held-out prompt
file (dot-product sign agreement vs. chance), and writes the winning layers
back into `data/vectors/<trait>.meta.json` as `preferred_layers`. The runtime
loader prefers these automatically on the next simulation run — no other
config file needs updating.

## 6. Point a run config at the vectors

Confirm your run config's `steering.metadata_files.vectors` points at
`configs/steering.layers.yaml`, then do a quick mock-model check that
everything loads before spending time on a full HF run:

```bash
python3 -m orchestrator.cli configs/run.small.yaml --mock-model --env research
```

Then run for real:

```bash
python3 -m orchestrator.cli configs/run.small.yaml --env research
```

## Re-running after prompt or layer changes

Any edit to a trait's prompt file or its `configs/steering.layers.yaml` entry
should be followed by steps 3-4 again, and the resulting `.meta.json` diff
checked into your experiment log so later analysis can confirm which
vector-store ID and norms were actually used.

## Related

- [explanation-steering.md](explanation-steering.md) — why this pipeline is shaped this way.
- [reference-config.md](reference-config.md#steering-vector-metadata-configssteeringlayersyaml) — full `steering.layers.yaml` schema.
- [howto-migrate-legacy-data.md](howto-migrate-legacy-data.md) — converting old prompt schemas and removing legacy trade records.
