# How to compute and evaluate steering vectors

For the reasoning behind this pipeline, see
[explanation-steering.md](explanation-steering.md). This doc is purely the
steps to go from prompt files to vectors your simulation runs can load.

## Prerequisites

- A trait's forced-choice prompt file under `data/prompts/<trait>.jsonl` (schema: `data/prompts/schema.py`)
- A GPU-capable environment if extracting from a real model (CPU works but is slow)
- Confirm the model and layers in `configs/steering.layers.yaml`; vectors are model- and layer-specific

## 1. Validate or convert your prompt files

```bash
# If you have legacy situation/positive/negative-style files:
python -m data.prompts.schema convert old_prompts.jsonl data/prompts/newtrait.jsonl

# Always validate before extracting:
python -m data.prompts.schema validate data/prompts/*.jsonl
```

Validation enforces that every record has exactly one `option_a_is_high` /
`option_b_is_high` flag set to `true`.

## 2. Confirm `configs/steering.layers.yaml` matches the intended model

The checked-in metadata targets `Qwen/Qwen2.5-32B-Instruct`, and the extraction
scripts now use its `defaults.model` automatically. The checked-in E/A/C
vectors were extracted for that same model. If you intentionally set
`MODEL_NAME` to another checkpoint, change the layer map and vector-store IDs
too; the runtime will reject artifacts whose model, layer, width, or hashes do
not match.

## 3. Extract vectors

```bash
# All traits declared in configs/steering.layers.yaml
./scripts/compute_vectors.sh

# Just specific traits
TRAITS=A,C ./scripts/compute_vectors.sh

# Deliberate model override (also update the metadata layers and artifact IDs)
MODEL_NAME=<hf-model-id> TRAITS=A ./scripts/compute_vectors.sh
```

This writes `data/vectors/<vector_store_id>_layer<N>.npy` per layer and
`data/vectors/<trait>.meta.json` (SHA256 of the prompt file, model name,
layers used, per-layer norms). Inspect the `.meta.json` to confirm it matches
what you expect.

## 4. Evaluate before trusting the vectors in a simulation run

```bash
TRAIT_ALPHAS=E=0.8,A=0.5,C=0.6 \
INFERENCE_DTYPE=bf16 \
DELTA_THRESHOLD=0.1 SIGN_THRESHOLD=0.55 \
./scripts/eval_vectors.sh
```

This regenerates vectors from the configured model and runs `steering.eval`, writing
`artifacts/steering_eval/report.json` and `.md`. Use `TRAIT_ALPHAS` for the
actual per-trait experimental doses; `STEERING_ALPHA` remains the shared
fallback. The evaluator uses mean conditional log-probability per continuation
token as its primary statistic and also archives summed log-probability. It
tokenizes the prompt, delimiter, and option as one string to preserve the real
token boundary. Set `OVERWRITE=1` only when deliberately replacing an existing
report.

The checked-in E/A/C evaluation files contain 20 items per trait and are
disjoint from the eight extraction items. Verify that separation after any
prompt edit:

```bash
uv run python scripts/split_eval_prompts.py --verify-existing
```

The verifier rejects shared IDs, exact/normalized-text leakage, and highly
similar question stems. These are authored held-out behavioral scenarios, not
an independently normed Big Five instrument, so report them as an internal
generalization check.

`steering.eval` accepts all five Big Five aliases, but the CLI defaults to E/A/C
because only those traits have checked-in extraction prompts, held-out sets,
and Qwen32 vector artifacts. O/N require those data and artifacts before they
can be added to a real evaluation arm.

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
