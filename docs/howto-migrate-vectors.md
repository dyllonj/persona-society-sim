# A100 runbook: migrate steering vectors to the YAML config

Use this when recomputing CAA vectors for the model declared in
`configs/steering.layers.yaml`. The config is now the source of truth for
`defaults.model`, `defaults.layers`, `defaults.num_hidden_layers`, per-trait
prompt files, layer lists, and `vector_store_id` values.

## 1. Prepare the machine

Run on an A100 80GB node for the checked-in Qwen2.5-32B config.

```bash
cd /path/to/persona-society-sim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/fast/hf-cache
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
```

Authenticate to Hugging Face before the first run if the configured model
requires gated access:

```bash
huggingface-cli login
```

## 2. Validate config and prompts

```bash
python -m data.prompts.schema validate data/prompts/*.jsonl
python - <<'PY'
import yaml
from pathlib import Path

path = Path("configs/steering.layers.yaml")
cfg = yaml.safe_load(path.read_text())
print(cfg["defaults"]["model"])
print(cfg["defaults"]["num_hidden_layers"])
for trait, spec in cfg["traits"].items():
    print(trait, spec["vector_store_id"], spec.get("layers") or cfg["defaults"]["layers"])
PY
```

The extraction CLI will fail if the loaded model reports a different
`model.config.num_hidden_layers` than the YAML declares, or if a requested
layer index is outside the model's decoder layer range.

## 3. Compute vectors

For all traits:

```bash
VECTOR_METADATA=configs/steering.layers.yaml ./scripts/compute_vectors.sh
```

For a subset:

```bash
TRAITS=E,A VECTOR_METADATA=configs/steering.layers.yaml ./scripts/compute_vectors.sh
```

Only use `MODEL_NAME=...` when intentionally overriding the YAML model for an
experiment. If you do, also pass compatible `--layers`/config values; vectors
do not transfer across model families or layer counts.

To write to a scratch vector root without editing the YAML:

```bash
VECTOR_ROOT=/mnt/fast/qwen32b-vectors ./scripts/compute_vectors.sh
```

## 4. Run the steerability smoke gate

Before committing to the full extraction, run a small model-backed activation
probe over the YAML prompt metadata. This loads the model once, samples a small
set of A/B pairs per trait, computes high-minus-low activation diffs at the
configured layers, and reports `preliminary_steerability` as mean cosine
agreement with the mean diff direction.

```bash
python -m steering.steerability_smoke_test \
  --config configs/steering.layers.yaml \
  --traits E,A,C \
  --sample-pairs 10 \
  --pass-threshold 0.3 \
  --warn-threshold 0.2 \
  --min-count 5 \
  --json-output artifacts/steering_smoke/pre_extraction_gate.json
```

Any trait with all sampled layers below the gate exits non-zero. Treat this as
a stop signal before spending A100 time on full vector extraction.

If your extraction job saved activation-diff probes as JSON/JSONL/NPY, gate
them without loading a model:

```bash
python -m steering.steerability_smoke_test \
  --diffs artifacts/steering_smoke/E_layer36_diffs.json \
  --pass-threshold 0.8 \
  --warn-threshold 0.6 \
  --min-count 20 \
  --json-output artifacts/steering_smoke/E_layer36_gate.json
```

For vector diffs, include `reference_vector` in the JSON payload or pass
`--reference-vector-file path/to/vector.npy`. A `FAIL` exits non-zero; `WARN`
and `PASS` exit zero.

## 5. Evaluate and archive

```bash
STEERING_ALPHA=1.0 DELTA_THRESHOLD=0.1 SIGN_THRESHOLD=0.55 ./scripts/eval_vectors.sh
```

Archive the following with the experiment:

- `configs/steering.layers.yaml`
- `data/vectors/*.meta.json`
- `data/vectors/index.jsonl`
- `artifacts/steering_eval/report.json`
- any `artifacts/steering_smoke/*_gate.json` files

Before using the vectors in a simulation run, confirm the run config points
`steering.metadata_files.vectors` at the same YAML file used for extraction.
