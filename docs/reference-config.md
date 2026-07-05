# Configuration reference

Full schema for every YAML config file the simulator reads. See
[reference-cli.md](reference-cli.md) for how these interact with CLI flags.

## Run config (`configs/run.*.yaml`)

Loaded as `schemas/run.py::RunConfig`. Top-level shape:

```yaml
run_id: str                    # defaults to "run-<uuid6>" if absent
git_commit: str
model_name: str                # HF model id; ignored under --mock-model/--gemini
population: int                # agent count
steps: int                     # tick count for this run
scenario: str                  # freeform label, not consumed by the CLI
seed: int                      # drives scheduler RNG, base persona sampling

steering:
  enabled: bool                # default true; also gated by --no-steering
  strength: float               # global alpha multiplier ("dose"), default 1.0
  coefficients: {E,A,C,O,N: float}   # base persona trait means before per-agent jitter
  vector_norm: {E,A,C,O,N: float}    # NOT READ by any code path — see note below
  metadata_files:
    personas: path              # -> configs/personas.bigfive.yaml
    vectors: path                # -> configs/steering.layers.yaml

logging:
  db_url: str                  # SQLAlchemy URL (sqlite:/// or postgresql+psycopg://)
  parquet_dir: str              # directory for per-tick Parquet dumps

meta_orchestrator:
  enabled: bool
  playbook_file: path          # -> configs/meta.playbook.yaml (optional; falls back to
                                # a hardcoded default playbook if missing)

safety:
  alpha_clip: float             # ceiling on |alpha| after strength scaling
  toxicity_threshold: float     # declared but NOT READ — governor.py is pure substring
                                 # matching, not threshold-scored classification (placeholder
                                 # for a future classifier per its own docstring)
  governor_backoff: float       # fraction each flagged generation nudges alphas toward 0

inference:
  temperature: float
  top_p: float
  max_new_tokens: int

optimization:
  reflect_every_n_ticks: int    # planner cache lifetime; see reference-modules.md#planner
  use_quantization: bool         # 4-bit quantization (fast config only)
  batch_size: int                 # declared "(future feature)" — not read anywhere
  max_gpu_memory_gb: float
  max_cpu_memory_gb: float
  offload_folder: path

objectives:
  enabled: bool

probes:                         # optional; see configs/probes.yaml below
  ...
```

**`steering.vector_norm` is dead config** — no loader in `orchestrator/cli.py`,
`agents/language_backend.py`, or `steering/hooks.py` reads it. Actual per-layer
vector norms come from `VectorStore`/`.meta.json` metadata, recorded purely as
diagnostic provenance (the *direction* is what gets used; magnitude is always
governed by `alpha`). Don't expect edits to this block to change runtime
behavior.

### Shipped configs compared

| Field | `run.small.yaml` (`debug32`) | `run.medium.yaml` (`study100`) | `run.fast.yaml` (`fast_test`) |
|---|---|---|---|
| `model_name` | Llama-3.1-8B-Instruct | Llama-3-70b-instruct | Llama-3.1-8B-Instruct |
| `population` | 32 | 100 | 32 |
| `steps` | 200 | 500 | 200 |
| `logging.db_url` | sqlite | **postgresql** (needs a running Postgres instance) | sqlite |
| `safety.toxicity_threshold` | 0.4 | 0.35 (stricter) | 0.4 |
| `safety.governor_backoff` | 0.2 | 0.15 (gentler — a fixed fraction compounds over more ticks/agents at scale) | 0.2 |
| `inference.max_new_tokens` | 128 | 160 | **48** (explicit throughput optimization) |
| `optimization.reflect_every_n_ticks` | 10 | 10 | 5 |
| `optimization.use_quantization` | — | — | `true` |

`run.medium.yaml` is the only shipped config that needs a live Postgres
instance reachable at its `db_url` before you run it — `small`/`fast` use
SQLite and need no external service.

## Steering vector metadata (`configs/steering.layers.yaml`)

```yaml
vector_root: ../data/vectors        # resolved relative to this file's directory
defaults:
  layers: [int, ...]                # fallback decoder layers if a trait entry omits `layers`
  prompt_dir: ../data/prompts
  prompt_masking: true               # documentation only — masking is always applied
                                      # whenever prompt_length is passed to SteeringController,
                                      # this flag is not read to decide that
  extraction: caa_ab_normalized       # documentation only
  model: str                          # documentation only — see warning below
  num_hidden_layers: int               # documentation only
  notes: str
traits:
  <TRAIT_CODE>:                        # e.g. E, A, C (single letter)
    name: str
    vector_store_id: str
    prompt_file: path                  # relative to this file's directory
    layers: [int, ...]
    description: str                    # human rationale, documentation only
```

Consumed by `scripts/compute_vectors.sh` (extraction), `scripts/eval_vectors.sh`
(evaluation), and `orchestrator.cli.load_trait_vectors` (runtime loading, via
`steering.metadata_files.vectors` in the run config).

**Live inconsistency to check before you extract vectors**: this file's
`defaults.model` field currently names a 64-layer Qwen2.5-32B model with layer
indices (12/36/60, 16/40/58, 20/44/60) sized for that model, but no code
actually reads `defaults.model` — the real model used is `$MODEL_NAME`
(`scripts/compute_vectors.sh`) or `--model` (`steering.compute_caa`), which
both default to `meta-llama/Llama-3.1-8B-Instruct` (32 layers). Running
`compute_vectors.sh` today with the checked-in YAML and no `MODEL_NAME`
override would ask Llama-3.1-8B for out-of-range layers like 60. The vectors
actually present in `data/vectors/*.npy` are Llama-3.1-8B vectors at
different (in-range) layers than the YAML currently describes. Always pass
an explicit `MODEL_NAME` matching the layer indices you intend to extract, and
treat the YAML's layer/vector-store-id values as aspirational until you've
regenerated vectors from it. See [explanation-known-gaps.md](explanation-known-gaps.md#steering-config-describes-a-model-that-doesnt-match-the-checked-in-vectors).

## Role/reminder playbook (`configs/meta.playbook.yaml`)

```yaml
<environment>:            # research | policy | nav
  <RoleName>:
    planning_hints: [str, ...]
    reminders: [str, ...]
```

Merged on top of a hardcoded `MetaOrchestrator.default_role_playbook()` — today
the file's contents are identical in substance to that default, so editing it
is currently the only way to diverge from the built-in playbook (the code
never regresses to defaults for keys you *do* override).

## Probe schedule (`configs/probes.yaml`)

```yaml
likert:
  cadence: int                  # default cooldown (ticks) between assignments per agent
  questions:
    - id: str
      trait: str                 # informational only; not used for gating
      question: str
      instructions: str
      cadence: int (optional)     # per-question override
behavior:
  cadence: int
  scenarios:
    - id: str
      scenario: str
      instructions: str
      outcomes:
        <label>:
          keywords: [str, ...]     # substring match, case-insensitive
      cadence: int (optional)
```

At most one probe is ever "in flight" per agent. Likert probes are preferred
over behavior probes when both are due on the same tick. Whatever the agent
says on its next turn is scored as the probe's answer — there's no check that
the response actually addresses the probe. See
[explanation-known-gaps.md](explanation-known-gaps.md#probe-preamble-is-injected-twice)
for a related injection bug.

## Persona sampling (`configs/personas.bigfive.yaml`)

```yaml
ranges:
  <trait>: {min, max, distribution, mean, std}
percentiles:
  low, mid, high
sampling:
  strategy: str
  seed: int
  jitter: float
```

Only `sampling.jitter` is actually consumed (`orchestrator.cli._persona_sampling_jitter`,
default `0.2` if missing). `ranges`, `percentiles`, and `sampling.strategy`/`seed`
document an intended richer sampling design that isn't implemented — the
actual per-agent persona is `steering.coefficients[trait] + Uniform(-jitter, jitter)`,
driven by the run config's top-level `seed`, not a Latin-hypercube or
normal-distribution sampler.

## Related

- [reference-cli.md](reference-cli.md) — flags that interact with these configs.
- [explanation-known-gaps.md](explanation-known-gaps.md) — every dead-config-field and doc/code mismatch found across this project, in one place.
