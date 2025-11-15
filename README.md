# Persona Society Sim

Social town simulator where 30–300 activation-steered LLM agents live, converse, collaborate, and produce emergent dynamics. Agents use persona steering vectors (Contrastive Activation Addition) instead of prompt-only roles, combining Smallville-inspired memory loops with lightweight world mechanics and measurement harnesses.

## Project scope
- Build an open-world town loop (observation → reflection → planning → action) with a scheduler, social graph, and light economy/institutions.
- Compute Big-Five steering vectors via CAA/ActAdd from public IPIP items; inject them during inference for dose-controlled personalities.
- Run comparative studies vs. prompt-only roleplay to test persona stability, social-network outcomes, capability retention, and macro dynamics.
- Capture rich telemetry (messages, actions, steering states, graph snapshots, metrics) for analysis and paper-ready artifacts.

## Architecture snapshot
1. **Steering vectors** — `steering/compute_caa.py` builds trait vectors from JSONL prompt pairs and saves per-layer `.npy` blobs; `steering/hooks.py` adds them to model residuals with per-agent coefficients.
2. **Agents & memory** — `agents/agent.py` wires the perceive→reflect→plan loop; `agents/memory.py` implements observation/reflection/planning stores inspired by Generative Agents; `agents/planner.py` converts goals + world state into next actions or utterances.
3. **World & scheduler** — `env/world.py`, `env/actions.py`, and `env/economy.py` encode the text town, actions, and markets; `orchestrator/scheduler.py` samples encounters; `orchestrator/runner.py` executes multi-agent ticks and logs outputs.
4. **Storage & metrics** — `schemas/*.py` define Pydantic models mirrored in SQL; `storage/db.py` wraps SQLite/Postgres; `storage/log_sink.py` streams logs to DB + Parquet; `metrics/*.py` compute graph, cooperation, and polarization metrics for notebooks in `metrics/` and reports in `docs/`.
5. **Instrumentation & probes** — every `ActionLog` now carries cognitive traces (reflection summaries, plan suggestions, prompt hashes) so persona steering can be audited; `metrics/graphs.py` + `metrics/social_dynamics.py` emit graph snapshots and macro metrics each tick; `metrics/tracker.py` joins persona coefficients and per-message steering alphas; the ProbeManager injects self-report + behavioral probes; research telemetry (facts, citations, report grades) is stored in first-class tables for downstream analysis.
6. **Data & configs** — `data/prompts/*.jsonl` host IPIP-derived multiple-choice contrast pairs; `configs/run.*.yaml` store reproducible run configs (population, steps, steering, safety bounds).

## Getting started
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# dry-run without heavyweight models (Research Sprint default)
python3 -m orchestrator.cli configs/run.small.yaml --mock-model --env research --difficulty 3

# run with HF weights + steering vectors (Research Sprint)
python3 -m orchestrator.cli configs/run.small.yaml --env research --difficulty 3

# follow live logs with colored, truncated output
python3 -m orchestrator.cli configs/run.small.yaml --live --env research

# disable truncation when tailing live output (may be very verbose)
python3 -m orchestrator.cli configs/run.small.yaml --live --full-messages --env research

pytest  # optional
```

### Simulation parameters

**Events per tick:**
Control the maximum number of agent encounters/interactions per simulation tick with `--max-events` (default: 16):
```bash
python3 -m orchestrator.cli configs/run.small.yaml --max-events 32
```

**Total steps:**
The total number of simulation ticks is configured in your YAML config file (default: 200):
```yaml
# configs/run.small.yaml
steps: 200
```

### Persona prompt schema & CLI

`data/prompts/*.jsonl` now use a forced-choice format so each record contains a shared stem plus two contrasting options. The high/low answers are flagged explicitly so downstream tooling knows which continuation represents a stronger trait expression:

```json
{
  "id": "A01",
  "question_text": "A teammate makes a mistake that slowed the project.",
  "option_a": "I address it gently, focus on fixes, and offer help.",
  "option_b": "I criticize them bluntly and assign blame.",
  "option_a_is_high": true,
  "option_b_is_high": false
}
```

Use `python -m data.prompts.schema validate data/prompts/*.jsonl` to ensure every file labels exactly one high-trait answer per stem. If you have legacy `situation`/`positive`/`negative` JSONL items you can convert them with `python -m data.prompts.schema convert old.jsonl new.jsonl`. When authoring new contrast items, keep the stem trait-agnostic, provide two concise behavioral options, and flag whichever option represents the higher expression of the trait with `*_is_high = true`.

### Vector metadata, normalization, and masking

- `steering/compute_caa.py` encodes both options for every prompt, subtracts `high - low` per decoder layer, and normalizes the resulting residual before saving it to `data/vectors/<trait>.npy`. The metadata file that sits beside every `.npy` artifact records the SHA256 of the prompt file, the model name, the layer list used during extraction, and the norm of every stored vector.
- `configs/steering.layers.yaml` is the new single source of truth for vector metadata. It specifies the vector root on disk, per-trait `vector_store_id` values, training prompt files, and the decoder layers you want to activate. `scripts/compute_vectors.sh` reads this file so there is no longer a hidden `[12, 16, 20]` default; update the YAML whenever you add a trait or sweep different layer bands.
- Runtime steering applies a `steering.strength` multiplier from your run config (`steering.strength: 0.8` globally scales all trait coefficients) and uses prompt-aware masking to ensure the added residuals only touch generated continuations. Prompt tokens remain untouched so system prompts, memories, and instructions are preserved regardless of how aggressively you steer the completion.

### Steering evaluation harness

Run `./scripts/eval_vectors.sh` to regenerate vectors, evaluate them with `steering.eval`, and capture a JSON + Markdown report under `artifacts/steering_eval/`. The harness:

1. Regenerates vectors with the metadata-aware loader so the right layers and vector-store IDs are selected.
2. Scores held-out prompt files (use the `_eval.jsonl` suffix) with and without steering to compute accuracy deltas and log-probability gaps.
3. Surfaces directional-fidelity checks (`delta_threshold`, `sign_threshold`) and optional transcript samples to sanity-check qualitative behavior.

Set `STEERING_ALPHA` before running the script to mirror the `steering.strength` used in your simulation configs so the evaluation curve matches in-sim usage.

### Migration notes

The v2 steering pipeline requires A/B prompt files, metadata-aware vector extraction, and re-running the evaluation harness before you launch new simulations. Follow [`docs/migration.md`](docs/migration.md) for a checklist that covers converting legacy prompt schema files, regenerating vectors, validating them, and updating your run configs.

### Optional: 3D web viewer (prototype)

You can stream live simulation events to a lightweight WebSocket bridge and view a simple 3D layout in your browser.

```bash
python3 -m orchestrator.cli configs/run.small.yaml --mock-model --env research --difficulty 3 --live --viewer
# Then open http://127.0.0.1:19123 in your browser
```

Notes:
- The viewer opens a WebSocket at `ws://127.0.0.1:8765/ws` and serves static assets from `viewer/static/` at `http://127.0.0.1:19123`.
- Agents are rendered as colored spheres around radial “room” anchors; colors are mapped from persona traits.
- This is a minimal prototype to validate the streaming API; it’s designed to be replaced by a full engine (Godot/Unity/Unreal) later.

### Live console options

Use `--live` to see tick-by-tick actions and dialogues streamed to the terminal. Output is truncated to keep multi-agent runs
readable, but you can disable truncation with `--full-messages` (pairs well with log tailing or when debugging a single agent).
Combine with `--no-color` if your terminal cannot render ANSI sequences.

Key scripts:
- `scripts/compute_vectors.sh` — run CAA extraction for all traits.
- `scripts/run_small.sh` — launch a 32-agent, 200-step smoke test.
- `scripts/analyze_run.sh` — aggregate logs into metric snapshots.
- `scripts/verify_steering.py` & `scripts/analyze_simulation.py` — validate persona application and visualize steering, cognitive traces, and the new graph/macro metrics dumps.

### Instrumentation & probes
- **Cognitive traces**: `agents/agent.py` caches reflection summaries, planner suggestions, prompt text, and prompt hashes per action, and `orchestrator/runner.py` writes them into `ActionLog` + Parquet for replay/debugging without re-running the LLM.
- **Graph & macro metrics**: every tick `SimulationRunner` feeds per-action edges into `metrics/graphs.py` and aggregates economy/conflict data via `metrics/social_dynamics.py`, producing `graph_snapshot` + `metrics_snapshot` tables plus Parquet dumps (`storage/log_sink.py`).
- **Persona-aware tracker**: `metrics/tracker.py` now registers each agent’s baseline `PersonaCoeffs` and ingests `MsgLog.steering_snapshot` values so efficiency/collab stats can be sliced by trait bands or alpha magnitudes.
- **Probe scheduling**: the ProbeManager (see `configs/probes.yaml`) injects periodic self-report questionnaires and scripted behavioral probes that log the injected prompt bundle plus Likert/rubric scores to the new `probe_log`/`behavior_probe_log` tables, giving direct coverage of RQ1 behavioral adherence.
- **Structured research telemetry**: research actions emit `ResearchFactLog`, `CitationLog`, and `ReportGradeLog` records instead of opaque JSON strings so fact coverage, citation diversity, and grading drift can be analyzed per persona trait.

See `AGENTS.md` for a deeper dive into the agent loop, probe lifecycle, and how the new telemetry hooks tie into persona steering.

## Milestones
1. **M1 — Persona vector library**: implement CAA pipeline, validate dose-response & capability checks.
2. **M2 — Town core loop**: finalize scheduler, memory, and action execution with logging + safety guardrails.
3. **M3 — First comparative study**: run N=100, 500 steps; evaluate persona adherence, drift, social graph, productivity.
4. **M4 — Ablations & scale**: layer/coeff sweeps, memory ablations, event richness toggles, N=300 scaling trials.

## Repo layout
```
persona-society-sim/
├── agents/           # agent loop, memory, planning
├── configs/          # YAML configs for runs/personas/steering layers
├── data/             # IPIP multiple-choice prompt pairs and saved vectors
├── docs/             # design notes, evaluation plan
├── env/              # world model, actions, economy, institutions
├── metrics/          # network + social dynamics analytics
├── orchestrator/     # scheduler + runner
├── schemas/          # pydantic models mirroring DB schemas
├── steering/         # CAA computation, hooks, vector store
├── storage/          # DB adapters, parquet dumps
├── scripts/          # helper shell scripts
├── tests/            # smoke tests for hooks, memory, env
└── pyproject.toml
```

## Research questions & evaluation hooks
- **RQ1 — Personality controllability**: log persona adherence metrics (self-report probes, behavioral probes) and compare CAA vs prompt-only loops.
- **RQ2 — Social structure/outcomes**: compute homophily, polarization, productivity metrics at regular ticks.
- **RQ3 — Method validity**: capability probes to ensure CAA preserves base performance vs prompt steering, plus safety governor logs for alpha backoff events.
- **RQ4 — Macro dynamics**: seed opinion topics and measure homophily/polarization trajectories across populations 30/100/300.

For detailed evaluation protocols, see `docs/eval.md`.
### Experiment environments

You can select among a few RL-style environments using `--env` and a simple difficulty parameter with `--difficulty`:

- `--env research` (Research Sprint):
  - Goal: collect facts at the library and `submit_report`.
  - Difficulty: target facts count (used for dataset sizing; grader reports correctness regardless).
  - Example: `python3 -m orchestrator.cli configs/run.small.yaml --env research --difficulty 3 --live`

- `--env policy` (Policy Checklist):
  - Goal: fill checklist fields, draft a summary, and `submit_plan` for compliance.
  - Difficulty: number of required fields (actions `fill_field`, `propose_plan`, `submit_plan`).
  - Example: `python3 -m orchestrator.cli configs/run.small.yaml --env policy --difficulty 5 --live`

- `--env nav` (Navigation + Discovery):
  - Goal: visit unique rooms and `scan` tokens while coordinating coverage.
  - Difficulty: discovery tokens required per agent (`scan` actions consume finite room tokens).
  - Example: `python3 -m orchestrator.cli configs/run.small.yaml --env nav --difficulty 6 --live`

Notes:
- Research Sprint uses `research`, `cite`, and `submit_report` plus a grader tied to the reference corpus.
- Policy runs now track checklist progress in-world. Agents must complete unique `fill_field` actions, optionally `propose_plan`, and only a successful `submit_plan` counts toward their objective.
- Navigation runs mint a limited pool of tokens per room. Agents must move around to `scan` fresh rooms; repeat scans that fail to collect a token no longer advance their goals.
