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
5. **Data & configs** — `data/prompts/*.jsonl` host IPIP-derived contrast pairs; `configs/run.*.yaml` store reproducible run configs (population, steps, steering, safety bounds).

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

### Optional: 3D web viewer (prototype)

You can stream live simulation events to a lightweight WebSocket bridge and view a simple 3D layout in your browser.

```bash
python3 -m orchestrator.cli configs/run.small.yaml --mock-model --env research --difficulty 3 --live --viewer
# Then open http://127.0.0.1:8000 in your browser
```

Notes:
- The viewer opens a WebSocket at `ws://127.0.0.1:8765/ws` and serves static assets from `viewer/static/` at `http://127.0.0.1:8000`.
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
├── data/             # IPIP prompt pairs and saved vectors
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
