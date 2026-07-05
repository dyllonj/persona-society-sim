# How to run simulations

Assumes you've done the [getting-started tutorial](tutorial-getting-started.md)
once. This is a task-oriented reference for the run variations you'll actually
want day to day. Full flag semantics live in [reference-cli.md](reference-cli.md).

## Prerequisites

- Repo installed: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`
- For `--gemini` runs: `export GEMINI_API_KEY="..."`
- For `configs/run.medium.yaml`: a reachable Postgres instance matching its `logging.db_url` (small/fast configs need nothing beyond SQLite, which needs no setup)

## Pick an environment and difficulty

```bash
# Research Sprint: gather facts at the library, then submit_report
python3 -m orchestrator.cli configs/run.small.yaml --env research --difficulty 3 --live

# Policy Checklist: fill checklist fields, draft a summary, submit_plan
python3 -m orchestrator.cli configs/run.small.yaml --env policy --difficulty 5 --live

# Navigation + Discovery: visit rooms and scan tokens
python3 -m orchestrator.cli configs/run.small.yaml --env nav --difficulty 6 --live
```

`--difficulty` means something different per environment (target fact count /
required checklist fields / discovery tokens per agent) — see
[reference-cli.md](reference-cli.md#scenario-and-environment).

## Choose a backend

```bash
# No model weights needed — deterministic stub output, fastest iteration loop
python3 -m orchestrator.cli configs/run.small.yaml --mock-model --env research

# Local HF model with activation-based (CAA) steering — the default if you omit both flags
python3 -m orchestrator.cli configs/run.small.yaml --env research

# Gemini API with prompt-based steering
export GEMINI_API_KEY="your_api_key_here"
python3 -m orchestrator.cli configs/run.small.yaml --gemini --env research
```

Before relying on `--gemini` for a persona-adherence study, read
[explanation-known-gaps.md](explanation-known-gaps.md#gemini-persona-steering-silently-no-ops)
— persona steering currently no-ops on this path when driven through the
normal simulation loop.

To run agents with neutral (unsteered) personas for a baseline comparison,
add `--no-steering` to any of the above.

## Watch a run live

Three independent viewing modes — pick one based on where you're running:

```bash
# Any terminal: tick/action/dialogue stream to stdout
python3 -m orchestrator.cli configs/run.small.yaml --live --env research

# Same, without truncating long dialogue (verbose)
python3 -m orchestrator.cli configs/run.small.yaml --live --full-messages --env research

# SSH / low-resource: full-screen ASCII dashboard (pip install rich first)
python3 -m orchestrator.cli configs/run.small.yaml --tui --gemini --env research

# Browser: 3D Three.js viewer at http://127.0.0.1:19123
python3 -m orchestrator.cli configs/run.small.yaml --mock-model --env research --live --viewer
```

`--tui` always wins if combined with `--live` or `--viewer` — see
[reference-modules.md](reference-modules.md#viewer-viewer) for exactly what
each mode shows (the TUI, notably, only logs dialogue, not actions).

## Tune per-tick throughput

```bash
# More/fewer simultaneous agent encounters per tick (default 16)
python3 -m orchestrator.cli configs/run.small.yaml --max-events 32

# Route agent decisions through a worker pool (speeds up concurrent LLM calls,
# not world-state mutation — see reference-cli.md)
python3 -m orchestrator.cli configs/run.small.yaml --decision-workers 4 --batch-decisions-per-encounter

# Move logging/viewer I/O onto background threads for larger runs
python3 -m orchestrator.cli configs/run.medium.yaml --queued-runtime --decision-workers 2
```

Total steps per run come from the config file (`steps: 200`), not a CLI flag.

## Inspecting a completed run

Parquet dumps land under `logging.parquet_dir` from your config (e.g.
`storage/dumps/debug32/`), one file per tick per table — see
[reference-data-schema.md](reference-data-schema.md) for the full table list.

```bash
# Aggregate logs into metric snapshots (takes a run ID, not a path — it
# builds storage/dumps/<run_id> itself)
./scripts/analyze_run.sh debug32

# Validate persona steering actually varied across agents/messages
python3 scripts/verify_steering.py --messages-dir storage/dumps/debug32/messages

# Visualize steering, cognitive traces, and graph/macro metrics
python3 scripts/analyze_simulation.py --dump-dir storage/dumps/debug32
```

If you're upgrading a dump directory created before the `trade` action was
removed, run the migration script first — see
[howto-migrate-legacy-data.md](howto-migrate-legacy-data.md).

## Related

- [tutorial-getting-started.md](tutorial-getting-started.md) — first run, step by step.
- [howto-compute-steering-vectors.md](howto-compute-steering-vectors.md) — building the vectors these runs steer with.
- [reference-cli.md](reference-cli.md) — every flag in detail.
- [reference-config.md](reference-config.md) — the YAML configs referenced above.
