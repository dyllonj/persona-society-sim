# Tutorial: your first simulation

You'll run a small multi-agent simulation with mock (no-GPU-needed) output,
watch persona-steered agents act in real time, and see the structured logs it
leaves behind — no model weights or API keys required.

## What you'll need

- Python 3.11+ and a virtual environment
- Nothing else — this tutorial uses `--mock-model`, which needs no GPU, no Hugging Face weights, and no API key

## Step 1: Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

python3 -m orchestrator.cli configs/run.small.yaml --mock-model --env research --difficulty 3 --live
```

Within a second you'll see live tick output like this:

```
================================================================================
TICK 0 | 16 events scheduled
================================================================================
  ✓ agent-007 RESEARCH doc:doc1 facts:1 prompt#4d47cc42
    💬 agent-007 @ library
       [mock tokens=128] Persona[A:+0.34, C:+0.51, E:+0.63, N:-0.17, O:+0.08] responds to: agent-007's response:
       [276→128 tokens] [A:+0.3, C:+0.5, E:+0.6, N:-0.2]
```

Each line is one agent's action for that tick. The `Persona[A:..., C:..., E:...]`
block shows that agent's current steering coefficients — this is
`--mock-model` standing in for a real language model so you can see the
simulation mechanics (scheduling, actions, persona coefficients) without
waiting on GPU inference.

Let it run for a bit, then stop it with `Ctrl+C` — 200 ticks (the default in
`run.small.yaml`) takes a while even in mock mode, and you already have
enough to look at.

## Step 2: See what it built

Every action, message, and metric got logged to `storage/dumps/debug32/` as
you watched (the directory comes from `configs/run.small.yaml`'s
`logging.parquet_dir`):

```bash
ls storage/dumps/debug32/
# actions/  behavior_probes/  citations/  graph_snapshots/  messages/
# metrics_snapshots/  probe_logs/  report_grades/  research_facts/  safety/
# run_debug32_agents.parquet  run_debug32_alpha_aggregates.parquet  ...

ls storage/dumps/debug32/actions/
# actions_t00001.parquet  actions_t00002.parquet  ...
```

One Parquet file per tick per record type. `actions/` has every action any
agent took; `messages/` has every line of dialogue with its steering snapshot
attached; `research_facts/`, `citations/`, `report_grades/` are specific to
the `research` environment you picked in Step 1. Full schema for all of these:
[reference-data-schema.md](reference-data-schema.md).

## Step 3: Verify steering is actually doing something

```bash
python3 scripts/verify_steering.py --messages-dir storage/dumps/debug32/messages
```

This reads back the `steering_snapshot` column from your run's message logs
and checks that persona coefficients actually varied across agents (rather
than every agent secretly getting the same values) — a quick sanity check
before you trust a longer run's results.

## What you built

You ran a 32-agent town simulation where each agent has its own Big-Five
persona coefficients, watched them research library facts and talk to each
other live, and confirmed the simulation's structured telemetry captured
every action with its persona/steering context attached — the same data
pipeline a real (non-mock) run uses for research analysis.

From here:

- Swap `--mock-model` for a real backend and see actual LLM-generated
  dialogue: [howto-run-simulations.md](howto-run-simulations.md#choose-a-backend)
- Try the other two scenarios (`--env policy`, `--env nav`) or a different
  viewing mode (`--tui`, `--viewer`): [howto-run-simulations.md](howto-run-simulations.md)
- Compute your own persona steering vectors instead of using the checked-in
  ones: [howto-compute-steering-vectors.md](howto-compute-steering-vectors.md)
- Understand *why* the simulation is built this way:
  [design.md](design.md)
