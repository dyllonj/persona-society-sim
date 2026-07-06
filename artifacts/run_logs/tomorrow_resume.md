# Tomorrow Resume Notes

## Stopped State

- Date stopped: 2026-07-06T08:01:34Z
- Branch: `codex/qwen-a100-steering-workflow`
- Last pushed checkpoint before final data commit: `d3920fe`
- Workflow stopped at: Step 8, baseline comparison.
- Reason stopped: the full three-arm baseline was running cleanly but was taking many hours; it was interrupted at the next flushed tick boundary so the Vast instance can be stopped overnight.
- Interrupted command:
  `.venv/bin/python scripts/run_baselines.py configs/run.small.yaml --env research --output artifacts/baseline_comparison.json`
- Completed before stop: Steps 1-7 passed, including steerability smoke, vector computation, layer sweep, full alpha-grid plus bleed eval, and CI gates.
- Partial Step 8 state: targeted arm only, stopped at max tick `44`.

## Saved Data

- Partial targeted dump: `storage/dumps/debug32-targeted`
- Analysis text: `artifacts/run_logs/targeted_partial_analysis.txt`
- Analysis plots: `storage/analysis/action_distribution.png`, `storage/analysis/steering_heatmap.png`, `storage/analysis/activity_over_time.png`
- Local SQLite DB: `storage/sim_debug32-debug32-targeted.db` (ignored by git; parquet dump is committed instead)
- Run log: `artifacts/run_logs/full_workflow_20260706T051505Z.log`

## Partial Targeted Counts

- Max tick: `44`
- Actions/messages: `1307`
- Metrics snapshots: `263`
- Graph snapshots: `81`
- Research facts: `501`
- Citations: `409`
- Report grades: `73`

## Partial Analysis Highlights

- Action distribution: research `501` (38.3%), cite `409` (31.3%), talk `305` (23.3%), submit_report `73` (5.6%), move `19` (1.5%).
- Outcomes: 100% success over `1307` actions.
- Agents active: 32/32.
- Token usage: `1,652,268` total tokens, average `1144.8` input and `119.4` output tokens per message.
- Steering snapshots present for 32 agents.

## Resume Options

The current baseline script does not support resuming a partially completed arm from the parquet/SQLite dump. To complete the exact Step 8 comparison, rerun:

```bash
cd /workspace/persona-society-sim
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/run_baselines.py configs/run.small.yaml \
  --env research \
  --output artifacts/baseline_comparison.json
```

If the goal is to avoid another very long run, use a capped diagnostic tomorrow:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/run_baselines.py configs/run.small.yaml \
  --env research \
  --steps 10 \
  --max-events 4 \
  --run-tag resume-smoke \
  --output artifacts/baseline_comparison.resume_smoke.json
```

After Step 8 completes, continue with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m orchestrator.cli configs/run.small.yaml --env research --live

.venv/bin/python scripts/analyze_simulation.py --dump-dir storage/dumps/debug32
```

Only read `experiments/autoresearch/` after the full workflow succeeds.
