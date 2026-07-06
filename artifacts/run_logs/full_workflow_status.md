# Full Workflow Status

- Updated: 2026-07-06T08:01:34Z
- Branch: `codex/qwen-a100-steering-workflow`
- Current step: Step 8, full baseline comparison.
- Command: `.venv/bin/python scripts/run_baselines.py configs/run.small.yaml --env research --output artifacts/baseline_comparison.json`
- Status: interrupted intentionally at the next durable tick boundary for overnight shutdown.
- Active arm: `targeted`
- Active DB: `storage/sim_debug32-debug32-targeted.db`
- Active dump root: `storage/dumps/debug32-targeted`
- Current persisted counts: `action_log=1307`, `msg_log=1307`, `metrics_snapshot=263`, `graph_snapshot=81`, `research_fact_log=501`, `citation_log=409`, `report_grade_log=73`, `run_summary=0`
- Max completed tick: `44`
- GPU status after interrupt: A100 80GB, `0 MiB / 81920 MiB` used, `0%` utilization.
- Last pushed commit before this checkpoint: `d3920fe`

Step 8 was progressing normally but is very long. The baseline artifact `artifacts/baseline_comparison.json` has not been finalized because the full three-arm comparison did not complete.
