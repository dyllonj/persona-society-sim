# Local V1 results — October 17, 2026

The current engineering milestone is a reliable final decision with all clues available. Qwen3-8B in 4-bit passed **60/60 valid, correct, fully supported submissions**, on **30 fresh task instances with two inference replicates each**. These are 60 decisions, not 60 independent cases or full society episodes. This passes the predeclared local engineering gate of at least 95%; it does not establish general collaboration or benchmark mastery.

## What changed and what failed

| Run | Completed evidence | Result |
| --- | --- | --- |
| Original town pilot | One short episode, 12 actions | 10 valid model actions, two planner fallbacks; both reports covered one of three facts |
| Initial distributed clues | One complete communication episode; batch stopped | Some evidence moved, but all three final actions were invalid shares instead of reports |
| Explicit phase instructions | Seven complete episodes; planned batch stopped | Both seed-7 replicates failed in all three conditions, even with every clue; seed-8 communication got 3/3 correct |
| Final-turn prompt development | Five cases × two prompt formats | Original phase prompt 4/5 correct; explicit constraint intersection prompt 5/5 |
| Fresh single-agent verification | 30 new cases × two replicates | Explicit intersection prompt 60/60 correct, valid and supported |

The fresh verification uses seeds 100–129, distinct from development seeds 7–11. It is a small, synthetic constraint-intersection family. Different seeds do not establish transfer to different task families. Earlier incomplete batches remain excluded from any completed comparison. The seven completed phase-instruction episodes were replayed separately; they are not a finished paired benchmark.

The new final-turn prompt renders the known constraint lists and asks the model to select a plan present in every list. It never computes or inserts the solution. The diagnostic advances one fixture wait turn, then makes one actual model call. That fixture is explicitly recorded and is not model behavior. The next collaboration check should preserve private evidence access while applying this answer format after sharing.

## Reproduce and audit

Use the client environment and local server from [the local setup](howto-local-collaboration.md). On the 16 GB Apple M4 used here, the successful server setting was:

```sh
MAX_MODEL_LEN=1536 GPU_MEMORY_UTILIZATION=0.50 bash scripts/serve_local_8b.sh
python scripts/run_single_agent_diagnostic.py --output artifacts/new-single-check --seeds 100 101 102 --replicates 2 --modes intersection-v1 --purpose fresh-verification
python scripts/replay_distributed_clues.py artifacts/new-single-check --diagnostic
```

Checkpoint: `mlx-community/Qwen3-8B-4bit`, revision `545dc4251c05440727734bcd94334791f6ab0192`. Serving: vLLM 0.31.0+cpu with vLLM Metal 0.31.0; temperature 0.2, top-p 0.9, maximum 256 generated tokens, thinking disabled. Local Metal inference only; no paid inference, GPU credits or expert hours.

[Development traces and receipts](../artifacts/local-v1/single-agent-dev-001/) and [fresh verification traces and receipts](../artifacts/local-v1/single-agent-verification-001/) are portable exports. Their manifest output directory is made relative; raw requests, responses and traces are unchanged. Replay reconstructs prompts and observations, checks request settings and response receipts, then regrades actions deterministically. It checks internal consistency, not cryptographic proof of when a run happened or a model vendor's identity. Source hashes describe the launch-time code; subsequent replay hardening can change current source hashes.

The targeted regression suite passed **81 tests, with one legacy GPU test skipped**. It covers environment boundaries, strict parsing, transport/receipt corruption, paired denominators, replay, and failure manifests. Scripted tests are engineering evidence, separate from the live model results above. Direct MLX and hosted adapters are preliminary and were not validated with live inference here.

Personality interventions, human calibration, a G-study, a PettingZoo adapter and RL training are not implemented milestones of this pilot. See [the task and measurement design](distributed-clues-evaluation.md) for controls, prior works and limitations.
