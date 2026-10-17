# Local V1 collaboration pilot

This is the first engineering slice of the [research roadmap](collaboration-roadmap.md), not the completed collaboration benchmark. It runs the existing town with three agents and no explicit persona or activation steering. Assigned job roles and the legacy planner advice remain in the prompts. Invalid model actions are counted as planner fallbacks, not credited as valid model decisions.

## Runtime

The configured checkpoint is `mlx-community/Qwen3-8B-4bit`, revision `545dc4251c05440727734bcd94334791f6ab0192`. The local server uses vLLM Metal 0.31.0 with vLLM core 0.31.0 on native Apple Silicon Python 3.12. The simulation client uses a separate environment with `requirements-local.txt`; do not install the legacy CUDA/torch requirements into either environment.

The model is served on loopback only, one request at a time, with a 2,048-token context and 256-token completion cap. The first 4,096-token attempt loaded the model but failed the runtime's available-memory check, so this is a smaller explicitly labelled pilot. Qwen thinking mode is disabled explicitly. A capped or refused completion makes the episode incomplete. Four-bit model weights and this runtime are part of the condition and must appear in any comparison.

The server/client environments and model download are under the task's `work/` directory. The launcher assumes this checkout is under the task's `outputs/`; on another machine set `VLLM_BIN` and `MODEL_DIR` to your installation paths. Follow the [official vLLM Metal installation instructions](https://github.com/vllm-project/vllm-metal/blob/main/docs/installation.md) for compatible core/plugin wheels. Record runtime versions; a server alias alone does not prove model identity.

## Run

From the repository root, start the server in one terminal:

```sh
bash scripts/serve_local_8b.sh
```

In another terminal, with the simulation environment active:

```sh
python scripts/run_local_pilot.py --output artifacts/local-pilot-001 --steps 4 --population 3 --seed 7
```

The output directory must not exist. This prevents accidental mixing of different attempts. Keep episodes bounded while measuring speed and memory on the actual machine. The launcher uses this Mac's GPU; it does not call a paid API.

## Inspect

- `manifest.json`: resolved source/config identity, client runtime versions, elapsed time and complete/incomplete status.
- `resolved-config.yaml`: the actual inherited configuration.
- `console.log`: startup, action progression and failures.
- `simulation.db` and `logs/`: action, message, objective and research evidence.
- `receipts/`: request/response pairs, including truncated responses. Credentials are not used by the local endpoint.
- `summary.json`: decision source counts, action outcomes, report submissions and token usage. It is written only after execution and summary extraction succeed.

Receipts allow replaying an identical inference request without resampling. They are **not** a simulation checkpoint. Do not resume a partially advanced world by simply rerunning its output directory. HTTP serving does not expose exact token IDs here, so receipt replay is not hidden-state/Jacobian replay.

The primary result is not the proportion of successful action handlers. Inspect task completion and report quality, and separate model decisions from planner fallback. A short single-seed run may have no completed objectives. That is useful diagnostic evidence, not a general conclusion about model collaboration.

## Remaining V1 work

The opt-in [distributed-clue task](distributed-clues-evaluation.md) now adds private information, audited communication restrictions, paired task seeds, separate inference replicates and answer-based grading. It has a dedicated runner and should not be mixed with the legacy pilot's scores. Memory controls, fixed-partner comparisons, full world checkpoint/resume, repair of legacy metric labels, and confirmatory statistical analysis remain work to do.

V2 hosted adapters and V3 persona controls are separate tracks. Jev integration, human calibration and RL training are not implemented by this pilot.

The [single-agent verification report](local-v1-results.md) records the later 60-decision diagnostic and portable raw traces. The newer distributed-clue runner supports an opt-in compact final-turn prompt, retaining the old prompt version for comparison and replay. Its smaller-context server flags are documented in the [distributed-clue guide](distributed-clues-evaluation.md).
