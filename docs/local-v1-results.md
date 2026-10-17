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

The new final-turn prompt renders the known constraint lists and asks the model to select a plan present in every list. It never computes or inserts the solution. The diagnostic advances one fixture wait turn, then makes one actual model call. That fixture is explicitly recorded and is not model behavior. The subsequent collaboration check preserved private evidence access while applying this answer format after sharing; its results are below.

## Reproduce and audit

Use the client environment and local server from [the local setup](howto-local-collaboration.md). On the 16 GB Apple M4 used here, the successful server setting was:

```sh
MAX_MODEL_LEN=1536 GPU_MEMORY_UTILIZATION=0.50 bash scripts/serve_local_8b.sh
python scripts/run_single_agent_diagnostic.py --output artifacts/new-single-check --seeds 100 101 102 --replicates 2 --modes intersection-v1 --purpose fresh-verification
python scripts/replay_distributed_clues.py artifacts/new-single-check --diagnostic
```

Checkpoint: `mlx-community/Qwen3-8B-4bit`, revision `545dc4251c05440727734bcd94334791f6ab0192`. Serving: vLLM 0.31.0+cpu with vLLM Metal 0.31.0; temperature 0.2, top-p 0.9, maximum 256 generated tokens, thinking disabled. Local Metal inference only; no paid inference, GPU credits or expert hours.

[Development traces and receipts](../artifacts/local-v1/single-agent-dev-001/) and [fresh verification traces and receipts](../artifacts/local-v1/single-agent-verification-001/) are portable exports. Their manifest output directory is made relative; raw requests, responses and traces are unchanged. Replay reconstructs prompts and observations, checks request settings and response receipts, then regrades actions deterministically. It checks internal consistency, not cryptographic proof of when a run happened or a model vendor's identity. Source hashes describe the launch-time code; subsequent replay hardening can change current source hashes.

The targeted regression suite passed **85 tests, with one legacy GPU test skipped**. It covers environment boundaries, strict parsing, transport/receipt corruption, paired denominators, replay, and failure manifests. Scripted tests are engineering evidence, separate from the live model results above. Direct MLX and hosted adapters are preliminary and were not validated with live inference here.

Personality interventions, human calibration, a G-study, a PettingZoo adapter and RL training are not implemented milestones of this pilot. See [the task and measurement design](distributed-clues-evaluation.md) for controls, prior works and limitations.

## Completed three-case collaboration check

The compact final-turn condition completed all nine episodes on seeds 200, 201 and 202, with one inference replicate and three deliberation rounds per resident. This is a small development comparison, not a confirmatory benchmark.

| Condition | Correct final answers | Fully supported answers | Invalid actions |
| --- | --- | --- | --- |
| Communication enabled | 9/9 | 9/9 | 0/36 |
| Communication disabled | 1/9 | 0/9 | 0/36 |
| Single agent, all clues | 3/3 | 3/3 | 0/12 |

All nine episodes and 84 decisions passed deterministic replay. There were **81 unique inference requests and three cached reuses**. The three all-clues final requests matched earlier communication-condition requests exactly, including sampling seed, and reused the same responses. They are not independent fresh samples. The separate 60-decision diagnostic remains the fresh single-agent evidence.

The batch's unique responses used 35,833 input tokens and 2,865 output tokens. Per-condition summaries count tokens associated with each decision, including cache reuse; they are not a sum of fresh serving work. The replay report separately records unique-response totals. Matched inference seeds and identical-request caching are explicit parts of this paired pilot; results do not demonstrate stochastic repeat reliability.

The enabled condition got every resident all three clues. Raw answer accuracy increased from 1/9 to 9/9 across these three cases. Complete-support gains are partly built into the information restriction, so interpret them alongside raw accuracy. More task families, cases, inference replicates and partner configurations remain necessary before making general collaboration claims.

[Complete comparison, traces, receipts and review packets](../artifacts/local-v1/distributed-clues-003/) are published. Portable export changes only local path metadata and review-packet context labeling; inference requests, responses, trace rows and scores are unchanged. The manifest identifies the archived source commit whose file hashes match the launch-time source. Review packets are unrated and do not provide an ethics assessment. See [the plain-language model journey and scoring rubric](collaboration-walkthrough.md).

## Powered communication comparison: 30 fresh instances

This replaces the three-case check with an estimate. The setup has 30 new task instances (seeds 300–329, never used for development) and two inference replicates per instance. It uses the compact final-turn prompt (`phase-v2-intersection-final-v1`), the same model and serving settings as above, and all three conditions. The runner caps a batch at 20 seeds, so the study ran as two complete batches of 15 instances. Both passed deterministic replay. Each batch covers 90 episodes and 840 decisions, with 812 unique inference receipts and 28 identical-request reuses. Neither batch had a failed or excluded episode.

| Condition | Answer accuracy (95% CI by instance) | Fully supported | Invalid turns | Input / output tokens |
| --- | --- | --- | --- | --- |
| Communication | **180/180, 100%** | 172/180 | 1.0% | 353,913 / 31,077 |
| No communication | **28/180, 15.6%** (6–26%) | 0/180 by construction | 0% | 268,932 / 19,870 |
| Plurality vote over no-communication answers | 15.0% of episodes | — | — | no extra calls |
| Single agent, all clues | 60/60, 100% | 60/60 | 0% | 104,652 / 10,307 |

- **Paired effect.** Communication minus no communication averaged **+84 points (95% CI 74–94)**. Communication won 29 of 30 instances and tied the other. The confidence interval resamples task instances, after averaging over residents and replicates within each instance.
- **Talking versus voting.** A fixed plurality vote over the isolated answers adds no inference and reached 15%. That is 85 points below communication (CI 72–97). The gain comes from moving information, not from pooling guesses.
- **Where communication failed.** No resident answered incorrectly despite holding every clue, and no answer was invalid. In 8 of 180 communication submissions, the resident answered correctly without holding all three clues. Instruction failures were 1% of turns, and every one consumed its turn without a fallback.
- **How isolated agents guess.** All 188 partial-information answers satisfied every clue the resident held. Across both conditions, 78% chose the first listed candidate, 22% the second, and one chose the third. Uniform guessing among consistent candidates would score 26%; isolated agents scored 16%. This is a position preference in the model, not a generator artifact. Over 2,000 seeds the answer's position is uniform. A no-communication baseline therefore reflects how each instance happens to order its candidates. Compare it with the uniform-guess reference, not with zero.
- **Variance.** For no-communication accuracy, 91% of the variance is between task instances (0.079), not between inference replicates of the same instance (0.007). Precision improves much faster with more instances than with more replicates. Use that ratio to size a confirmatory study.
- **Ceiling.** Communication is 100% on every instance, so the together layout cannot distinguish coordination strategies, roles, or personalities. The scattered-start and dropout variants exist to create that headroom.

Figures: [communication effect](assets/fig-communication-effect.svg), [partial-information guesses](assets/fig-partial-information-guesses.svg). Traces are in [dyllonj/persona-society-swarm-local](https://huggingface.co/datasets/dyllonj/persona-society-swarm-local) as `batches/clues-tier1-power-a` and `-b`.

```sh
python scripts/run_distributed_clues.py --output artifacts/clues-tier1-power-a --seeds $(seq 300 314) --replicates 2 --rounds 3 --prompt-version phase-v2-intersection-final-v1
python scripts/run_distributed_clues.py --output artifacts/clues-tier1-power-b --seeds $(seq 315 329) --replicates 2 --rounds 3 --prompt-version phase-v2-intersection-final-v1
python scripts/replay_distributed_clues.py artifacts/clues-tier1-power-a   # and -b
python scripts/analyze_distributed_clues.py artifacts/clues-tier1-power-a artifacts/clues-tier1-power-b
```

Batch a launched from commit `b64868b`. Batch b launched from `6247181`, which changed only the metric naming and touched no distributed-clue code. Each manifest records its own source hashes. Wall time was 43 and 45 minutes.
