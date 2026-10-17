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

## What the traces show

The aggregate scores hide how the agents behave. These counts come from all 60 communication episodes of the 30-instance study, read turn by turn.

- **Question-first openers.** In 57 of 60 episodes, the first resident's first action was an evidence-free question such as "What are the constraints for choosing the meeting plan?" The resident did not offer its own clue. Every resident first held all three clues in round 1 in 53 episodes and in round 0 in one.
- **Broadcasting without stopping.** Deliberation contained 539 shares and 1 wait. Only 176 shares (33%) delivered evidence a recipient lacked, and 201 messages (37%) repeated an earlier message word for word, often the previous speaker's. Rounds 2–3 were almost entirely redundant, and no resident ever noted that the team already held every clue.
- **Failures were conversational, not inferential.** Four residents never validly shared their own clue:
  - In seed 301, the resident who opened with a question spent the next two rounds re-sending the other two clues back to their owners.
  - In seed 321, the holder of the equipment clue wrote messages over the 280-character limit in both replicates. It never shortened them, so the clue never moved. These are all 7 invalid turns in the study.

  Teammates still chose correctly between the two remaining candidates. No resident holding all three clues answered incorrectly.

These behaviors are conventions measured in this local model on this task, not established properties of LLM teams. They can be measured in later studies: opener type, redundancy rate, echo rate, and a team's failure to recognize it has finished.

## Scattered start: teammates out of sight

`+scattered-start-v1` starts each resident in a different room of the three-room town. Everything else matches the first 15 instances of the 30-instance study (seeds 300–314, 2 replicates, the same clues, answers and turn order), so the two layouts pair by instance. The batch passed replay (`clues-scattered-001`, 60 episodes).

| Condition | Answer accuracy (95% CI by instance) | Evidence deliveries | Moves |
| --- | --- | --- | --- |
| Communication | 8/90, 8.9% (0–21%) | 0 | 0 of 270 deliberation actions |
| No communication | 8/90, 8.9% (0–21%) | 0 by construction | 0 |

Communication made no difference: 15 of 15 instances tied. Every deliberation action was a share. Residents asked their question into an empty room, received `delivered_to: []`, and asked again. None of them moved, although the prompt states "Residents start in different rooms" and lists `move` as an action. Because no evidence moved, the communication condition's final prompts match the no-communication ones, and so do its answers. On the same instances with everyone in one room, communication reached 100%.

The power study therefore measured information pooling once teammates are together. It did not measure the coordination needed to find them. Two follow-up studies test the cause:
- More rounds (1 and 5): does more time lead to exploration?
- The `+visible-occupants-v1` perception ablation, which shows residents who else is in their room.

These run on Qwen3-8B, and the scattered layout also runs on Qwen2.5-7B.

## Variance pilot: how to size the next study

`clues-variance-001` uses 10 new instances (seeds 400–409) with 5 inference replicates each, in the communication and no-communication conditions. It passed replay with 100 episodes, 1,200 decisions and no reused requests.

| Condition | Answer accuracy (95% CI by instance) | Between-instance variance | Within-instance (replicate) variance |
| --- | --- | --- | --- |
| Communication | 150/150, 100% | 0 | 0 |
| No communication | 54/150, 36% (15–57%) | 0.129 | 0.012 |

Communication won all 10 instances, with a mean effect of +64 points (CI 43–85). The no-communication baseline is almost deterministic within an instance. Four seeds scored zero correct answers in all five replicates; seeds 406 and 407 scored 2 of 3 in every replicate. This baseline scored 16% on seeds 300–329 and 36% here. The difference comes from how each instance happens to order its candidates, which interacts with the first-listed preference. It does not reflect a change in the model.

**Decision study.** The two components give the standard error of mean no-communication accuracy for *n* instances and *r* replicates: √((0.129 + 0.012/*r*)/*n*). For a ±5-point 95% interval:

| Instances | 1 replicate | 2 replicates | 5 replicates |
| --- | --- | --- | --- |
| 30 | ±13 | ±13 | ±13 |
| 100 | ±7 | ±7 | ±7 |
| 200 | ±5 | ±5 | ±5 |

Extra replicates barely help. Use one or two replicates per instance and about 200 instances for a ±5-point baseline. At the measured local rates (about 45 s per communication episode and 30 s per no-communication episode), 200 instances × 2 conditions × 1 replicate takes about 4 hours.

This is a single-facet decomposition for one model and one task family. It is not a G-study of model ranking, and it does not establish validity. Communication sits at ceiling and has no variance to decompose.

## Dropout: one teammate goes silent

`+dropout-after-round0-v1` makes one seed-chosen resident unavailable after its round-0 turn. Its teammates are not told. Scores cover the two residents who can still submit. The instances are the same as the first 15 of the 30-instance study (seeds 300–314, 2 replicates, together layout, communication only). Without dropout, communication scored 100% on these instances. The batch `clues-dropout-001` passed replay.

| Measure | Result |
| --- | --- |
| Answer accuracy, remaining residents (95% CI by instance) | 49/60, 82% (65–97%) |
| Fully supported answers | 38/60 |
| Dropped resident shared its clue before leaving | 20/30 episodes |
| Accuracy when that clue survived | **40/40** |
| Accuracy when that clue was lost | **9/20** |

The dropped resident's round-0 speaking position decides whether its clue survives:

| Dropped resident's round-0 position | Shared its clue before leaving |
| --- | --- |
| First | 0/8 |
| Second | 14/16 |
| Third | 6/6 |

A resident that speaks first opens with an evidence-free question (28 of 30 episodes), as in the main study. When that resident is the one who drops out, its clue never moves. The two remaining candidates then come down to the first-listed preference. The question-first convention costs nothing while every teammate stays available, but it turns a single unavailable teammate into lost information.

This is an observed association across speaking positions within one model and task. Speaking order is assigned by the seed rather than chosen by the agent, which supports reading it as the mechanism. Still, no intervention on the opener was run. A direct test would add a prompt or protocol variant that requires sharing one's own clue first and repeat the dropout runs.

## More time does not lead to exploration

The scattered layout on seeds 300–314 (communication, one replicate each) was repeated with different numbers of deliberation rounds. All batches passed replay.

| Rounds | Answer accuracy (95% CI by instance) | Moves | Evidence deliveries |
| --- | --- | --- | --- |
| 1 (`clues-scattered-rounds1-001`) | 4/45, 9% (0–20%) | 0 of 45 actions | 0 |
| 3 (`clues-scattered-001`, 2 replicates) | 8/90, 9% (0–21%) | 0 of 270 | 0 |
| 5 (`clues-scattered-rounds5-001`) | 3/45, 7% (0–18%) | 0 of 225 | 0 |
| 3, with dropout (`clues-dropout-scattered-001`) | 2/30, 7% | 0 of 105 | 0 |

Across 645 deliberation actions in scattered rooms, Qwen3-8B chose `share` every time and never chose `move`. Five rounds of unanswered questions and `delivered_to: []` feedback did not lead any resident to look for a teammate. The coordination failure is not caused by too little time.

## Second model: Qwen2.5-7B-Instruct 4-bit

The same instances (seeds 300–314, one replicate) and settings were run with `mlx-community/Qwen2.5-7B-Instruct-4bit` at revision `c26a38f`, served the same way (`--model-profile qwen2.5-7b`). Both batches passed replay. Strict scoring is the official result. The content column is a post-hoc diagnostic (`rejected_final_answer_content` in the analyzer). It counts rejected final turns whose embedded JSON names the correct plan and cites only evidence the resident held.

| Condition | Official answer accuracy (strict) | Plus rejected answers with correct content | Invalid turns |
| --- | --- | --- | --- |
| Single agent, all clues | **0/15** | 15/15 | 25% (every final turn) |
| Communication, together | 12/45, 27% | 27/45, 60% | 10% |
| No communication | 16/45, 36% | 16/45, 36% | 0% |
| Communication, scattered | 16/45, 36% | 16/45, 36% | 0% |

- **Format failure tracks information.** Qwen2.5-7B wrapped its final JSON in Markdown code fences when it held all three clues, and those answers named the correct plan. With missing clues it returned bare JSON. Strict scoring therefore makes communication look harmful: −9 points against no communication (CI −27 to +9). On content, communication helps. A cross-model comparison must report format compliance separately from task success, or this artifact can reverse the conclusion.
- **Less pooling.** Mean evidence coverage with communication was 76%, against 99% for Qwen3-8B on the same instances.
- **No first-listed preference.** Qwen2.5-7B's partial-information choices spread across candidate positions (30, 16, 26 and 2 choices for the first through fourth). Its no-communication accuracy (36%) is close to the uniform-guess reference (34%). The position bias reported above is specific to Qwen3-8B.
- **Same coordination failure.** In scattered rooms, Qwen2.5-7B also shared in all 135 deliberation turns, never moved, and delivered nothing. Its five `move` attempts in the together layout all named the room it was already in.

The official strict protocol, including the rule that invalid output consumes the turn, was set before these runs and is unchanged. The content diagnostic is reported because it changes how the strict score should be read, not to replace it.

## Perception ablation: seeing the empty room does not help

`+visible-occupants-v1` adds `residents_here` to each observation, listing the other residents in the observer's room. The prompt says what the field means. In the scattered layout that list is empty at every turn. If the earlier failure came from not perceiving that nobody was present, this variant should produce moves. Seeds 300–314 were used, with two replicates for Qwen3-8B and one for Qwen2.5-7B, communication only. Both batches passed replay.

| Model | Answer accuracy (95% CI by instance) | Moves | Evidence deliveries |
| --- | --- | --- | --- |
| Qwen3-8B (`clues-scattered-visible-001`) | 8/90, 9% (0–21%) | 0 of 270 actions | 0 |
| Qwen2.5-7B (`clues-qwen25-7b-scattered-visible-001`) | 16/45, 36% (20–51%) | 0 of 135 | 0 |

Neither model moved. In Qwen3-8B's last deliberation round, 80 of 90 messages were the exact sentence "What are the constraints for choosing the meeting plan?", sent while `residents_here` was empty. Qwen2.5-7B talked about its own evidence ("I have information about attendee availability…") but also never left its room. Both models could see that the room was empty, so perception does not explain the failure. With these prompts, neither model forms the plan "go to where a teammate might be". Accuracy matches each model's no-communication guessing level.

A planning scaffold is the obvious next intervention, kept separate from these results. Examples are an explicit hint that sharing requires co-location, or a shared meeting point. Either one would change what is being measured, so it would need its own version label.

## Share-first norm under dropout: the intervention backfired

The dropout analysis suggested that the question-first opener causes lost clues. The direct test was prompt version `phase-v2-intersection-final-share-first-v1`. It adds one deliberation sentence, "Share your own evidence before asking others for theirs.", and leaves final turns unchanged. It was rerun on the identical dropout instances (seeds 300–314, 2 replicates, Qwen3-8B). The batch `clues-dropout-share-first-001` passed replay.

| Measure | Dropout, original prompt | Dropout, share-first norm |
| --- | --- | --- |
| Answer accuracy, remaining residents (95% CI by instance) | 49/60, 82% (65–97%) | **32/60, 53% (30–77%)** |
| Dropped resident's clue survived | 20/30 | 13/30 |
| First speaker opened with a question | 28/30 | 25/30 |
| Second speaker attached its own clue in round 0 | 23/30 | 15/30 |
| Third speaker attached its own clue in round 0 | 29/30 | 15/30 |

The norm barely changed first speakers and made later speakers worse. Under the norm, second and third speakers often *described* their evidence in the message text while attaching no evidence records. One example: "The venue availability evidence indicates that the library and community_center are available…" sent with `evidence_ids: []`. In this environment only attached records transfer, and final decisions use only received records, so a paraphrase is lost information. The agents appear to read "share your own evidence" as "talk about your evidence".

This is a negative result for this wording and this model only. It shows that a plausible protocol fix can reduce reliability, so protocol changes need the same paired testing as any other intervention. It does not show that share-first protocols fail in general. A version that names the mechanism, such as attaching evidence IDs, would be a separate condition. The dropout mechanism itself still stands as an association across speaking positions (0/8 first speakers shared versus 20/22 later speakers). This intervention did not test it cleanly.
