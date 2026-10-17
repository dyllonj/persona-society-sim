# Distributed clues: local collaboration evaluation

This is an opt-in task variant on the original `World` town map. It retains the library, community center, town square, movement and co-located encounters. It uses a small dedicated turn runner rather than the legacy planner, objective manager, reflection loop or research grader. It is not a replication of the old full society simulation or its scores. Original files and pilot results remain available.

## Task and controls

Three residents start in the town square. Each holds one private constraint on eight possible meeting plans. The constraints represent venue, attendee and equipment availability. Every individual clue permits four plans, any two permit two, and all three identify exactly one. Plan-to-constraint assignments and clue owners vary by environment seed. The venue/time labels do not reveal the private constraints.

During deliberation a resident can share known evidence, ask a question using a share with no evidence, move to an adjacent town location, or wait. A share delivers complete evidence records and a short message only to residents in the same location. Evidence can be forwarded. This structured transfer deliberately makes provenance auditable; it is easier than unrestricted natural-language negotiation and should be reported as such.

Each resident has one final report turn. A report supplies a plan ID and supporting evidence IDs. The model sees its own observations, received evidence and a bounded message history, never the global answer, other residents' private observations or grading feedback. Invalid actions consume their turn; there is no planner fallback. Truncated generation or transport failure makes the batch incomplete instead of silently fabricating an action.

Conditions share the generated problem:

- `communication`: three residents, one initial clue each, normal co-located delivery.
- `no_communication`: identical private clues, schedule and per-resident action budget; no message or evidence delivery. Peer movement, actions and submission results are hidden to prevent side channels.
- `all_clues_single`: one resident starts with all clues. This checks underlying task solvability; it has fewer total calls and is **not** a matched-compute single-agent baseline.

There are no persona prompts, activation steering, assigned expertise roles, human judges, or paid API calls in this runner. The local vLLM backend is restricted to loopback. The instance schedule uses its own random stream, and inference replicates use separately derived sampling seeds. Neither guarantees bitwise deterministic serving.

## Scores and interpretation

- Answer accuracy: correct final selections divided by all scheduled submitters. Invalid or missing submissions remain in the denominator.
- Supported success: correct selection accompanied by all three evidence records that the submitter actually received. Report both per-resident fraction and whether any resident succeeded.
- Evidence coverage and new deliveries: observable information movement, not proof of internal reasoning.
- Invalid actions, tokens and elapsed time: reliability and resource cost.

No-communication cannot obtain complete support by construction. Therefore a supported-success gap alone is **not** an empirical demonstration of collaboration benefits. Compare answer accuracy across matched task instances as well. A resident with one clue can still guess among four candidates; multiple residents create multiple opportunities to guess. Report individual accuracy, not only whether any team member guessed correctly.

Correctness measures constraint satisfaction in this synthetic task. It does not establish human-like social behavior, general collaboration, personality, clinical safety or clinical appropriateness. There is no reward for message count or agreement. The all-clues result distinguishes inference difficulty from information-transfer difficulty.

## Local execution

Start `bash scripts/serve_local_8b.sh`, then use the existing client environment:

```sh
python scripts/run_distributed_clues.py --output artifacts/clues-dev-001 --seeds 7 8 9 --replicates 2 --rounds 3
```

This schedules 18 development episodes and 168 generation calls: six episodes per condition, comprising three task instances with two inference replicates. It is a debugging study, not a sufficient benchmark sample-size claim. The output directory must not already exist. Select fewer seeds or `--replicates 1` for a smaller smoke test. The endpoint and model checkpoint match the [local pilot setup](howto-local-collaboration.md).

Artifacts include source hashes, resolved arguments, model revision, full inference receipts, per-turn observations/raw completions/actions, per-episode summaries and paired descriptive differences. `evaluator-only.json` contains privileged answers and must never be routed to a policy. `review-packets.json` contains short, contextualized questions with blank ratings. These are review materials, not expert labels. Receipts are not complete environment checkpoints; failed batches retain completed episodes but must not be silently mixed into a completed comparison.

On the development Mac, a restart with a 2,048-token context failed the available-memory check. The distributed-clue pilot uses `MAX_MODEL_LEN=1536 GPU_MEMORY_UTILIZATION=0.50 bash scripts/serve_local_8b.sh`. Recent message context is capped at three messages while all received evidence records remain visible. Keep the actual server flags with each run; available memory varies with other applications.

## Related code and design references

- [Hanabi Learning Environment](https://github.com/google-deepmind/hanabi-learning-environment): an open-source cooperative game environment and Gym-like RL interface. It is a reference for private-information cooperation, not a drop-in town implementation. The repository was archived in April 2024; do not present it as a newly maintained dependency.
- [Melting Pot](https://github.com/google-deepmind/meltingpot): social multi-agent evaluation across situations and partners, including held-out scenarios. Its evaluation organization is relevant to later task-family and partner generalization tests; its game engine is not included here.
- [PettingZoo AEC API](https://pettingzoo.farama.org/api/aec/): a reference for a future standard multi-agent adapter. This small observe/step runner is not yet a conforming PettingZoo environment.

The current implementation reuses this repository's `World` locations and local vLLM client. The constraint generator, evidence-transfer rules and grader are new project code. No source from the above projects was copied or vendored, and no claim of compliance with an industry-wide benchmark standard is made.

## Integrating measurement frameworks

### Reliability and sample size

Use a G-study to estimate relevant sources of measurement variability, then a decision study to explore designs with different case and repeat counts. This is an application of [generalizability theory](https://pmc.ncbi.nlm.nih.gov/articles/PMC6699529/), not an implemented G-study in the current runner. It requires a defined population of cases and a defensible sampling design. A reliability coefficient cannot by itself define mastery or establish construct validity.

For a later multi-model evaluation, model is the object of measurement; task instances are nested within task families, inference replicates are repeated measurements, and reviewer is an additional facet only for human-rated scores. Communication condition is a fixed experimental intervention: estimate effects separately instead of pooling it into measurement noise. Decide whether families are fixed coverage requirements or sampled from a broader population before estimating variance.

The present single-model, single-family pilot cannot estimate general model-ranking reliability. Use it to debug and characterize within-task variation. Preserve environment seed, inference replicate, condition, model/runtime and rubric identity so later data can support variance estimation. Treat correlated residents and turns within an episode, and repeated runs on one instance, accordingly; do not inflate case count by counting messages as independent cases.

Before a confirmatory study, specify the minimum useful communication effect, required precision, intended scope and any passing standard. Use pilot variation to plan cases and replicates, with paired analysis of the same instances and uncertainty clustered at the instance level. Ranking models and classifying mastery have different error requirements. Add qualitatively different task families, such as distributed resource allocation or recovery after a changed constraint, before claiming general collaboration ability. A thousand seeds of this same puzzle do not provide that breadth.

### Expert criteria and clinical boundaries

The [RAND/UCLA Appropriateness Method](https://www.rand.org/content/dam/rand/pubs/monograph_reports/2011/MR1269.pdf) combines evidence review and structured expert ratings for clinical appropriateness. An **adaptation** for this benchmark could review whether tasks require meaningful collaboration and whether behavioral rubric items are useful. Draft items, obtain independent ratings, discuss disagreement and re-rate, preserving unresolved differences. Do not describe a simplified process as validated RAM or Delphi merely because multiple reviewers participated.

No expert hours are commissioned or claimed. Locally generated packets allow later reviewers to see what the resident knew, the action and delivery result, with one narrow question. Permit not applicable and insufficient context. Future calibration should include both a representative sample and separately reported difficult cases, double ratings, and blinded model/treatment labels where feasible. A model judge cannot substitute for this evidence by agreeing with another model judge.

For now deterministic constraint checks remain the outcome authority. Clinical harm and appropriateness need clinical scenarios, evidence and qualified review; a wrong meeting plan must not be relabeled as a validated clinical harm measure. Statements about another organization's internal patient-agent use are not supported by these experiments.

### Evaluation versus training

This run is tagged `development`. No training or RL updates happen. A later training pool should be distinct from frozen evaluation cases and rubric versions; disjoint seeds alone do not guarantee meaningful separation when templates and answers are similar. Audit scenario/template overlap, protect held-out cases, and report learning curves for training volume. Evidence-generation scale and reliable evaluation sample size are different decisions.

## Engineering validation

Tests check complementary information over 500 generated instances, private observations, blocked sharing/retrieval/citation of unknown evidence, no-communication delivery and location side channels, co-location, report grading without feedback leakage, malformed action handling, inference seed separation, and scripted-policy solvability. Scripted policies are engineering fixtures, never model-performance evidence.
