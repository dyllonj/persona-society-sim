# Evaluation Plan

## Personality controllability (RQ1)
- **Self-report probes**: `orchestrator/probes.ProbeManager` loads `configs/probes.yaml`, injects IPIP-style Likert prompts into the observation stream, and writes the parsed score + prompt text to `probe_log` rows so adherence can be compared against target coefficients.
- **Behavioral probes**: Situational mini-scenarios injected into the loop (e.g., conflict, planning) with rubric keywords stored in `BehaviorProbeDefinition.outcomes`; responses are classified into categories and logged via `behavior_probe_log` for downstream rubric scoring.
- **Drift metrics**: Track cosine similarity between early vs late persona embeddings derived from generated text; log α adjustments required to maintain scores.

## Social structure (RQ2)
- **Graph metrics**: Degree/centrality/clustering/assortativity computed each tick; compare across trait bands.
- **Dyadic sentiment**: Sentiment polarity of messages per tie to quantify tie positivity vs trait intensity.
- **Task outcomes**: Completion latency + quality proxies for chores/projects, grouped by trait combinations.

## Method validity (RQ3)
- **Capability probes**: Run held-out QA/reasoning tasks before/after steering to ensure minimal performance loss.
- **CAA vs prompt-only**: Mirror Generative Agents prompt recipes and compare persona adherence + capability metrics.
- **ActAdd variants**: Evaluate different layer sets and coefficient magnitudes.

## Macro dynamics (RQ4)
- **Opinion cascade studies**: Seed topics weekly, measure variance/bimodality/polarization modularity.
- **Homophily**: Compute assortativity coefficients by trait and opinion alignment.
- **Well-being proxies**: Diary sentiment, reciprocity rates, conflict counts per agent.

## Steering vector evaluation harness

- `scripts/eval_vectors.sh` regenerates steering vectors via the metadata-aware loader, runs `steering.eval`, and writes both JSON + Markdown summaries to `artifacts/steering_eval/`. E/A/C each have 20 checked-in held-out items; `uv run python scripts/split_eval_prompts.py --verify-existing` rejects ID, exact-text, normalized-text, and near-duplicate-stem leakage against extraction data.
- The harness reports baseline vs steered accuracy, log-prob deltas, and sign consistency for every prompt so you can catch regressions before running a multi-agent sim.
- Set `STEERING_ALPHA` to the same value as `steering.strength` in your run config to evaluate the correct dose. Use `DELTA_THRESHOLD` and `SIGN_THRESHOLD` to fail the script when the expected gains disappear.
- `steering.eval` can also capture transcripts with steering toggled on/off to manually verify tone changes. These transcripts, along with `vector_store_id`, prompt metadata, and evaluation hashes, serve as the reproducibility record for persona experiments.
- `--traits` defaults to `extraversion agreeableness conscientiousness` because O/N do not yet have checked-in prompt sets or vector artifacts. The evaluator already recognizes their aliases.

## Experimental matrix
| Condition | Population | Steps | Notes |
|-----------|------------|-------|-------|
| Baseline | 30 | 200 | Prompt-only personas |
| CAA single-trait | 30 | 200 | Only dominant trait steered |
| CAA multi-trait | 100 | 500 | Full Big-Five profiles |
| Memory ablation | 100 | 500 | Remove reflection or planning |
| Scale stress | 300 | 500 | Event-rich weeks |

The implemented confirmatory small-study matrix is
`experiments/society_study/matrix.yaml`: neutral, E-only, A-only, C-only,
E+A+C, and a seeded trait-label-derangement active control, each with five
paired world seeds, 30 agents, and 100 ticks. `scripts/analyze_society_study.py`
first collapses every run to one row; uncertainty is calculated across worlds,
never across nested agents or actions.

## Logging & reproducibility
- Persist `RunConfig` + git hash per experiment.
- Snapshot α values per message to correlate outcomes with steering strength.
- Store metrics in Parquet with schema versioning for notebook analysis.
