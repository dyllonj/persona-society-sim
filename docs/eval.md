# Evaluation Plan

## Personality controllability (RQ1)
- **Self-report probes**: Periodic IPIP-inspired questionnaires inside the sim; score responses via Likert mapping and compare to target coefficients.
- **Behavioral probes**: Situational mini-scenarios injected into the loop (e.g., conflict, planning). Code rubric-based scoring rules in `metrics/social_dynamics.py`.
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

## Experimental matrix
| Condition | Population | Steps | Notes |
|-----------|------------|-------|-------|
| Baseline | 30 | 200 | Prompt-only personas |
| CAA single-trait | 30 | 200 | Only dominant trait steered |
| CAA multi-trait | 100 | 500 | Full Big-Five profiles |
| Memory ablation | 100 | 500 | Remove reflection or planning |
| Scale stress | 300 | 500 | Event-rich weeks |

## Logging & reproducibility
- Persist `RunConfig` + git hash per experiment.
- Snapshot α values per message to correlate outcomes with steering strength.
- Store metrics in Parquet with schema versioning for notebook analysis.
