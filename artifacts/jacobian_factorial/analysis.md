# Paired live-generation factorial analysis

This is a descriptive paired-prompt diagnostic. The unit is a prompt/seed pair, not an independently replicated simulation. No personality score or model-judge score is constructed.

## Provenance

| Field | Value |
| --- | --- |
| Run | factorial-17bb09a6a0d2335f |
| Model | Qwen/Qwen2.5-32B-Instruct |
| Revision | 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd |
| Events SHA-256 | f15b15e19b979925f72f4fbbb61d4954c5ee1228084a8cb2ccd4b202bd10e32e |
| Prompt/seed pairs | 60 |
| Generations | 360 |
| Analysis SHA-256 | fdd08d535f82538c80a45ea8f62eb29f93aa301cc8c23aa7c87184f7d0ca43ce |

## Arm summaries

| Arm | N | JSON syntax valid | Structured valid | Mean tokens | Divergence from neutral | Actions |
| --- | --- | --- | --- | --- | --- | --- |
| neutral | 60 | 1 | 1 | 41.25 | 0 | {"fill_field": 1, "gift": 2, "move": 1, "propose_plan": 8, "research": 6, "scan": 1, "submit_report": 1, "talk": 36, "work": 4} |
| E_only | 60 | 1 | 1 | 41.0833 | 0.116667 | {"fill_field": 1, "gift": 2, "move": 1, "propose_plan": 8, "research": 6, "scan": 1, "submit_report": 1, "talk": 36, "work": 4} |
| A_only | 60 | 1 | 1 | 41.25 | 0.166667 | {"fill_field": 1, "gift": 2, "move": 1, "propose_plan": 8, "research": 6, "scan": 1, "submit_report": 1, "talk": 36, "work": 4} |
| C_only | 60 | 1 | 1 | 41.1167 | 0.1 | {"fill_field": 1, "gift": 2, "move": 1, "propose_plan": 8, "research": 6, "scan": 1, "submit_report": 1, "talk": 36, "work": 4} |
| E_A_C | 60 | 1 | 1 | 41.1 | 0.1 | {"fill_field": 1, "gift": 1, "move": 1, "propose_plan": 8, "research": 6, "scan": 1, "submit_report": 1, "talk": 37, "work": 4} |
| placebo_shuffled | 60 | 1 | 1 | 40.85 | 0.133333 | {"fill_field": 1, "gift": 2, "move": 1, "propose_plan": 8, "research": 6, "scan": 1, "submit_report": 1, "talk": 36, "work": 4} |

## Paired contrasts against neutral

| Arm | Pairs | Path divergence | Mean token Δ | Action change (both valid) | External rubric Δ |
| --- | --- | --- | --- | --- | --- |
| E_only | 60 | 0.116667 | -0.166667 | 0 | — |
| A_only | 60 | 0.166667 | 0 | 0 | — |
| C_only | 60 | 0.1 | -0.133333 | 0 | — |
| E_A_C | 60 | 0.1 | -0.15 | 0.0166667 | — |
| placebo_shuffled | 60 | 0.133333 | -0.4 | 0 | — |

## Origin-stratum summaries

### A

Prompt/seed pairs: 20.

| Arm | Structured valid | Mean tokens | Divergence |
| --- | --- | --- | --- |
| neutral | 1 | 38.95 | 0 |
| E_only | 1 | 39.05 | 0.1 |
| A_only | 1 | 39.25 | 0.15 |
| C_only | 1 | 38.95 | 0.05 |
| E_A_C | 1 | 38.7 | 0.15 |
| placebo_shuffled | 1 | 39.1 | 0.15 |

### C

Prompt/seed pairs: 20.

| Arm | Structured valid | Mean tokens | Divergence |
| --- | --- | --- | --- |
| neutral | 1 | 43.6 | 0 |
| E_only | 1 | 43.25 | 0.15 |
| A_only | 1 | 43.45 | 0.1 |
| C_only | 1 | 43.3 | 0.15 |
| E_A_C | 1 | 43.35 | 0.1 |
| placebo_shuffled | 1 | 43.35 | 0.2 |

### E

Prompt/seed pairs: 20.

| Arm | Structured valid | Mean tokens | Divergence |
| --- | --- | --- | --- |
| neutral | 1 | 41.2 | 0 |
| E_only | 1 | 40.95 | 0.1 |
| A_only | 1 | 41.05 | 0.25 |
| C_only | 1 | 41.1 | 0.1 |
| E_A_C | 1 | 41.25 | 0.05 |
| placebo_shuffled | 1 | 40.1 | 0.05 |

The JSON companion contains every prompt/seed contrast, including first token divergence, action changes, validity deltas, length deltas, and any externally supplied rubric delta.
