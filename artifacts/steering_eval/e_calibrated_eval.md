# Steering Vector Evaluation

*Model*: Qwen/Qwen2.5-32B-Instruct
*Alpha*: 1.0
*Timestamp*: 2026-07-06T04:42:40+00:00

| Trait | Prompts | Baseline Acc. | Steered Acc. | Delta Acc. | Sign Consistency | Anti-Steerable |
| --- | --- | --- | --- | --- | --- | --- |
| extraversion (E) | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |

## Thresholds
- sign_threshold: 0.55
- anti_steerable_threshold: 0.5

## Grading prompts

## Alpha grid
| Alpha | Trait | Delta Acc. | Logprob Delta | Anti-Steerable |
| --- | --- | --- | --- | --- |
| 0.500 | extraversion (E) | 0.000 | 0.030 | 0.500 |
| 1.000 | extraversion (E) | 0.000 | 0.008 | 0.000 |
| 2.000 | extraversion (E) | 0.000 | 0.058 | 0.000 |
| 3.000 | extraversion (E) | 0.000 | 0.089 | 0.000 |
