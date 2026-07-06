# Steering Vector Evaluation

*Model*: Qwen/Qwen2.5-32B-Instruct
*Alpha*: 0.5
*Timestamp*: 2026-07-06T04:27:18+00:00

| Trait | Prompts | Baseline Acc. | Steered Acc. | Delta Acc. | Sign Consistency | Anti-Steerable |
| --- | --- | --- | --- | --- | --- | --- |
| extraversion (E) | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| agreeableness (A) | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| conscientiousness (C) | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.500 |

## Thresholds
- delta_threshold: 0.1
- sign_threshold: 0.55
- anti_steerable_threshold: 0.5

## Grading prompts

## Failing conditions
- extraversion delta 0.000 < 0.100
- extraversion anti-steerable 1.000 > 0.500
- agreeableness delta 0.000 < 0.100
- conscientiousness delta 0.000 < 0.100
