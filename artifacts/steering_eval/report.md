# Steering Vector Evaluation

*Model*: meta-llama/Llama-3.1-8B-Instruct
*Alpha*: 1.0
*Timestamp*: 2025-11-15T13:35:45+00:00

| Trait | Prompts | Baseline Acc. | Steered Acc. | Δ Acc. | Sign Consistency |
| --- | --- | --- | --- | --- | --- |
| extraversion (E) | 10 | 1.000 | 1.000 | 0.000 | 1.000 |
| agreeableness (A) | 10 | 1.000 | 1.000 | 0.000 | 1.000 |
| conscientiousness (C) | 10 | 1.000 | 1.000 | 0.000 | 1.000 |

## Thresholds
- delta_threshold: 0.1
- sign_threshold: 0.55

## Grading prompts

## Failing conditions
- extraversion delta 0.000 < 0.100
- agreeableness delta 0.000 < 0.100
- conscientiousness delta 0.000 < 0.100
