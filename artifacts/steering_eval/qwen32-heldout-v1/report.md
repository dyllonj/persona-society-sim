# Steering Vector Evaluation

*Model*: Qwen/Qwen2.5-32B-Instruct
*Resolved dtype*: bfloat16
*Alpha*: 2.0
*Per-trait alphas*: {'E': 0.8, 'A': 0.5, 'C': 0.6}
*Primary logprob metric*: mean_per_continuation_token
*Timestamp*: 2026-07-10T23:42:20+00:00

| Trait | Alpha | Prompts | Baseline Acc. | Steered Acc. | Delta Acc. | Mean logprob gap Δ | Summed logprob gap Δ | Sign Consistency | Anti-Steerable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| extraversion (E) | 0.800 | 20 | 1.000 | 1.000 | 0.000 | -0.000 | -0.017 | 1.000 | 0.550 |
| agreeableness (A) | 0.500 | 20 | 1.000 | 1.000 | 0.000 | 0.002 | 0.027 | 1.000 | 0.400 |
| conscientiousness (C) | 0.600 | 20 | 1.000 | 1.000 | 0.000 | -0.003 | -0.041 | 1.000 | 0.500 |

## Thresholds
- delta_threshold: 0.1
- sign_threshold: 0.55
- anti_steerable_threshold: 0.5

## Grading prompts

## Cross-trait bleed
| Source \ Target | agreeableness | conscientiousness | extraversion |
| --- | --- | --- | --- |
| agreeableness | 0.002 | -0.001 | -0.001 |
| conscientiousness | 0.003 | -0.003 | -0.002 |
| extraversion | 0.000 | -0.002 | -0.000 |

## Failing conditions
- extraversion delta 0.000 < 0.100
- extraversion anti-steerable 0.550 > 0.500
- conscientiousness delta 0.000 < 0.100
