# Steering Vector Evaluation

*Model*: Qwen/Qwen2.5-32B-Instruct
*Alpha*: 2.0
*Timestamp*: 2026-07-06T04:48:03+00:00

| Trait | Prompts | Baseline Acc. | Steered Acc. | Delta Acc. | Sign Consistency | Anti-Steerable |
| --- | --- | --- | --- | --- | --- | --- |
| extraversion (E) | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.500 |
| agreeableness (A) | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| conscientiousness (C) | 2 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |

## Thresholds
- delta_threshold: 0.1
- sign_threshold: 0.55
- anti_steerable_threshold: 0.5

## Grading prompts

## Alpha grid
| Alpha | Trait | Delta Acc. | Logprob Delta | Anti-Steerable |
| --- | --- | --- | --- | --- |
| 0.250 | extraversion (E) | 0.000 | -0.020 | 1.000 |
| 0.250 | agreeableness (A) | 0.000 | 0.035 | 0.000 |
| 0.250 | conscientiousness (C) | 0.000 | 0.015 | 0.000 |
| 0.500 | extraversion (E) | 0.000 | -0.020 | 1.000 |
| 0.500 | agreeableness (A) | 0.000 | 0.060 | 0.000 |
| 0.500 | conscientiousness (C) | 0.000 | 0.015 | 0.500 |
| 0.750 | extraversion (E) | 0.000 | 0.007 | 0.500 |
| 0.750 | agreeableness (A) | 0.000 | 0.049 | 0.500 |
| 0.750 | conscientiousness (C) | 0.000 | 0.049 | 0.000 |
| 1.000 | extraversion (E) | 0.000 | -0.001 | 0.500 |
| 1.000 | agreeableness (A) | 0.000 | 0.038 | 0.000 |
| 1.000 | conscientiousness (C) | 0.000 | 0.055 | 0.000 |
| 1.500 | extraversion (E) | 0.000 | 0.007 | 0.500 |
| 1.500 | agreeableness (A) | 0.000 | 0.070 | 0.000 |
| 1.500 | conscientiousness (C) | 0.000 | 0.122 | 0.000 |
| 2.000 | extraversion (E) | 0.000 | 0.050 | 0.500 |
| 2.000 | agreeableness (A) | 0.000 | 0.091 | 0.000 |
| 2.000 | conscientiousness (C) | 0.000 | 0.165 | 0.000 |
| 3.000 | extraversion (E) | 0.000 | 0.065 | 0.500 |
| 3.000 | agreeableness (A) | 0.000 | 0.142 | 0.000 |
| 3.000 | conscientiousness (C) | 0.000 | 0.230 | 0.000 |

## Cross-trait bleed
| Source \ Target | agreeableness | conscientiousness | extraversion |
| --- | --- | --- | --- |
| agreeableness | 0.091 | 0.112 | -0.030 |
| conscientiousness | 0.108 | 0.165 | 0.089 |
| extraversion | 0.039 | -0.065 | 0.050 |
