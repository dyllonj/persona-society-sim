# Jacobian Lens phase-two results

Status: mechanistic fit, seven-layer replay, held-out evaluation, and the
single-trait factorial completed on 2026-07-10. The replicated society study is
the remaining confirmatory stage.

## Locked runtime and artifacts

All runs used `Qwen/Qwen2.5-32B-Instruct` at revision
`5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, BF16 inference,
`torch==2.7.1+cu128`, `transformers==5.13.0`, and `jlens==0.1.0` from commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.

The missing-layer fit reproduced the independently completed but subsequently
lost first artifact byte-for-byte. Its SHA-256 was
`9df545b8536a5e3eb6a209b14050fe58720c9c4e126bb5f830e84977040b6068`
in both runs. This is strong evidence that fitting is deterministic under the
pinned environment.

| Artifact | SHA-256 |
| --- | --- |
| Existing L36/L58/L62 parent | `0162010f99a5fc5a513e015798661d7423beeebc811af8b4db88afdc3e1b4068` |
| New L16/L20/L40/L44 parent | `9df545b8536a5e3eb6a209b14050fe58720c9c4e126bb5f830e84977040b6068` |
| Exact seven-layer union | `d2904d15e552f752e252e57584a3f2e34d90c370eafe60ae92623ca5d3289377` |
| Trait-space analysis content | `a79add52b74059b57575a20f7d7e72e519ea716cab4553f665b23e6aff3f1d2d` |
| Seven-layer trace Parquet | `bf942721956834c28c2ca81fb5f63578c70ff78439b6c6d6f932a76a810251d6` |
| Factorial generations | `f15b15e19b979925f72f4fbbb61d4954c5ee1228084a8cb2ccd4b202bd10e32e` |
| Factorial analysis content | `fdd08d535f82538c80a45ea8f62eb29f93aa301cc8c23aa7c87184f7d0ca43ce` |

The merge contains exactly layers 16, 20, 36, 40, 44, 58, and 62. Every
matrix in the union is `torch.equal` to its declared parent, the parents have
no overlapping layers, and all 21 compatibility fields match.

Large binaries and the complete remote artifact tree are archived in the
[Hugging Face dataset](https://huggingface.co/datasets/dyllonj/persona-society-jacobian-lens)
at verified commit `bd7a053ed18aab889147c34ab51bb280071b1b2d`. Because the
project `.gitignore` excludes `artifacts/**/lens.pt`, the three lens binaries
are stored under `lenses/` in that dataset. Their Hub LFS SHA-256 values match
the three hashes in the table above exactly.

## Fit behavior

The four-layer fit processed 100 fixed corpus prompts in 17,844 seconds. Most
late running-mean deltas were between 0.02 and 0.05. Prompt 40 was an isolated
high-leverage sample: its plain-maskray taxonomy passage, which contains many
uncommon scientific and proper-name tokens, produced
`max(||J|| / sqrt(d)) = 8.215` and a running-mean delta of 0.47. The next two
prompts immediately returned to the normal range. This was a corpus outlier,
not sustained numerical instability.

## Seven-component geometry

All seven configured components were transported to the common target-layer
residual basis before comparison. Raw cross-layer vector cosines were not
used.

The clearest result is downstream convergence between nominally distinct A
and C interventions:

- A-trait aggregate versus C-trait aggregate cosine: **0.834305**.
- A@16 versus C@20 cosine: **0.973536**.
- A@40 versus C@44 cosine: **0.645739**.
- A@58 versus C@62 cosine: **0.608108**.

E points in the opposite target-space direction: aggregate A-versus-E cosine
is **-0.750569**, and C-versus-E is **-0.682824**. At deployed alphas
E=0.8, A=0.5, and C=0.6, the combined direction has norm 1.82827 and a
coherence ratio of 0.604556. A and C therefore reinforce one another while E
partly cancels their shared direction.

Early A@16 and C@20 directions receive unusually selective transport gains:
their directional-to-isotropic gain ratios are 1.895 and 1.756 respectively.
This makes the near-collinearity harder to dismiss as a generic property of
the corresponding Jacobians.

Vocabulary projections are dominated at several components by punctuation,
whitespace, multilingual fragments, and tokenizer artifacts. Some late-layer
positive projections are semantically suggestive (`strong`, `detailed`,
`firm`), but the lists do not support a clean psychological label on their
own. They should be treated as directional logit diagnostics, not generated
behavior.

## Seven-layer replay

The pilot replay exported 8,820 rows: two conditions, seven layers, 63 source
positions, and top-10 projections. It covered one prompt position and 62
generated positions per condition. Direct and replayed final logits agreed
exactly (`final_logit_replay_max_abs_error = 0.0`).

The actual next token becomes progressively legible in the transported
readout: its median rank is in the thousands at layers 16/20, 512 at layer 36,
3 at layer 58, and 1 at layer 62. Observed-versus-neutral rank changes are
small on the fixed token path, which is why the independent-generation
factorial is necessary.

## Held-out deployed-dose evaluation

Each trait used 20 authored prompts disjoint from the eight vector-extraction
items. Baseline forced-choice accuracy was already 1.0 for every trait, so
accuracy had no headroom. Mean conditional log-probability-gap changes were:

| Trait | Alpha | Gap change | Directionally improved | Anti-steerable |
| --- | ---: | ---: | ---: | ---: |
| E | 0.8 | -0.000492 | 0.45 | 0.55 |
| A | 0.5 | +0.002007 | 0.60 | 0.40 |
| C | 0.6 | -0.003381 | 0.50 | 0.50 |

The evaluator correctly exited nonzero because E and C failed preregistered
thresholds. Off-diagonal bleed magnitudes were all below 0.003. These data do
not establish useful held-out trait adherence at deployed doses; only A shows
a small positive shift.

## Independent-token-path factorial

The paired factorial contains 60 prompt/seed blocks and 360 generations. All
outputs were syntactically valid JSON and valid structured actions.

| Arm | Path divergence from neutral | Action changes | Mean token delta |
| --- | ---: | ---: | ---: |
| E only | 0.1167 | 0/60 | -0.1667 |
| A only | 0.1667 | 0/60 | 0.0000 |
| C only | 0.1000 | 0/60 | -0.1333 |
| E+A+C | 0.1000 | 1/60 | -0.1500 |
| Coordinate-permuted placebo | 0.1333 | 0/60 | -0.4000 |

Steering changes sampled token paths, but at these doses it almost never
changes the selected action. A has the highest surface-path divergence without
changing any action. The combined intervention is weaker than the placebo on
path divergence and changes only one action (gift to talk). Therefore the
strong A/C Jacobian alignment is currently a representational result, not
evidence of a practically meaningful action-policy effect.

## Research decision

The A/C convergence survives the full seven-component mechanistic test but
does not survive as a clear behavioral effect in either held-out forced-choice
scoring or the live action factorial. The society study remains useful as a
high-powered falsification attempt because repeated interactions may amplify
small text-level shifts. Any society-level claim must use the simulation run,
not individual actions, as the independent unit and must compare real vectors
against its active placebo arm.
