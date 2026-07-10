# Jacobian Lens Qwen32 pilot report

## Outcome

The end-to-end engineering pilot passed on 2026-07-09. A 100-prompt
Anthropic Jacobian Lens was fitted for the exact Qwen2.5-32B revision used by
the simulator, one structured CAA-steered agent decision was captured, and
observed/neutral fixed-token-path traces were exported and replay-validated.

This is an engineering validation artifact, not a confirmatory personality or
mechanistic study.

## Compute and teardown

- GPU: NVIDIA H100 PCIe, 81,559 MiB VRAM.
- Service: fixed-price on-demand Vast.ai instance. No bid, interruptible, or
  preemptible capacity was used for the completed pilot.
- Total fixed instance rate including the configured disk: $1.7667/hour.
- Approximate instance lifetime: 1 hour 56 minutes; estimated fixed instance
  charge: $3.42 before any metered network transfer.
- Peak observed fit memory: 66,590 MiB; GPU utilization held at 98-99%.
- The instance was destroyed after artifact recovery, and the authenticated
  instance listing returned no row for it.

The API credential was never written to the repository, an artifact, a
process argument, or shell history.

## Lens artifact

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-32B-Instruct` |
| Model/tokenizer revision | `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` |
| Jacobian Lens commit | `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Lens ID | `jlens-0162010f99a5fc5a` |
| Lens SHA-256 | `0162010f99a5fc5a513e015798661d7423beeebc811af8b4db88afdc3e1b4068` |
| Corpus | WikiText-103 v1, 100 prompts |
| Corpus SHA-256 | `230d5b93dcf759354ad21f288af3f97f6931f800319a4ad780710da8107f5e50` |
| Sequence settings | 128 maximum tokens, first 16 positions skipped |
| Source layers | 36, 58, 62 |
| Target layer | 63 |
| Representation | unquantized BF16 |
| Dimension batch | 4 |
| Fit time | 6,350.8 seconds |

The recovered 151 MiB `lens.pt` matches the manifest hash. It is intentionally
gitignored because it is a reproducible binary; the small manifest, corpus,
fit log, event, and trace artifacts are retained in the repository.

## Agent capture

The pilot used effective alphas `E=0.8`, `A=0.5`, and `C=0.6` with no persona
labels in the prompt. Qwen selected a valid `talk` action and generated 63
tokens. The exact event hash is
`0582826638eb79c006ff9f81a547169799e3d33f8b3c7d9da21a259368322075`.

The real-forward startup smoke test measured a nonzero local residual change
at every active injection site:

| Trait | Layer | Residual delta norm |
|---|---:|---:|
| E | 36 | 9.0019 |
| A | 16 | 9.0340 |
| A | 40 | 9.0677 |
| A | 58 | 8.5580 |
| C | 20 | 9.0555 |
| C | 44 | 9.1127 |
| C | 62 | 7.6284 |

## Exported trace

- Conditions: observed steering and neutral replay over the same recorded
  token path.
- Shape: 63 predicted generated positions x 3 source layers x 10 tokens x 2
  conditions = 3,780 long-form rows.
- Parquet SHA-256:
  `21002228033f6a8ec13b9c181e35a419a0e28777f14b9f843a431c538c7501f6`.
- Maximum final-logit replay error: `0.0`.
- Median actual-next-token rank under the lens: 2 in both conditions.
- Actual next token in lens top 10: 59.8% in both conditions.
- Observed versus neutral top-1 token differed at 4 of 189 layer-position
  comparisons (2.1%).

The small top-1 change is not evidence that CAA has no effect. This is one
event, only three of seven active intervention layers are read out, and the
neutral condition holds the generated token path fixed. The result proves
that the export/replay machinery is exact enough to support a real study; it
does not estimate a population behavioral effect.

## Acceptance decision

The integration clears its engineering gate:

- Exact model, tokenizer, corpus, vector, lens, event, and trace provenance is
  available and content-hashed.
- All configured CAA hooks are observable on a real Qwen32 forward.
- A model-selected action reaches a replay-complete inference event.
- Observed and neutral conditions export successfully.
- Direct final logits and replayed final-layer logits agree exactly.
- Artifacts survive teardown and the GPU instance no longer exists.

The next phase is specified in
[jacobian-lens-phase-two.md](jacobian-lens-phase-two.md). Held-out E/A/C data,
missing-layer merge/report tooling, the live factorial, and a simulation-level
replicated-study runner are implemented. Remaining execution gates are the
four missing fitted matrices, exact-checkpoint held-out/factorial results,
quantization matching, paid independent world replicates, preregistered token
sets, and explicit fallback-rate analysis.

## Recovered files

- Lens manifest: `artifacts/jacobian_lens/qwen32-alltraits-n100/manifest.json`
- Fit log: `artifacts/jacobian_lens/qwen32-alltraits-n100/run.log`
- Captured event: `artifacts/jacobian_lens/qwen32-alltraits-n100/pilot-event.jsonl`
- Capture manifest: `artifacts/jacobian_lens/qwen32-alltraits-n100/pilot-event.manifest.json`
- Trace Parquet: `artifacts/jacobian_lens/qwen32-alltraits-n100/pilot-trace.parquet`
- Trace manifest: `artifacts/jacobian_lens/qwen32-alltraits-n100/pilot-trace.manifest.json`
