# CAA Activation Steering: Known Limitations Research

Compiled 2026-07-05. Sources prioritized by direct relevance to our CAA personality-trait
steering pipeline (Llama-3.1-8B -> Qwen2.5-32B migration).

---

## Finding 1 — Steering vectors do NOT transfer across models (critical for Llama->Qwen)

- **Source:** Rimsky (Panickssery) et al., "Steering Llama 2 via Contrastive Activation
  Addition," ACL 2024. https://arxiv.org/abs/2312.06681
  + Ayyub, "What I Learned (And Didn't) Steering Qwen3 Models," Jan 2026.
  https://omar.bet/2026/01/17/What-I-Learned-Steering-Qwen3-Models/
  + IBM activation-steering library FAQ (vectors are model/layer-specific by construction).
  https://github.com/IBM/activation-steering/blob/main/docs/faq.md
- **Blocker:** CAA vectors live in a specific model's residual-stream basis at a specific
  layer index. They are not portable: (a) different architectures (Llama vs Qwen) have
  different hidden dims (Llama-3.1-8B = 4096; Qwen2.5-32B = 5120), so the vectors are not
  even dimensionally compatible; (b) even within the SAME family and SAME size, Ayyub's
  Qwen3 sweep shows steerability is "idiosyncratic in ways that don't track obvious
  properties" and "steering vectors that work on one model won't necessarily transfer
  within the same family." Optimal layer depth shifts with training method (RL models
  steer at 70-85% depth vs 50-65% for distilled).
- **Resolution:** None — vectors MUST be recomputed from scratch on the target model using
  the same A/B contrastive prompt corpus. Layer indices must be re-tuned via a layer sweep
  on the new model; do not assume `[12, 16, 20]` (or whatever Llama layers we use) map to
  Qwen. Treat the Llama vectors and `configs/steering.layers.yaml` as throwaway on migration.
- **Applicability:** Direct. Plan a full re-extraction run on Qwen2.5-32B with
  `scripts/compute_vectors.sh` and a fresh layer sweep before trusting any persona output.

## Finding 2 — Anti-steerability: ~1/3 of samples steer the WRONG way

- **Source:** Tan et al., "Analyzing the Generalization and Reliability of Steering
  Vectors," NeurIPS 2024. https://arxiv.org/abs/2407.12404
  + Braun et al., "A Sober Look at Steering Vectors for LLMs," AI Alignment Forum, Nov 2024.
  https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/
  + Braun et al., "Understanding (Un)Reliability of Steering Vectors in Language Models,"
  May 2025. https://arxiv.org/html/2505.22637v1
- **Blocker:** Across 36 binary-choice datasets, the fraction of "anti-steerable" samples
  (where the vector shifts the logit-difference opposite to intended) ranges from 3% to 50%,
  averaging ~1/3. Aggregate metrics (mean logit-diff) mask this because positive and
  negative per-sample effects cancel. Braun et al. (2025) confirm no prompt type
  (prefilled / instruction / 5-shot / non-prefilled) consistently avoids this; all seven
  prompt variants produce net-positive but high-variance effects. Some behaviors are
  effectively unsteerable even after sweeping all layers and strengths.
- **Resolution:** Not fully resolved. Predictors: (1) directional agreement (mean cosine
  similarity between individual activation differences and the steering vector) and (2)
  separability (discriminability index d' along the difference-of-means line) both
  predict steerability. Use these as pre-flight diagnostics on the training set. Report
  per-sample anti-steerable fraction, not just mean effect.
- **Applicability:** Our `steering.eval` harness currently reports accuracy deltas and
  log-prob gaps. Add: (a) anti-steerable sample fraction per trait, (b) directional
  agreement / d' on the training activations as a steerability predictor, (c) per-sample
  variance. Gate trait launches on these, not just aggregate accuracy.

## Finding 3 — Personality trait vectors are entangled; orthonormalization doesn't fix it

- **Source:** Bhandari et al., "Do Personality Traits Interfere? Geometric Limitations of
  Steering in Large Language Models," Jan 2026.
  https://arxiv.org/html/2602.15847
- **Blocker:** Directly about Big Five steering. Steering one trait (e.g. Openness)
  consistently moves other traits ( Agreeableness, Conscientiousness, Extraversion up;
  Neuroticism down) — traits sit in a "slightly coupled subspace," not orthogonal axes.
  Crucially, hard orthonormalization (Löwdin / Gram-Schmidt) enforces zero cosine overlap
  in activation space but does NOT eliminate cross-trait behavioral bleed in generation
  (Bmax remains ~3.0 on a 1-5 Likert scale even under C5 hard orthonormalization).
  Orthonormalization ALSO degrades fluency (Extraversion fluency drops 4.9->3.8).
  Cross-model: pattern is consistent across LLaMA-3-8B and Mistral-8B, but Neuroticism
  is strongly steerable in LLaMA (T~3.1) and nearly unsteerable in Mistral (T~0.0-0.7) —
  model-specific, not geometric.
- **Resolution:** Soft projection (C4: partial attenuation when |cos|>tau) gives the best
  trade-off — preserves ~0.85-1.0 signal retention with reduced (not eliminated) bleed.
  But no scheme achieves independent trait control. The authors recommend treating
  interference as an empirical phenomenon to measure, not something to fully remove.
- **Applicability:** High. Our pipeline applies trait vectors independently per agent.
  Expect cross-trait bleed; instrument it. Do NOT assume orthogonal trait alphas produce
  orthogonal behavior. Consider C4-style soft projection if bleed becomes problematic,
  but accept the fluency cost. Re-verify per-trait steerability on Qwen (Neuroticism may
  behave very differently than on Llama).

## Finding 4 — Layer choice is critical and model-specific

- **Source:** Ayyub, "What I Learned Steering Qwen3 Models," Jan 2026.
  + Bhandari et al. (2026) hybrid layer selection.
  + Braun et al. (2025) — used layer 13 following Tan et al.
- **Blocker:** Optimal steering layer varies by model, by training method, and by concept.
  Ayyub: RL-trained Qwen models peak at 70-85% depth; distilled at 50-65%. This held
  across all 6 datasets. Bhandari et al. use a hybrid per-trait prior-layer selection
  plus a runtime dynamic check. A single fixed layer set (e.g. our `[12, 16, 20]`) is
  fragile — it was tuned for Llama-3.1-8B's 32-layer geometry and will not map cleanly
  to Qwen2.5-32B's 64 layers.
- **Resolution:** Per-trait, per-model layer sweep. Bhandari's hybrid approach: select an
  offline prior layer per trait via neutral probe prompts (apply small signal, measure
  distributional sensitivity at next token), then a lightweight runtime dynamic check.
- **Applicability:** `configs/steering.layers.yaml` must be regenerated for Qwen2.5-32B.
  Run a layer sweep per trait (probably sampling every ~4-8 layers across the 64) and
  record the sensitivity curve. RL vs distilled distinction doesn't apply (Qwen2.5-32B
  is SFT+DPO, not the Qwen3 RL pipeline), so don't assume the 70-85% rule — measure it.

## Finding 5 — Standard evaluation metrics overestimate steering effectiveness

- **Source:** Pres et al., "Towards Reliable Evaluation of Behavior Steering Interventions
  in LLMs," MINT @ NeurIPS 2024. https://arxiv.org/abs/2410.17245
  + Braun et al., "A Sober Look," Nov 2024.
- **Blocker:** Most steering papers (including CAA) train AND evaluate on multiple-choice
  / logit-difference settings. Pres et al. identify four missing evaluation properties:
  (1) open-ended generation contexts, (2) model likelihoods instead of sampled tokens,
  (3) standardized cross-behavior comparison, (4) meaningful baselines. When they add
  these, CAA is "less effective than previously reported." Ayyub independently confirms:
  logit-diff effect size does NOT predict generation-based behavioral change (sycophancy:
  logit effect 1.04 -> only 8% behavior change; corrigibility: 1.62 -> 45%).
- **Resolution:** Pres et al. propose an evaluation pipeline covering all four. Ayyub:
  "for anything safety-relevant, generation-based evaluation is probably necessary
  despite the cost."
- **Applicability:** Our `steering.eval` harness uses held-out contrast prompts (logit-based).
  This is the cheap proxy. Add a generation-based eval: have steered models produce free
  text, judge with a separate LLM (Bhandari et al. use GPT-4o-mini) on Big Five Likert
  items. Do not launch a trait on logit-diff alone. The "Sober Look" authors also note
  vectors that work in-distribution tend to work OOD, so a small held-out generation
  check is a cheap gate.

## Finding 6 — Steering degrades fluency / general capability

- **Source:** Stickland et al., 2024 (MT-Bench degradation ~ halving pretrain compute).
  https://arxiv.org/abs/2406.15518
  + von Rütte et al., 2024 (perplexity increase on OpenAssistant).
  + Rimsky et al., 2024 (large magnitudes degrade open-ended text per GPT-4 + human).
  + Bhandari et al., 2026 (orthonormalization specifically degrades fluency, Fig 1).
- **Blocker:** Any steering magnitude trades trait expression against fluency/coherence.
  Orthonormalization (Finding 3) makes this worse. Large `steering.strength` scalars
  degrade general capability measurably.
- **Resolution:** Report degradation explicitly (tiny benchmark perplexity, MT-Bench-style
  scores). Tune `steering.strength` to the lowest value that achieves the target trait
  shift. The "Sober Look" authors recommend this as a mandatory reporting requirement.
- **Applicability:** Our MetricTracker aggregates by trait band and |alpha| bucket but we
  should confirm we log a fluency/coherence proxy per steered message. Tune the global
  `steering.strength` scalar down on Qwen (32B is more capable; smaller residual may
  suffice and oversteering is more visibly damaging).

## Finding 7 — Constant (uniform) steering across token positions is not faithful

- **Source:** Heyman & Vandeputte, "Steer Like the LLM: Activation Steering that Mimics
  Prompting," ICML 2026. https://arxiv.org/html/2605.03907v1
- **Blocker:** Standard CAA applies the same coefficient at every steered token position
  (or only at the last prompt token). Empirical analysis of how PROMPT steering actually
  intervenes shows token-specific strength: strong on some tokens, near-zero on others
  (Fig 2). Constant steering is "not faithful to the mechanics of prompt steering."
  Their PSR (Prompt Steering Replacement) models with learned per-token coefficients
  outperform constant CAA, especially at high coherence.
- **Resolution:** PSR / token-specific coefficients learned from activations. Also relevant:
  Stolfo et al. 2025 and Hedström et al. 2025 normalize per-token projection to mitigate
  oversteering. Our pipeline already does prompt-aware masking (only steer continuation
  tokens, not prompt/instruction tokens) — this IS the right instinct and aligns with
  the paper's finding that prompt tokens should not be uniformly intervened on.
- **Applicability:** Our existing prompt-aware masking (steering only the generated
  continuation, leaving system/instruction tokens untouched) is validated as correct by
  this work. The further step — per-continuation-token variable coefficients — is a
  future enhancement, not a blocker. Keep the masking; it matters.

## Finding 8 — No standardized benchmarks; methods aren't comparable

- **Source:** Braun et al., "A Sober Look," Nov 2024 (section "Methods are not compared
  on the same benchmarks and metrics").
  + Brumley et al., 2024 (ICVs vs FVs — each wins only on its own task type).
- **Blocker:** Each steering paper uses custom datasets; cross-method comparison is
  essentially impossible. "Without a universal benchmark... it remains unclear how well
  steering methods actually generalize outside of these specific setups."
- **Resolution:** Community-level unresolved. For us: keep a fixed internal eval corpus
  (the A/B prompt files + held-out contrast set) stable across the Llama->Qwen migration
  so our before/after numbers are comparable.
- **Applicability:** Don't change the eval prompt set when migrating; only change the
  model and recompute vectors. This isolates the model-swap effect.

---

## Migration checklist (Llama-3.1-8B -> Qwen2.5-32B)

1. Recompute ALL trait vectors from the A/B prompt corpus on Qwen2.5-32B. Llama vectors
   are dimensionally and representationally incompatible. (Finding 1)
2. Run a per-trait layer sweep across Qwen's 64 layers; regenerate
   `configs/steering.layers.yaml`. Do not port Llama layer indices. (Finding 4)
3. Re-tune the global `steering.strength` scalar — start lower; 32B is more sensitive
   to oversteering. (Finding 6)
4. Add anti-steerable fraction + directional agreement / d' to `steering.eval` output.
   (Finding 2)
5. Add a generation-based eval (LLM-judge on Big Five Likert) before launching any trait.
   Logit-diff alone overestimates effectiveness. (Finding 5)
6. Measure cross-trait bleed on Qwen; expect it. Do NOT assume independent trait control
   even with high trait-alpha orthogonality in PersonaCoeffs. (Finding 3)
7. Keep prompt-aware masking (continuation-only steering). It is validated correct.
   (Finding 7)
8. Keep the eval prompt corpus fixed across the migration for comparability. (Finding 8)
9. Re-verify per-trait steerability from scratch — Neuroticism-style model-specific
   failures (strong on Llama, ~zero on Mistral) can appear on any trait on Qwen.
   (Findings 2, 3)

## Key sources (sorted by relevance)

1. Bhandari et al. 2026 — Personality trait entanglement (Big Five, LLaMA+Mistral).
   https://arxiv.org/html/2602.15847  [HIGHEST RElevance — same task setup]
2. Ayyub 2026 — Qwen3 family steering, layer-depth, logit-vs-generation divergence.
   https://omar.bet/2026/01/17/What-I-Learned-Steering-Qwen3-Models/  [Qwen-specific]
3. Tan et al. 2024 (NeurIPS) — Reliability, anti-steerability.
   https://arxiv.org/abs/2407.12404
4. Braun et al. 2025 — (Un)reliability, prompt-type, directional agreement, d'.
   https://arxiv.org/html/2505.22637v1
5. Braun et al. 2024 — "A Sober Look" overview (evaluation + degradation + benchmarks).
   https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/
6. Pres et al. 2024 (MINT@NeurIPS) — Reliable evaluation pipeline (4 properties).
   https://arxiv.org/abs/2410.17245
7. Heyman & Vandeputte 2026 (ICML) — Token-specific steering, prompt-masking validation.
   https://arxiv.org/html/2605.03907v1
8. Rimsky (Panickssery) et al. 2024 (ACL) — Original CAA paper.
   https://arxiv.org/abs/2312.06681
9. Stickland et al. 2024 — Capability degradation (MT-Bench).
   https://arxiv.org/abs/2406.15518
