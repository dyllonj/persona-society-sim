# Prior-Work Synthesis: Blockers, Red Flags, and Pareto-Frontier Actions

Compiled 2026-07-05 from 5 parallel research agents covering: Generative Agents/Smallville,
LLM multi-agent failure modes, CAA activation steering limits, persona drift, and social
simulation validity threats. ~40 primary sources reviewed.

This document maps what other teams have already discovered and resolved so we can avoid
rediscovering them. Items are prioritized by impact on our GPU experimentation readiness.

---

## P0: Hard blockers for GPU experiments (must address before A100 runs)

### B1. CAA vectors do NOT transfer across models
**Source:** Rimsky et al. ACL 2024; Ayyub 2026 (Qwen3 steering); IBM activation-steering FAQ
**Blocker:** Steering vectors live in a model-specific residual-stream basis at a specific
layer. Llama-3.1-8B (4096-dim) and Qwen2.5-32B (5120-dim) are dimensionally incompatible.
Even within the same family, steerability is idiosyncratic and optimal layer depth shifts
with training method (RL models peak at 70-85% depth vs 50-65% for distilled).
**Resolution:** Vectors MUST be recomputed from scratch on Qwen2.5-32B. Layer indices must
be re-tuned via layer sweep. Treat Llama vectors and `configs/steering.layers.yaml` as
throwaway. Run `scripts/compute_vectors.sh` + `steering/layer_sweep.py` on the A100 before
any persona trial.
**Status in our codebase:** `steering.layers.yaml` already names Qwen2.5-32B (Issue #4 in
known-gaps) but vectors are Llama-3.1-8B. The YAML config is aspirational, not operational.

### B2. Evaluation metrics overestimate steering effectiveness
**Source:** Pres et al. MINT@NeurIPS 2024; Ayyub 2026; Braun et al. 2024/2025
**Blocker:** Logit-diff/MCQA evaluations (what our `steering.eval` harness uses) do not
predict generation-based behavioral change. Ayyub: sycophancy logit effect 1.04 -> only 8%
behavior change. CAA is "less effective than previously reported" under rigorous eval.
**Resolution:** Add a generation-based eval: have steered models produce free text, judge
with a separate LLM on Big Five Likert items. Do not launch a trait on logit-diff alone.
Keep eval prompt corpus fixed across migration for comparability.
**Status:** Our eval harness (Issue #3 in known-gaps) doesn't even have held-out prompts yet.

### B3. Steering degrades fluency (inverted-U curve)
**Source:** Bas & Novak 2026; Stickland et al. 2024; Bhandari et al. 2026
**Blocker:** Trait adherence rises then falls as steering coefficient grows; coherence drops
monotonically and approaches zero at large coefficients. Our global `steering.strength`
scalar multiplies every trait uniformly, but different traits have different peak coefficients.
**Resolution:** Grid-search per-trait coefficients. Start lower on 32B (more sensitive to
oversteering). Report fluency/coherence degradation alongside trait expression. Add a
coherence guard to detect "high trait, broken prose" failure mode.

### B4. Anti-steerability: ~1/3 of samples steer the wrong way
**Source:** Tan et al. NeurIPS 2024; Braun et al. 2025
**Blocker:** 3-50% of samples produce reverse-effect steering. Aggregate metrics mask this
because positive and negative per-sample effects cancel. Some behaviors are effectively
unsteerable.
**Resolution:** Add anti-steerable fraction per trait to eval output. Use directional
agreement (cosine sim of activation diffs) and separability (d') as pre-flight diagnostics.
Report per-sample variance, not just mean effect.

### B5. Personality trait vectors are entangled
**Source:** Bhandari et al. 2026 (directly on Big Five steering)
**Blocker:** Steering one trait moves others. Hard orthonormalization enforces geometric
orthogonality but does NOT eliminate behavioral bleed (Bmax ~3.0 on 1-5 Likert even with
hard orthonormalization) AND degrades fluency. Model-specific failures (Neuroticism
steerable on LLaMA, ~zero on Mistral).
**Resolution:** Measure cross-trait bleed. Consider C4-style soft projection if bleed is
problematic. Re-verify per-trait steerability on Qwen from scratch. Do NOT assume
orthogonal trait alphas produce orthogonal behavior. Note: our `enforce_orthogonality()`
in `compute_caa.py` is a rejection-based check (cosine threshold 0.2), not a transform.

---

## P1: Critical red flags for result validity (must instrument before trusting results)

### R1. Persona drift is measurable within ~8 conversation turns
**Source:** Li et al. ICML 2024; Choi et al. 2024; SPASM/Luo & Laban ACL 2026
**Blocker:** Significant persona drift appears within 8 rounds of self-chat. Agents not only
lose their own persona but ADOPT the conversational partner's persona ("echoing"). Over
long horizons this leads to identity collapse. Larger models drift MORE (counterintuitive).
Root cause: attention decay to system-prompt tokens; instruction-tuned models snap back
toward "helpful assistant" persona.
**Resolution:** (a) Our CAA steering re-injects trait vectors each generation, which should
counteract attention decay. (b) Adopt ECP (Egocentric Context Projection) from SPASM: store
dialogue history perspective-agnostically, re-project per agent (SELF vs PARTNER). (c) Add
persona-stability curve: re-issue fixed persona-probe at intervals, measure embedding
distance from baseline. (d) Track population-level trait-variance-over-tick; if variance
collapses, agents are converging.

### R2. Action diversity collapse amplified by steering
**Source:** Yu et al. ICML 2024 (Affordable Generative Agents); Xtra-Computing ACL 2026
**Blocker:** LLM agents "can only generate finite behaviors in fixed environments." Strong
retrieval consistency makes agents converge on profile-aligned repetitive actions (top event
repeated 78x in one study). Our CAA steering amplifies trait consistency, compounding this.
Dense communication topologies accelerate premature convergence. Alignment acts as a semantic
regularizer.
**Resolution:** (a) Add "mind wandering" stochastic perturbation: occasionally inject
low-relevance random memories into retrieval to break convergence. (b) Measure action-type
entropy over the run as a diversity metric. (c) Consider NGT-style independent reflection
phases before social exposure. (d) Test whether high-steering agents STILL collapse.

### R3. Monoculture collapse: same base model = correlated vulnerabilities
**Source:** Reid et al. (Gradient Institute) 2025; Xtra-Computing ACL 2026
**Blocker:** 32-300 agents on one HF backbone = maximum monoculture-collapse risk. Steering
perturbs the residual stream but not shared training-data priors. All agents can misread the
same environmental cue. Conformity bias: agents reinforce each other's errors. Sycophancy
cascades: multi-agent networks amplify errors ~17x (DeepMind).
**Resolution:** (a) Run a no-steering baseline as control. (b) Consider controlled
heterogeneity (different model checkpoints or quantization levels). (c) Instrument
theory-of-mind gaps via cognitive-trace logs (does agent A's plan account for B's utterance?).
(d) Check cooperation-rate metrics against a no-social-interaction control.

### R4. "Emergent" behavior may be training-data leakage
**Source:** Barrie & Tornberg 2025; Larooij & Tornberg 2026; PIMMUR/Zhou et al. 2025
**Blocker:** LLMs trained on scientific literature may reproduce known social dynamics
(cooperation, polarization, echo chambers) from training data rather than genuinely
self-organizing. 50.8% of frontier LLMs correctly identify the underlying experiment.
89.7% of LLM social sim studies violate at least one PIMMUR validity principle. 61% of
prompts exert excessive control (goal-injection).
**Resolution:** (a) Audit whether our LLM recognizes our experimental setup (query it
directly). (b) Make NOVEL predictions (steering strength modulates cooperation) as
validation target, not replication of known phenomena. (c) Use novel synthetic content
(our research-sprint facts help). (d) Keep behavioral probe prompts neutral. (e) Report
contamination risk as limitation.

### R5. LLMs lack behavioral heterogeneity ("average persona" problem)
**Source:** Wu et al. 2025; PNAS/Gao et al. 2025
**Blocker:** LLMs systematically underrepresent behavioral variance. They behave "nicer
than humans" in economic games. Distributional metrics (Gini, polarization) require variance;
if agents collapse toward average, these metrics are artifacts of model homogeneity, not
social dynamics.
**Resolution:** (a) Report behavioral variance alongside mean alignment for all metrics.
(b) Compare against human normative data (IPIP distributions, public goods game meta-analyses).
(c) Validate that persona steering produces SUFFICIENT variance, not just different means.

### R6. Reflection/summarization causes semantic drift
**Source:** Lam et al. 2026 (SSGM); Rath 2026 (Agent Drift)
**Blocker:** Repeated summarization (reflection) causes cumulative, persistent semantic
drift. Agent drift detectable after median 73 interactions; drifted systems show -42% task
success, +487% inter-agent conflicts. Memory system itself can cause drift via role-label
ambiguity and feedback loops. Drift occurs WITHOUT parameter updates (autoregressive
feedback).
**Resolution:** (a) Add consistency-check gate before committing reflections to MemoryStore
(does reflection contradict stored observations?). (b) Periodic memory consolidation/pruning
(51.9% drift reduction). (c) Adaptive behavioral anchoring: re-ground in baseline persona
when drift spikes (70.4% reduction). (d) Ground-truth anchoring against env action log.

### R7. Believability != accuracy
**Source:** Lu et al. (Amazon) 2025; Serapio-Garcia et al. Nature MI 2025
**Blocker:** First large-scale quantitative benchmark found only 11.86% next-action accuracy
for LLM agents vs real human behavior. Prior sims validated only qualitative "believability."
Base (non-instruct) models categorically fail psychometric validity. Personality measurement
shows limited temporal stability.
**Resolution:** (a) Don't rely on "does this look like a society?" checks. (b) Benchmark
agent action distributions against real human data. (c) Report Cronbach's alpha on probe
batteries per agent. (d) Report test-retest stability, not single-shot scores. (e) Require
instruction-tuned backend for valid personality measurement.

### R8. Memory retrieval uses fixed heuristic (Park et al. limitation)
**Source:** Park et al. UIST 2023; Larooij & Tornberg 2026
**Blocker:** Our MemoryRetriever uses the same recency+importance+relevance recipe with all
alpha=1 that Park et al. flagged as the most common error source. Retrieval-quality decays
at scale.
**Resolution:** (a) Make retrieval persona-conditioned (retrieve memories consistent with
trait profile, not just recency/importance). (b) Consider learned/graph-based retrieval
(HippoRAG, A-MEM) for future work. (c) Run sensitivity sweep on retrieval alpha weights.

### R9. 14 multi-agent failure modes (MASFT taxonomy)
**Source:** Cemri et al. 2025 (Berkeley, arXiv:2503.13657)
**Blocker:** 14 distinct failure modes across 3 categories. SOTA open-source MAS correctness
as low as 25%. Key risks for us: step repetition (loops), reasoning-action mismatch,
conversation reset, ignored peer input, loss of conversation history.
**Resolution:** (a) Run their open-source LLM-as-judge annotator over ActionLog traces.
(b) Add per-agent token budget + circuit breaker for loop detection. (c) Structural
redesign required, not just prompt engineering.

### R10. Circular validation: using LLMs to evaluate LLMs
**Source:** Larooij & Tornberg 2026; Panickssery et al. 2024
**Blocker:** LLM evaluators recognize and favor their own generations. If we use LLM-based
rubric scorers for behavioral probes, we have circularity.
**Resolution:** Use human raters for at least a validation subset. If using LLM scorers,
use a different model family and validate against human ratings first.

---

## P2: Infrastructure and scaling recommendations

### I1. Cost/throughput at 100+ agents
**Source:** Park et al. 2023; AgentSociety ACL 2025; OASIS 2024
**Blocker:** Naive Park-style architecture doesn't reach 100+ economically. Agent-agent
interactions scale quadratically.
**Resolution:** (a) Cache routine plans (Lifestyle Policy, ~40% cost reduction). (b)
Compress dialogue context into relationship summaries (Social Memory). (c) Tiered
LLM/rule policies for routine perceive steps. (d) Parallel inference (AgentSociety: 30k
agents, 24 GPUs, linear scaling).

### I2. Hallucination cascades poison MemoryStore
**Source:** Project Sid/Altera 2024
**Blocker:** A single hallucinated observation persists in MemoryStore and influences
retrieval forever. Hallucinations spread through inter-agent conversation.
**Resolution:** (a) Action-awareness check (expected vs. observed env outcome). (b) Loop
detector on repeated action types. (c) Ground-truth anchoring against env action log.

### I3. Token cost explosion (quadratic context growth)
**Source:** Augment Code 2026; Medium/Token Cost Trap 2025
**Blocker:** Naive agent loops grow token costs quadratically as history accumulates.
**Resolution:** (a) Our MemoryStore + retrieval-based recall is the right defense. (b)
Verify `_build_prompt` doesn't include growing raw transcript. (c) Add per-tick token cap.
(d) Per-agent cumulative-token-spent counter with auto-terminate.

---

## Quick-reference: Actions ranked by effort/impact

| Action | Effort | Impact | Category |
|--------|--------|--------|----------|
| Recompute vectors + layer sweep on Qwen | High (GPU) | Critical | P0-B1 |
| Add held-out eval prompts | Medium | Critical | P0-B2 |
| Add generation-based eval (LLM-judge) | Medium | Critical | P0-B2 |
| Grid-search per-trait steering strength | Medium | Critical | P0-B3 |
| Add anti-steerable fraction to eval | Low | High | P0-B4 |
| Measure cross-trait bleed | Low | High | P0-B5 |
| Run no-steering baseline control | Low | Critical | P1-R3/R4 |
| Add persona-stability curve metric | Medium | High | P1-R1 |
| Add action-type entropy metric | Low | High | P1-R2 |
| Add mind-wandering retrieval perturbation | Medium | High | P1-R2 |
| Add reflection consistency-check gate | Medium | High | P1-R6 |
| Add periodic memory consolidation | Medium | Medium | P1-R6 |
| Audit LLM for experiment recognition | Low | High | P1-R4 |
| Report variance + Cronbach's alpha | Low | High | P1-R5/R7 |
| Add per-agent token budget + circuit breaker | Medium | Medium | P2-I3 |
| Run MASFT annotator on traces | Medium | Medium | P1-R9 |
| Consider ECP for perspective-agnostic memory | High | High | P1-R1 |
| Add hallucination/loop detection | Medium | Medium | P2-I2 |

---

## Key papers for citation

1. Park et al. 2023 - Generative Agents (UIST) - arXiv:2304.03442
2. Larooij & Tornberg 2026 - Validation critical review (Springer AI Review) - doi:10.1007/s10462-025-11412-6
3. Barrie & Tornberg 2025 - Data leakage - arXiv:2505.23796
4. Zhou et al. 2025 - PIMMUR principles - arXiv:2509.18052
5. Wu et al. 2025 - Average persona / boundary - arXiv:2506.19806
6. Li et al. 2024 - Persona drift measurement (ICML) - arXiv:2402.10962
7. SPASM/Luo & Laban 2026 - Echoing + ECP (ACL) - arXiv:2604.09212
8. Serapio-Garcia et al. 2025 - Psychometric framework (Nature MI) - doi:10.1038/s42256-025-01115-6
9. Bhandari et al. 2026 - Big Five trait entanglement - arXiv:2602.15847
10. Tan et al. 2024 - Steering reliability (NeurIPS) - arXiv:2407.12404
11. Pres et al. 2024 - Reliable eval (MINT@NeurIPS) - arXiv:2410.17245
12. Rimsky et al. 2024 - Original CAA (ACL) - arXiv:2312.06681
13. Cemri et al. 2025 - MASFT 14 failure modes - arXiv:2503.13657
14. Rath 2026 - Agent drift quantification - arXiv:2601.04170
15. Lam et al. 2026 - SSGM memory governance - arXiv:2603.11768
16. Ong et al. 2025 - Cooperative personalities via steering - arXiv:2503.12722
17. Ichinose et al. 2026 - Steering as bias not control - arXiv:2601.05302
18. Lu et al. 2025 - Believability != accuracy (Amazon) - arXiv:2503.20749
19. Reid et al. 2025 - Governed MAS risk analysis - arXiv:2508.05687
20. Yu et al. 2024 - Affordable Generative Agents (ICML) - arXiv:2402.02053
