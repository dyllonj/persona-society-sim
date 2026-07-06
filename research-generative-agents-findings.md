# Prior-Work Research: Generative Agents / Smallville and Successors

Compiled findings for the persona society simulation project. Each entry lists source,
blocker, resolution (if any), and applicability to our architecture (Big Five + CAA steering,
perceive-reflect-plan-act loop, memory store + retrieval, social graph metrics, self-report
and behavioral probes, research-sprint environment).

Legend: APPLIES = directly relevant risk we should plan for; PARTIAL = relevant but conditional;
LOW = not directly applicable to our stack.

---

## 1. Original Generative Agents (Park et al. 2023) — architecture, limitations, costs

### 1.1 Memory-stream retrieval is a fixed hand-tuned heuristic
- Source: Park, J.S. et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST '23 / arXiv:2304.03442. https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763
- Blocker: Retrieval score = α_rec·recency + α_imp·importance + α_rel·relevance, with all α set to 1 and min-max normalization. Authors flag this as needing "extensive manual tuning of storage and summarization criteria" and that the system "could not learn from memory management errors." The retrieval function is the most common error source in their own evaluation: "the most common errors arose when the agent failed to retrieve relevant memories."
- Resolution: Park et al. only suggest future work: fine-tune the recency/relevance/importance functions. Later work (HippoRAG, A-MEM, Memory-R1) replaces fixed heuristics with graph-based spreading activation or RL-learned memory ops.
- Applies: APPLIES. Our MemoryStore + MemoryRetriever uses the same recency/importance/relevance recipe (per AGENTS.md, importance is length-derived and recency is recomputed from tick). Expect retrieval-quality decay at scale and consider learned/graph retrieval.

### 1.2 Cost: thousands of dollars for 25 agents over 2 days
- Source: Park et al. 2023 (same), Discussion + Future Work.
- Blocker: "The present study required substantial time and resources to simulate 25 agents for two days, costing thousands of dollars in token credits and taking multiple days to complete." Cost scales with number of LLM calls per agent per tick (perceive + reflect + plan + act + dialogue).
- Resolution: Authors suggest parallelizing agents or models designed for generative agents. Affordable Generative Agents (Yu et al. 2024) cuts cost via Lifestyle Policy (reuse cached plans) + Social Memory (compress dialogue context), reporting ~40% of baseline token cost with equivalent believability.
- Applies: APPLIES. We run open HuggingFace models (Llama-3.1-8B, Qwen2.5-32B) so per-token cost is compute rather than API spend, but inference throughput is the binding constraint at 100+ agents. Lifestyle-Policy-style caching of repetitive perceive/plan steps is directly applicable.

### 1.3 Reflection triggers and reflection quality untested at long horizons
- Source: Park et al. 2023, Reflection section; evaluation limited to "a relatively short timescale."
- Blocker: Reflections are triggered when cumulative recent-memory importance exceeds a threshold, then synthesized recursively. Authors admit evaluation "was limited to a relatively short timescale" and that long-run robustness "is still largely unknown," including "memory hacking" and hallucination.
- Resolution: Unresolved in the original paper.
- Applies: APPLIES. Our reflect_and_plan runs every tick; reflection quality degradation over long runs is a primary risk (see §5).

### 1.4 Overly formal / generic speech inherited from instruction-tuned base model
- Source: Park et al. 2023, footnote 3.
- Blocker: "The conversational style of these agents can feel overly formal, likely a result of instruction tuning in the underlying models." This is a persona-consistency leak from the base model's RLHF persona.
- Resolution: Authors expect future models to be more controllable. Persona-drift literature (Lu et al. 2026 "Assistant Axis") shows this is the base model's default "Assistant" persona bleeding through.
- Applies: APPLIES. Our CAA steering vectors are explicitly designed to counter this, but prompt-aware masking leaves system/instruction text un-steered, so the formal/Assistant register can still leak into utterances. Worth measuring.

---

## 2. Replication / follow-up papers that identified problems

### 2.1 Validation is the central unresolved challenge (systematic review)
- Source: Larooij, M. & Törnberg, P. "Validation is the central challenge for generative social simulation: a critical review of LLMs in agent-based modeling." Artificial Intelligence Review 59:15, 2026. https://link.springer.com/article/10.1007/s10462-025-11412-6
- Blocker: Systematic review of the generative-ABM literature. Key findings: (a) LLMs "may exacerbate rather than alleviate" validation problems because of black-box structure, stochasticity, and cultural bias; (b) most studies rely on "face-validity or outcome measures that are only loosely tied to underlying mechanisms"; (c) generative ABMs "occupy an ambiguous methodological space—lacking both the parsimony of formal models and the empirical validity of data-driven approaches"; (d) ABM interactions "scale quadratically with the number of agents, whereas sensitivity analyses scale exponentially with the number of parameters"; (e) prior ABM field had a "replication crisis" where "some findings turned out to be the results of software bugs."
- Resolution: Calls for operational validity = purpose alignment + external grounding (human data / pre-registered benchmarks, not face-validity) + robustness (multiple runs + sensitivity checks). No turnkey fix.
- Applies: APPLIES. Our probe + cognitive-trace logging is exactly the kind of instrumentation this review demands; we should pre-register expected trait→behavior mappings and run sensitivity sweeps on steering.strength and retrieval α weights.

### 2.2 "Emergent" behavior may be indistinguishable from training-data leakage
- Source: Barrie, C. & Törnberg, P. "Emergent LLM behaviors are observationally equivalent to data leakage." arXiv:2505.23796, 2025.
- Blocker: Shows that apparent emergent social norms in LLM naming-game simulations are observationally equivalent to the models simply reproducing phenomena described in their training corpus. Since LLMs are trained on scientific literature, "what may appear as emergent dynamic can instead stem from a form of 'data leakage'." This threatens the validity of any emergent-behavior claim (norms, polarization, cooperation).
- Resolution: Authors argue researchers must design settings with "little or no historical precedent" and actively distinguish emergence from memorization. No automated detector proposed.
- Applies: APPLIES, and sharp. Our research-sprint environment (novel fact corpus, citation tracking) is a good countermeasure because facts are not in training data, but social-dynamics claims (cooperation/Gini/polarization) are vulnerable to leakage. Use novel synthetic content and report memorization checks.

### 2.3 Heuristic "validate-then-simulate" lacks statistical guarantees
- Source: Hullman, J. et al. "This human study did not involve human subjects: Validating LLM simulations as behavioral evidence." arXiv:2602.15785, 2026.
- Blocker: Heuristic validation (show LLM replicates direction/significance of some human effects, then treat LLM and human as interchangeable) "cannot, even under optimistic conditions, guarantee the absence of systematic biases." Concrete failures: LLMs replicate direction of up to 81% of main effects but produce significant results for up to 83% of effects that were NOT significant in humans; LLMs "overestimate human effect sizes"; LLMs "better conform to rationality and theoretical predictions" than humans do (a red flag, not a feature).
- Resolution: Proposes statistical calibration: combine a smaller human sample with a larger LLM sample under explicit assumptions to get unbiased, more precise causal estimates. Useful for confirmatory claims; heuristic substitution is fine only for exploratory work.
- Applies: PARTIAL. Our project is exploratory/sim-building, so heuristic validation is acceptable, but any quantitative RQ claims (RQ2/RQ4) should not lean on face-validity alone. Consider a small human-benchmark calibrator for probe rubrics.

### 2.4 Concordia (DeepMind) — generative ABM framework and its documented limits
- Source: Vahid, A. et al. "Generative agent-based modeling with actions grounded in physical, social, and digital spaces." arXiv:2312.03664 (Concordia), DeepMind 2023–2025. https://github.com/google-deepmind/concordia ; plus "Designing Reliable Experiments with Generative Agent-Based Modeling" arXiv:2411.07038, 2024.
- Blocker: Concordia is the main successor framework. Documented issues: (a) experiments are hard to design reliably — a whole companion paper is dedicated to "challenges when conducting large-scale experiments, particularly due to the simulations' complexity"; (b) components are non-deterministic and sensitive to wording, making reproducibility hard; (c) no standardized benchmarks across studies.
- Resolution: Concordia provides a modular GM/agent structure and emphasizes logging + component isolation. The companion paper offers reliability protocols (component-level unit tests, repeated runs).
- Applies: APPLIES. We should adopt component-level deterministic tests (our steering.eval harness is a start) and report run-to-run variance, not just means.

### 2.5 CRSEC norm-emergence: norm-utility clustering and semantic drift in auto-generated norms
- Source: Ren et al. "Emergence of Social Norms in Generative Agent Societies: Principles and Architecture" (CRSEC). arXiv:2403.08251, 2024.
- Blocker: Automated norm generation from agent thoughts "can be repetitive or introduce semantic drift"; immediate norm evaluation "may overestimate norm importance, resulting in utility scores clustering at high values" (ceiling effects). Reports 100% norm compliance within 2 days, which itself is a face-validity red flag (too clean).
- Resolution: Long-term synthesis compresses norm sets when aggregate utility exceeds threshold; sanity-check gates (consistency, uniqueness, conflict-freeness) filter candidate norms.
- Applies: PARTIAL. We do not run a norm-emergence module, but the lesson (auto-generated rubric/scoring items cluster and drift) applies to our behavioral-probe rubric generation. Add diversity + conflict checks to auto-generated probe rubrics.

---

## 3. Scaling blockers from ~25 agents to 100+ agents

### 3.1 Hallucination cascades and action loops poison downstream behavior
- Source: Altera.al (Ahn et al.) "Project Sid: Many-agent simulations toward AI civilization." arXiv:2411.00114, 2024.
- Blocker: At 50–1000 agents the dominant failure modes are (a) hallucination cascades: "even a small rate of hallucinations can poison downstream agent behavior when agents continuously interact with the environment via LM calls" (e.g., agent claims it ate a bagel it never ate, then never seeks food); (b) action loops: "agents often become stuck in repetitive patterns of actions or accumulate a cascade of errors"; (c) inter-agent hallucination propagation: one agent's hallucination spreads through conversation. At 1000 agents, runs "exceeded the computational constraints of our Minecraft server environment, causing agents to be sporadically unresponsive."
- Resolution: PIANO architecture adds an Action Awareness module that compares expected vs. observed action outcomes to ground the agent, and a Cognitive Controller bottleneck to keep multiple output streams coherent. These materially reduced loops but did not eliminate them.
- Applies: APPLIES strongly. Our perceive→reflect→plan→act loop re-feeds observations into MemoryStore every tick, so a single hallucinated observation persists and influences retrieval. An action-awareness check (expected vs. actual env outcome) and a loop detector on repeated action types are high-value additions.

### 3.2 Behavioral diversity collapses: "finite behaviors in fixed environments"
- Source: Yu, Y. et al. "Affordable Generative Agents." arXiv:2402.02053, ICML 2024.
- Blocker: Empirically shows LLM agents "can only generate finite behaviors in fixed environments." DBSCAN clustering of one agent's 740 non-idle memory events found 506 aligned with the profile setup; the most frequent event ("conversing about the Valentine's Day party") occurred 78 times. The retrieval framework's strong recency+relevance consistency "results in agent's limited behavior" — agents converge on profile-aligned repetitive actions.
- Resolution: "Mind wandering" module — randomly sample unrelated memory events (weighted by inverse cluster size) and inject into the prompt to perturb decisions, measurably increasing behavior richness.
- Applies: APPLIES. This is the action-diversity-collapse red flag. Our steering vectors intensify trait-consistent behavior, which could worsen convergence. A mind-wandering-style stochastic perturbation in retrieval (occasionally injecting low-relevance memories) is a cheap mitigation we should test.

### 3.3 Cost scales linearly with interactions; quadratic for agent-agent interactions
- Source: Larooij & Törnberg 2026 (review) + Yu et al. 2024 (AGA).
- Blocker: Agent-agent interactions scale ~quadratically with N; per-interaction LLM cost scales with prompt length, which grows as memory/context accumulates. AGA reports Lifestyle Policy alone cut cost to 40.2% of baseline and Social Memory compressed dialogue context further.
- Resolution: Cache repeated perceive/plan inferences (Lifestyle Policy); compress inter-agent dialogue context into Relationship/Feeling/Summary triples (Social Memory).
- Applies: APPLIES. Our talk/work/research/gift edges create N^2 interaction potential. Caching routine plans and compressing dialogue history into relationship summaries aligns with our existing edge/social-graph design.

### 3.4 Multi-agent coherence degrades: "say one thing, do another"
- Source: Project Sid (PIANO), §1.3 Reason 2.
- Blocker: When an agent has independent LLM modules (e.g., chat vs. function-calling), outputs diverge: chat says "Sure thing!" while the action module chooses "explore." "Incoherence also scales exponentially as the number of independent output modules increases." This misleads other agents and causes group dysfunction.
- Resolution: A single Cognitive Controller makes the high-level decision and broadcasts it to condition all downstream output modules (coherence via bottleneck).
- Applies: PARTIAL. Our planner produces an ActionDecision (action type + params + utterance) in one pass, which is already coherent-by-construction, but the reflect step and the plan step are separate LLM calls that can drift apart; ensure plan→utterance guidance is enforced.

### 3.5 OASIS / AgentSociety: scaling to 10k–1M agents requires architecture-level changes
- Source: Yang et al. "OASIS: Open Agents Social Interaction Simulations on one million agents." arXiv:2411.11581, 2024. Piao et al. "AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents." arXiv:2502.08691, 2025.
- Blocker: Scaling to thousands+ requires abandoning per-agent per-tick full LLM calls for most agents; they use hybrid LLM+rule policies, message-graph partitioning, and approximate retrieval. Naive Park-style architectures do not reach 100+ agents economically.
- Resolution: Hybrid agent policies (LLM for salient moments, cheaper models/rules for routine), structured social-network backends, and distributed orchestration.
- Applies: APPLIES if we target 100+. For 100+ we will likely need a tiered policy (steered Llama-3.1-8B for plan/act, lightweight rule/SLM for routine perceive) and batched/parallel inference.

---

## 4. Interview / probe methodologies for evaluating LLM-agent personality consistency

### 4.1 InCharacter: interview the agent with psychological scales
- Source: Wang, X. et al. "InCharacter: Evaluating Personality Fidelity in Role-Playing Agents through Psychological Interviews." ACL 2024. https://aclanthology.org/2024.acl-long.102/ ; arXiv:2310.17976.
- Blocker: Prior RPA evaluation focused on knowledge/linguistic patterns, not personality. Self-report assessments on RPAs have drawbacks (the agent can rationalize). Need a method that probes whether the agent's expressed personality matches the target.
- Resolution: "Interviewing Character agents for personality tests" — administer standard psychological scales (Big Five / IPIP) by conversational interview, then score. Across 32 characters and 14 scales, state-of-the-art RPAs achieved up to 80.7% alignment with human-perceived character personality.
- Applies: APPLIES directly. This is the canonical methodology for our self-report probes. Our ProbeManager self-report probes (Likert questions mapped to IPIP traits) are essentially InCharacter; we should adopt their scoring/validity checks and report trait-alignment accuracy as a probe-quality metric.

### 4.2 Psychometric framework: reliability/validity of LLM personality measurement
- Source: Serapio-García, G. et al. "A psychometric framework for evaluating and shaping personality traits in large language models." Nature Machine Intelligence 7:1954–1968, 2025. https://doi.org/10.1038/s42256-025-01115-6
- Blocker: Naively giving an LLM a personality test does not yield a valid measurement. Construct validity (reliability + convergent/discriminant/criterion validity) must be established. Key findings: (a) reliability/validity is "stronger for larger and instruction-fine-tuned models" — base models categorically fail; (b) personality can be shaped along desired dimensions via prompting at 9 levels × 104 trait adjectives; (c) shaped personality "verifiably influences LLM behaviour in common downstream tasks."
- Resolution: Structured prompting that varies demographic/contextual/linguistic conditions across ~1,250 administrations, then standard psychometric analysis (Cronbach's α, CFA, convergent/discriminant validity against a second inventory like BFI). Provides a validated shaping protocol.
- Applies: APPLIES directly. We use CAA steering rather than prompt shaping, but the validation methodology (multi-administration, α, convergent validity between IPIP and BFI, criterion validity via downstream task) is the right template for validating that our steering vectors actually produce the intended trait expression. Critically, base (non-instruct) models fail validity — relevant since we steer Llama-3.1 base/instruct variants.

### 4.3 Personality testing shows limited temporal stability
- Source: "Personality testing of large language models: limited temporal stability." Royal Society Open Science 11:10, 240180, 2024. https://royalsocietypublishing.org/rsos/article/11/10/240180
- Blocker: LLM personality test-retest reliability and temporal stability are limited — responses drift across administrations and over time, undermining single-shot personality claims.
- Resolution: Recommends repeated longitudinal administration and reporting stability coefficients rather than one-shot scores.
- Applies: APPLIES. Our probes run on a cadence; we should report test-retest stability per agent and treat single-tick probe scores as noisy. This also motivates tracking persona drift over the run.

### 4.4 Persona drift is measurable via activation-space projection (directly relevant to CAA steering)
- Source: Lu et al. "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models." arXiv:2601.10387, 2026. Chen et al. "Persona Vectors: Monitoring and Controlling Character Traits in Language Models." arXiv:2507.21509, 2025. Platnick et al. "ID-RAG: Identity Retrieval-Augmented Generation for Long-Horizon Persona Coherence." arXiv:2509.25299, 2025. (Summarized via emergentmind.com/topics/persona-drift.)
- Blocker: Persona drift = gradual deviation from assigned persona over long/multi-turn interactions. Measured as drop in projection onto a persona axis (PCA-extracted) or persona-consistency (NLI) scores. Causes: loose post-training tethering, contextual dilution/memory overwriting, latent collapse. Empirically, 20–40% drops in Assistant-axis projection over 10–15 turns; identity recall decays steadily in baseline generative agents over multi-step simulations.
- Resolution: Activation capping along the persona axis (reduces harmful drift ~60% with <2% capability drop); persona-vector steering at inference (subtract/add trait vectors — exactly CAA-style); ID-RAG retrieval of identity facts at each step (8–12% higher identity recall at run end vs. baseline decay).
- Applies: APPLIES, and validates our core approach. Our CAA steering vectors ARE persona-vector steering, so we are already using the recommended mitigation. The activation-capping idea (clamp residual magnitude) and ID-RAG (re-inject persona facts each step) are complementary safeguards we should consider. The measurement methodology (axis projection drift over the run) is a ready-made drift metric for our MetricTracker.

### 4.5 Sotopia / structured social-intelligence evaluation
- Source: Zhou, X. et al. "Sotopia: Interactive Evaluation for Social Intelligence in Language Agents." ICLR 2024.
- Blocker: Evaluating social behavior needs interactive, goal-directed scenarios, not just questionnaires.
- Resolution: Sotopia provides an interactive evaluation framework with scenario-driven social tasks and multi-dimensional scoring.
- Applies: PARTIAL. Our behavioral probes (scripted NPC actions, civic events, rubric scorers) are the same pattern; Sotopia's multi-dimensional scoring rubric is a useful template.

---

## 5. Red flags: memory store, reflection quality degradation, action diversity collapse

### 5.1 Semantic drift through iterative summarization (reflection degrades memory)
- Source: Lam, C. et al. "Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the SSGM Framework." arXiv:2603.11768, 2026.
- Blocker: Identifies that "an agent may gradually distort facts through repeated summarization (semantic drift), reinforce suboptimal workflows (procedural drift), or inadvertently internalize hallucinations and malicious injections as valid knowledge." Three compounding failure interfaces: input ingestion (poisoning), memory consolidation (drift), retrieval (hallucination). Explicitly names Generative Agents' reflection as a summarization step subject to this: errors "are cumulative and persistent," unlike static RAG.
- Resolution: SSGM proposes decoupling memory evolution from governance: pre-consolidation consistency verification, temporal decay modeling, ground-truth anchoring, and dynamic access control before any memory write.
- Applies: APPLIES. Our reflect_and_plan writes reflections back into MemoryStore, so each reflection is a summarization that can drift. Add a consistency-check gate before committing reflections (does the reflection contradict stored observations?) and ground-truth anchoring against the env action log.

### 5.2 Agent drift: quantified behavioral degradation over extended interactions
- Source: Rath, A. "Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions." arXiv:2601.04170, 2026.
- Blocker: Defines and simulates "agent drift" — progressive degradation of behavior, decision quality, and inter-agent coherence over long runs. Three forms: semantic drift (intent deviation), coordination drift (consensus breakdown), behavioral drift (unintended strategies). Projected results: detectable drift (ASI<0.85) after median 73 interactions; drift accelerates (positive feedback); drifted systems show -42% task success, +216% human interventions, +487% inter-agent conflicts, +52% token usage. Behavioral-boundaries component degrades fastest (-46% over 500 interactions). Critical: drift occurs WITHOUT parameter updates — it originates in contextual conditioning and autoregressive feedback (an agent's outputs become its own future inputs via shared memory).
- Resolution: Three mitigations, evaluated in simulation: (1) Episodic Memory Consolidation (periodic compression/pruning of interaction history) — 51.9% drift reduction; (2) Drift-Aware Routing (prefer stable agents, reset drifting ones) — 63.0%; (3) Adaptive Behavioral Anchoring (re-ground in baseline exemplars, weighted by drift) — 70.4%; combined — 81.5%. Architectural note: explicit long-term memory (vector DB / structured logs) gives 21% higher ASI retention than conversation-history-only; two-level hierarchies beat flat and deep ones.
- Applies: APPLIES strongly — this is the single most on-point paper. Our long runs will hit agent drift. Concrete actions: (a) periodic memory consolidation/pruning (we already buffer via LogSink; add a consolidation pass); (b) adaptive behavioral anchoring = re-inject baseline persona/seed exemplars when drift metrics spike (pairs naturally with CAA steering, which already re-applies trait vectors each tick); (c) adopt an ASI-style drift dashboard (response consistency, tool/action selection stability, inter-agent agreement) in MetricTracker.

### 5.3 Memory corruption taxonomy: poisoning, drift, conflict/hallucination
- Source: Lam et al. 2026 (SSGM), §5 failure taxonomy.
- Blocker: Four failure dimensions: Drift (semantic/procedural), Efficiency (retrieval latency blowup as store grows), Validity (stale/contradictory facts), Safety (privacy leakage of sensitive contexts solidified into long-term storage; topology-induced knowledge leakage in multi-agent memory).
- Resolution: SSGM design principles: pre-consolidation validation, ground-truth anchoring, temporal decay, decoupled governance.
- Applies: APPLIES. Our MemoryStore grows unboundingly per agent; without decay/consolidation, retrieval latency and contradiction risk grow. Temporal decay on importance and a periodic consolidation pass are needed. The multi-agent memory-leakage finding matters if agents share observations (they do, via talk/work edges).

### 5.4 Action diversity collapse from over-consistent retrieval
- Source: Yu et al. 2024 (AGA), Appendix D (see §3.2).
- Blocker: Strong recency+relevance retrieval makes agents converge on a small set of profile-aligned repeated actions; "the strong consistency inherent in LLMs constrains the emergence of the agents' diverse behavior."
- Resolution: Mind wandering — inject low-relevance random memories to perturb plans.
- Applies: APPLIES. Our CAA steering amplifies trait consistency, compounding this. Test a controlled stochastic-retrieval perturbation and measure action-type entropy over the run as a diversity metric.

### 5.5 Identity/persona coherence decays over multi-step simulations without retrieval reinforcement
- Source: Platnick et al. "ID-RAG" arXiv:2509.25299, 2025 (see §4.4).
- Blocker: Baseline generative agents show steady decay of identity recall over simulation steps; implicit persona representations are "diluted among unrelated episodic memories" over long horizons.
- Resolution: ID-RAG — explicitly retrieve and re-inject structured identity/persona facts at each action step. Yields 8–12% higher identity recall at simulation end vs. baseline decay.
- Applies: APPLIES. Our persona is carried partly by CAA steering (re-applied each tick, which already helps) and partly by prompt persona text. Ensuring the persona profile is re-injected into every prompt (and not just relying on memory retrieval surfacing it) is a cheap, high-value safeguard.

---

## Summary: highest-priority risks for our architecture

1. Reflection/summarization semantic drift over long runs (§5.1, §5.3) — add consistency-check gates + periodic consolidation before committing reflections to MemoryStore.
2. Action diversity collapse amplified by CAA steering (§3.2, §5.4) — add stochastic retrieval perturbation (mind wandering) and track action-type entropy.
3. Agent drift over extended interactions (§5.2) — adopt ASI-style drift metrics in MetricTracker; use adaptive behavioral anchoring (re-ground in baseline persona when drift spikes), which composes well with CAA steering.
4. Persona consistency bleed-through of base-model "Assistant" register (§1.4, §4.4) — measure Assistant-axis projection drift; CAA steering is the right mitigation, add activation clamping if needed.
5. Cost/throughput at 100+ agents (§1.2, §3.1, §3.3, §3.5) — cache routine plans (Lifestyle Policy), compress dialogue context (Social Memory), tiered LLM/rule policies.
6. Emergent-behavior validity vs. data leakage (§2.1, §2.2) — use novel synthetic content (research-sprint facts help), pre-register trait→behavior mappings, run sensitivity sweeps on steering.strength and retrieval α, report run-to-run variance.
7. Probe validity (§4.1, §4.2, §4.3) — adopt InCharacter interview scoring + Serapio-García psychometric validation (multi-administration, α, convergent validity IPIP vs BFI); report test-retest stability, not single-shot scores.
8. Hallucination cascades poisoning MemoryStore (§3.1) — action-awareness check (expected vs. observed outcome) and loop detection on repeated action types.
