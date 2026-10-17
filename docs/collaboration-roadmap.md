# Collaboration evaluation roadmap

Status: research and implementation plan, 17 October 2026. This document does not report completed new experiments. Preserve the existing town, encounters, action vocabulary, and memory architecture; introduce changes as named, versioned conditions.

## Research sequence

| Version | Primary question | Scope |
| --- | --- | --- |
| V1 | Can agents collaborate effectively, and which environment settings help? | Local 8B engineering pilot, then replicated collaboration experiments. No personality intervention in the initial baseline. |
| V2 | How does collaboration differ across frontier models? | Frozen API models in the same environment, with comparable information, roles, inference budgets, and scoring. |
| V3 | How do personality composition and intervention methods affect collaboration? | Prompt personas, LoRA training, and activation steering as separate experimental methods. |

V1 must establish a useful collaboration signal before personality becomes an explanatory variable. An 8B pilot validates execution and measures that particular model; it does not establish frontier performance or human behavioral fidelity.

## What the earlier evidence does and does not establish

The [archived Hugging Face repository](https://huggingface.co/datasets/dyllonj/persona-society-jacobian-lens/tree/bd7a053ed18aab889147c34ab51bb280071b1b2d) contains code, model traces, and debug artifacts, not independent human personality labels. Its viewer exposes 6,830 action rows. The archived 360-generation Qwen2.5-32B factorial contains 60 prompt blocks across six conditions. Inspecting the first JSON object, as the legacy parser does, gives zero action-type changes for each individual trait condition and one for the combined condition relative to neutral. This is a replay intervention result, not a full-society collaboration result.

The saved debug baseline comparison reports zero task completions for targeted, placebo, and disabled steering. Therefore, excessive steering alone is not established as the explanation. Model size remains a hypothesis: model family, training, precision, prompts, and environment also differ across historical runs.

Known concerns to address or preserve explicitly as legacy behavior:

- The current CAA extraction template omits answer-option text, and training labels are not counterbalanced. Rebuild and validate vectors before personality-specific causal claims.
- Older versions let the planner choose the action while the model mainly produced speech. Current versions accept validated model-selected actions, with logged planner fallbacks.
- The metric named polarization is based on satisfaction variance. It is not opinion polarization or graph modularity.
- Several personality measures are keyword proxies. Their output should not be presented as a validated psychological trait.
- A successful action is not necessarily a completed objective, and co-presence is not evidence of productive collaboration.

## V1: local collaboration baseline

Target machine: Apple M4, 16 GiB unified memory, macOS 15.6. Start with a pinned four-bit Qwen3-8B checkpoint, three agents, short episodes, bounded context, and sequential inference. One loaded model serves multiple independently stateful agents. No cloud GPU or paid inference is required for this pilot.

Use [vLLM Metal](https://github.com/vllm-project/vllm-metal), the Apple Silicon plugin for vLLM. It uses MLX for compute and requires macOS 15 or newer. Pin both the plugin and compatible vLLM core; validate the actual checkpoint locally. The ordinary CUDA vLLM installation is not the Mac installation path. A direct MLX backend may serve as a separately labelled diagnostic fallback, never silently substituted for vLLM results.

Initial conditions keep activation steering, explicit persona prompts, and experimental judges disabled. Preserve assigned job roles as part of the existing baseline; a later no-role control must remove the associated role goals and orchestration hints too. Do not call assigned specialization emergent.

### Environment controls

Change one factor at a time before attempting a full factorial:

1. Communication: ordinary encounters versus a carefully defined restricted channel. Audit spoken messages, shared notes, broadcasts, remembered dialogue, and action side effects. A talk-only switch is not complete social isolation.
2. Memory: the same encounter schedule with different retrieval budgets or history windows; preserve private observations and log which information was available at each decision.
3. Roles: assigned roles versus generic roles with matched task information and permissions.
4. Team size and inference budget: report both fixed per-agent budget and fixed total-team budget comparisons.

Keep initial scenario seeds, resources, objectives, model settings, and scoring versions fixed across paired arms. Separate environment randomness from generation randomness. Do not claim bitwise repeatability across hardware or provider versions. Random memory IDs and generation seeds derived from prompt hashes are existing reproducibility concerns to test.

### Outcomes and attribution

Report objective completion, unique verified evidence collected, redundant work, failed actions, fallback actions, total tokens, and wall time separately. Use run or episode as the statistical unit, with paired differences and uncertainty across independent seeds; agent messages from the same run are not independent samples.

Collaboration requires evidence of benefit from interaction. Compare communication conditions and focal agents with fixed partner pools. Include solo controls with explicit information and budget matching. If the existing task is solvable equally well by independent agents, report that limitation; add a named task variant with complementary private information or resource dependencies rather than claiming a communication effect from an insensitive task.

Planner fallback may remain for legacy reproduction, but successes following fallback must be separated from model-selected successes. An episode with API failure, truncation, or missing logs is incomplete, not silently omitted or marked successful. Report all scheduled episodes and the incomplete fraction.

### First engineering milestone

- A local server health check and real structured-action completion.
- A short run through the existing town and action router.
- A manifest containing source identity, resolved config, model revision, quantization, runtime versions, and seeds.
- Exact prompts, raw responses, action decisions, parse errors, and resulting state changes.
- A small report that separates model decisions, planner fallbacks, action success, and task completion.
- Tests for malformed output, timeout handling, log persistence, and prevention of unsupported steering.

Only after that milestone: implement and test the channel/memory/role controls, run a replicated pilot, and decide whether the task has enough sensitivity for V2. An inference cache is not a world-state checkpoint; full resumption needs world, agent memories, objectives, scheduler RNG, and logging state restored consistently.

## V2: frontier APIs

Reuse the same environment and action validation with provider adapters. Capture returned model identity, actual usage, refusals, truncations, retries, and latency. Where a provider does not expose exact token IDs, sampling seeds, or immutable revisions, record that limitation rather than inventing replay metadata.

Compare models under identical task access and report cost/latency tradeoffs. A frozen partner pool tests a focal agent's ability to cooperate; homogeneous teams test the whole deployed system. Keep these separate. Held-out scenario families and partner pools must not influence prompt or parameter selection.

## V3: personality and training

First compare no explicit persona, homogeneous prompt personas, and diverse prompt personas. No explicit persona does not mean the model has no behavioral tendencies. Separate job roles from personality. Validate that the intervention changes the intended measured behavior before interpreting null or positive collaboration effects.

The published ten-dimensional inventory is Big Five plus curiosity, assertiveness, warmth, formality, and creativity. Current Lindr code instead extends Big Five with assertiveness, ambition, resilience, integrity, and curiosity. Preserve named versions; matching dimension names alone does not recover the original scorer. Obtain exact prompts, anchors, scorer implementation, and run provenance before claiming replication.

Activation steering requires hidden-state access. Tune dose and layers on a development set, measure task capability alongside personality effects, and lock settings before held-out evaluation. A coefficient is not an absolute dose across different vector norms or models. Include zero-dose and appropriate placebo controls.

[Tinker](https://tinker-docs.thinkingmachines.ai/) supports LoRA training and sampling for its supported open-weight models. It does not expose arbitrary proprietary-model weights. Trained LoRA interventions and inference-time activation steering are different treatments; neither is automatically a validated personality intervention.

## Human review and candidate judges

Human reviewers assess bounded episodes with the focal action, relevant preceding interaction, objective, available evidence, and observed consequence. Use one specific question per item and allow insufficient context or no opportunity. For drift, compare matched earlier and later episodes. Redact model and treatment metadata where feasible, while recognizing that text can reveal a condition.

Use a representative sample to estimate ordinary performance and report deliberately difficult cases separately. Double-rate a subset, inspect disagreement, and keep calibration and held-out sets separate at the episode level. Reviewers validate local behavioral judgments; they do not certify a whole society's personality.

Jev is an optional future judge candidate, not integrated or validated by this roadmap. Compare any judge with human labels and a second judge on held-out episodes. Preserve probability distributions, check calibration and false positives, and keep judge scores outside the primary reward and agent feedback loop initially.

## RL interface after evaluation works

Wrap the same town in an explicit reset/observe/step interface, with turn order, action validation, per-agent rewards, termination, truncation, and metadata. [PettingZoo's AEC API](https://pettingzoo.farama.org/api/aec/) is the reference for sequential multi-agent interaction. Parallel decision collection must have explicit state-update semantics; it is not interchangeable with sequential encounters.

Reward verified objective progress and record resource/token costs. Avoid rewarding message volume, friendliness, or agreement as collaboration. Prevent repeated submissions from receiving duplicate reward. Keep personality diagnostic at first. Split training and held-out evaluation by scenario and partner pool; evaluate cross-play against frozen partners to detect brittle conventions. Proprietary API models can be evaluated as fixed policies without being trainable through the same API.

## Prior work and its specific role

| Source | What it contributes | Boundary |
| --- | --- | --- |
| [Measuring LLM Personality](https://dyllonj.com/writing/llm-personality-benchmark.html) | Original dimension inventory and model-comparison motivation | Public article is not the complete reproducibility package. |
| [Cross-Vendor Personality Comparison](https://dyllonj.com/writing/cross-vendor-personality.html) | Controlled context variation with repeated prompts | Reported findings are not revalidated here. |
| [The Convergent Mind](https://dyllonj.com/writing/convergent-mind.html) | Hypothesis that interaction changes behavioral diversity | A hypothesis, not proof of inevitable convergence. |
| [CoSER preliminary study](https://dyllonj.com/writing/coser-preliminary-results.html) | Interactive intervention design and motivation to count parsing failures | Different environment; not evidence about this town's collaboration. |
| [Generative Agents](https://arxiv.org/abs/2304.03442) | Memory and interaction architecture | This repository's heuristic reflection is not an exact reproduction. |
| [AgentSociety](https://arxiv.org/abs/2502.08691) | LLM society simulation and behavioral experiments | Scale alone does not establish behavioral fidelity. |
| [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) | Open-weight activation intervention and capability checks | The paper reports degraded text at larger multipliers; it does not diagnose this repository. |
| [Persona Vectors](https://arxiv.org/abs/2507.21509) | Monitoring and controlling character traits | Trait control does not by itself establish better collaboration. |
| [PettingZoo AEC](https://pettingzoo.farama.org/api/aec/) | Standard multi-agent environment interface | Planned wrapper, not an already completed integration. |

The staged rollout, failure accounting, manifests, and matched controls are engineering and experimental-design decisions for this project. They are not claimed to be direct reproductions of every cited method.
