# Jacobian Lens integration plan

## Status

Implemented locally through the fit/replay toolchain, runtime steering smoke
test, and full regression suite. The fixed-price on-demand GPU pilot remains
the final validation gate.

This document is the implementation contract for adding Anthropic's Jacobian
Lens to the simulator. The integration is intentionally split into two
processes:

1. The simulator records exact, reproducible inference events without loading
   Jacobian Lens.
2. A separately pinned interpretability environment replays selected events,
   applies a fitted lens, and writes analysis-ready Parquet traces.

The split is required because the simulator currently pins Transformers 4.45,
while `jlens` 0.1 requires Transformers 5.5 or newer. It also prevents
interpretability work from changing simulation timing or exhausting GPU memory
during a run.

## Terminology

A fitted Jacobian Lens contains one corpus-averaged matrix per source layer:

```text
J_l = E[d h_target / d h_l]
```

The matrix is a model-level artifact. It is not recomputed for each agent.
An agent's "Jacobian trace" is the layer-by-position vocabulary readout
obtained by applying the fitted matrices to the residual activations from one
recorded inference.

The system stores these as separate artifacts:

- `lens_manifest`: how a model-level lens was fitted and hashed.
- `inference_event`: the exact model input, output tokens, steering state, and
  sampling state for one agent decision.
- `jacobian_trace`: top-k lens readouts joined to an inference event.

## Architecture

```text
                     OFFLINE, ONCE PER MODEL REVISION

 exact model + generic corpus ------> fit J_l ------> lens.pt + manifest.json


                         ON EVERY CAPTURED DECISION

 simulator ------> inference event ------> versioned manifest Parquet
   |                    |
   |                    +-- prompt IDs and attention mask
   |                    +-- generated IDs and raw completion
   |                    +-- effective steering alphas and vector hashes
   |                    +-- decoding mode and deterministic seed
   |
   +------> normal action/message/metric logs


                              POST-HOC

 inference events + lens + exact model revision
                  |
                  +------> neutral replay
                  +------> steered replay
                  +------> layer x position x top-k trace Parquet
```

## Non-negotiable correctness gates

Paid GPU work must not begin until all of the following pass locally:

1. Enabled steering fails closed when a vector is absent or incompatible.
2. Vector paths are portable and all artifacts are content-hashed.
3. A startup smoke test observes a non-zero residual delta at every configured
   steering layer.
4. The recorded steering snapshot is the effective injected value after global
   and per-trait scaling.
5. Prompt and output token IDs can be replayed without decode/re-tokenize.
6. The generation mode (`do_sample`) and per-decision seed are explicit.
7. Concurrent calls cannot mutate one shared `SteeringController`.
8. A model-selected action is parsed and validated before execution; a planner
   suggestion is not silently treated as the model's decision.
9. CAA-only prompts contain no derived personality labels.
10. Cooperation and graph metrics pass controlled fixtures.

## Simulator configuration

The run configuration gains two explicit sections:

```yaml
inference:
  persona_prompt: false
  structured_actions: true

interpretability:
  enabled: true
  sample_rate: 0.05
  include_prompt_text: false
```

This records replay inputs only. The simulator does not load a lens or perform
a second forward pass. Probe, report-submission, and safety-event generations
are always captured; `sample_rate` applies to ordinary decisions.

## Inference event schema

Every captured generation must contain:

| Field | Meaning |
|---|---|
| `schema_version` | Versioned compatibility boundary |
| `trace_id` | Stable ID used by all post-hoc outputs |
| `run_id`, `tick`, `agent_id`, `action_id` | Simulation join keys |
| `cognitive_phase` | `decision`, `reflection`, `probe`, or another declared phase |
| `prompt_hash` | SHA-256 of the rendered model input |
| `prompt_text` | Optional human-readable rendered input |
| `input_ids` | Exact tokenizer IDs passed to the model |
| `attention_mask` | Exact prompt mask, including padding |
| `generated_ids` | Exact output IDs returned by generation |
| `raw_completion` | Unsanitized decoded completion |
| `effective_alphas` | Values actually used by the steering hooks |
| `steering_vector_ids` | Trait-to-artifact mapping |
| `steering_vector_hashes` | Content hashes for injected vectors |
| `model_id`, `model_revision` | Immutable checkpoint identity |
| `tokenizer_revision` | Immutable tokenizer identity |
| `dtype`, `quantization` | Numerical inference condition |
| `do_sample`, `temperature`, `top_p` | Decoding condition |
| `sampling_seed` | Per-decision deterministic seed, when sampling |
| `prompt_token_count`, `generated_token_count` | Integrity checks |

The event records raw and sanitized output separately. Sanitization must never
destroy the data needed to reconstruct an inference.

## Lens artifact manifest

Each `lens.pt` must be accompanied by a JSON manifest containing:

```text
schema_version
lens_id
lens_sha256
jlens_git_commit
model_id
model_revision
tokenizer_revision
model_config_sha256
torch_version
transformers_version
dtype
quantization
d_model
n_layers
source_layers
target_layer
corpus_name
corpus_sha256
n_prompts
max_seq_len
skip_first
dim_batch
created_at
```

The replay tool aborts if the lens dimension, model revision, tokenizer, or
requested layers disagree with the inference event.

## Trace schema

The primary trace dataset is long-form Parquet, partitioned by run and trace:

| Field | Meaning |
|---|---|
| `trace_id` | Join to the inference event |
| `condition` | `observed`, `neutral_replay`, or declared counterfactual |
| `position_phase` | `prompt` or `generated` |
| `source_position` | Residual position read by the lens |
| `predicted_position` | Token position predicted from the source position |
| `generated_offset` | Offset within the generated continuation, if applicable |
| `layer` | Source residual layer |
| `rank` | Rank within the exported top-k tokens |
| `lens_token_id`, `lens_token_text` | Readout vocabulary token |
| `lens_logit` | Pre-softmax lens score |
| `actual_next_token_id` | Token actually emitted from this source position |
| `actual_next_token_rank` | Its rank under the readout |
| `is_actual_next_token` | Convenience indicator |

Full-vocabulary logits are not stored by default. The current exporter writes
top-k output only. Support for a small preregistered token set is a required
follow-up before confirmatory concept testing; top-k inspection alone is
exploratory and selection-biased.

## Replay rules

1. Load the exact model and tokenizer revisions recorded in the event.
2. Consume `input_ids` directly. Never reconstruct them by tokenizing decoded
   text.
3. Concatenate prompt and recorded generated IDs.
4. Reapply the exact effective steering vector at generated-token positions
   only.
5. Register the activation recorder after steering hooks so it observes the
   post-intervention residual.
6. Run the model in evaluation mode with cache disabled.
7. Treat residual position `p` as predicting token position `p + 1`.
8. Export both an observed-condition replay and, when requested, a neutral
   replay over the same token path.
9. Validate final-layer replay logits against a direct model forward pass.

## Lens fitting protocol

### Pilot

- Exact open-weight model used by the simulation.
- Unquantized BF16 unless the simulation's primary research condition uses a
  different declared representation.
- 100 generic, pretraining-like sequences.
- 128 tokens per sequence.
- Start with the late intervention layers needed for the engineering trace;
  expand to all intervention layers only after the pilot memory profile is known.
- Checkpoint every five to ten prompts.

### Final artifact

- 1,000 generic sequences.
- Approximately 25 evenly spaced layers plus every intervention layer.
- Prompt-sharded fitting followed by `JacobianLens.merge()`.
- A generic corpus as the primary lens.
- An optional simulation-domain lens only as a sensitivity analysis.

The pilot validates engineering and basic readout fidelity. It must not be
reported as a full replication of the paper.

## Research design

The minimum causal design has the following arms:

| Arm | Persona prompt | CAA |
|---|---:|---:|
| Neutral | No | No |
| Prompt-only | Yes | No |
| CAA-only | No | Yes |
| Hybrid | Yes | Yes |
| Placebo-vector | No | Shuffled vector |

World seeds, scheduler seeds, model seeds, and probe schedules are paired
across arms. Simulation run is the independent experimental unit; individual
actions are nested observations.

Mechanistic endpoints include:

- J-lens readout of the steering vector itself: `unembed(J_l @ v_trait)`.
- Steered-minus-neutral concept-score deltas over identical token paths.
- Trait-congruent token ranks at preregistered workspace layers.
- J-space overlap and drift across ticks.
- Evaluation-awareness tokens during probes versus natural interactions.
- Whether one agent's concepts subsequently appear in another agent's trace.

Behavioral endpoints include validated action choice, reciprocal cooperation,
helping/gifting, conflict, information sharing, citation quality, completion
latency, and directed-network statistics.

J-lens output is an approximate token-indexed, first-order readout. It must not
be described as a verbatim chain of thought or as complete access to the
model's internal computation.

## Implementation audit and prioritized improvements

### Implemented now

- The simulation captures replay data without importing Jacobian Lens or
  performing an extra forward pass.
- Enabled steering fails closed on missing, wrong-model, out-of-range,
  wrong-width, or hash-mismatched artifacts.
- A real startup forward verifies a nonzero local residual change at every
  configured trait/layer hook.
- Shared-model generation is locked; sampled batches preserve independent
  per-decision seeds by running sequentially.
- CAA-only configs omit prompt-derived trait labels.
- Structured actions are schema-validated. Missing params and empty utterances
  produce an explicit recorded planner fallback rather than silently mixing
  model and planner decisions.
- Replays consume exact token IDs, validate final logits, and export top-k
  readouts rather than full vocabulary tensors.

### P0 before a confirmatory study

1. Add truly held-out trait evaluation prompts. The current steering harness
   falls back to its training items, so its quality estimates are optimistic.
2. Fit a lens that includes every active intervention layer. A three-layer
   engineering pilot can validate the pipeline but cannot characterize all
   seven E/A/C injection sites.
3. Match the numerical condition. The fast simulation config uses NF4 while
   the primary pilot lens is BF16; either run the research arm in BF16 or fit
   and report a quantization-matched sensitivity artifact.
4. Run multiple independently seeded simulations per arm. Actions are nested
   observations; treating thousands of actions from one world as independent
   would be pseudoreplication.
5. Add preregistered token/concept export. Top-k tokens discovered after seeing
   the traces are useful for hypothesis generation, not confirmatory evidence.
6. Report planner-fallback rate by arm and exclude or model fallback decisions
   in the primary action-choice analysis.

### P1 engineering optimizations

1. Batch post-hoc replays by model, sequence length, and condition. The current
   exporter processes events sequentially for correctness and simplicity.
2. Stream Parquet row groups instead of retaining all trace rows in memory.
3. Compute each trait vector's `unembed(J_l @ v_trait)` signature once per lens
   and cache it; it is model-level, not agent-level.
4. Shard lens fitting by prompt and merge with `JacobianLens.merge()` for the
   1,000-sequence artifact, retaining shard hashes and failure recovery.
5. Add trace-level summary tables so most analyses do not scan long-form top-k
   rows repeatedly.

### Interpretation limits

- A neutral replay over the observed token path estimates a local mechanistic
  contrast; it is not the behavior the neutral agent would have generated.
  Paired live neutral/CAA runs remain necessary for behavioral causal effects.
- Jacobian Lens is a corpus-averaged first-order approximation. Results should
  be sensitivity-checked across corpus choice, layer coverage, and lens seed.
- Cross-agent concept propagation is observational unless communication edges,
  timing, and randomized interventions are incorporated into the design.

## Performance strategy for the final artifact

- Capture manifests during the simulation; trace post-hoc.
- Sample ordinary events and always retain probes, safety events, and terminal
  task actions.
- Batch replays by model, sequence-length bucket, and steering condition.
- Use matrix multiplication across all selected positions at a layer.
- Store top-k scores, not full-vocabulary tensors.
- Compute each steering vector's standalone J-lens signature once.
- Buffer trace rows into large Parquet row groups instead of writing one small
  file per event.

## GPU budget guardrails

The initial external GPU budget is approximately USD 20. Automation must:

1. Query offers without renting anything.
2. Select a fixed-price on-demand instance with sufficient VRAM. Bid,
   interruptible, and preemptible offers are prohibited for this project.
3. Calculate the maximum runtime from the remaining budget before creation.
4. Refuse offers with unexpected storage, bandwidth, or on-demand charges.
5. Write resumable checkpoints frequently and recover them before teardown.
6. Stop on a fixed wall-clock timeout or cost threshold.
7. Download and hash artifacts before destroying the instance.
8. Destroy the instance in a `finally` path even when fitting fails.
9. Never persist the Vast.ai API credential in the repository, shell history,
   logs, process arguments, notebooks, or artifacts.

The first paid run is limited to a dependency/model-loading smoke test and a
small pilot. A 1,000-prompt lens is a separate decision after timing data from
the pilot exists.

## Acceptance tests

- Missing or incompatible steering vectors abort startup.
- Relative vector paths work after moving the repository.
- Nonzero configured steering produces a measured nonzero residual delta.
- Logged effective alphas match the tensor additions used by the hook.
- HF multi-worker mode cannot race one shared controller.
- Batched and sequential generation agree under deterministic decoding.
- Prompt masking uses the attention mask and never steers prompt tokens.
- CAA-only prompts contain no personality labels.
- Structured action output is schema-validated and affordance-checked.
- Cooperation fixtures produce correct rates above and below zero.
- Repeated graph interactions accumulate rather than overwrite.
- Inference events round-trip through Parquet without losing token IDs.
- Trace source positions align with actual next tokens.
- Final-layer replay logits match direct forward logits within tolerance.
- A lens/model/tokenizer mismatch fails before a GPU forward pass.

## Implementation order

1. Artifact loading and steering observability.
2. Reproducible inference events and concurrency protection.
3. Prompt-treatment separation and model-selected actions.
4. Macro and graph metric corrections.
5. Isolated lens fit/replay tools.
6. Local unit and integration tests.
7. Budgeted GPU smoke test and pilot.
8. Final research run only after pilot review.
