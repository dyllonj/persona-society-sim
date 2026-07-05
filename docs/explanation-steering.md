# Why Contrastive Activation Addition (CAA) instead of prompting

## The problem

Telling an LLM "act extraverted" in a system prompt is unreliable: it
competes with the rest of the prompt for attention, degrades as context grows
(memory, dialogue history, plan suggestions), and gives no quantitative dial
for *how much* extraversion to apply. For a simulation running 30-300 agents
concurrently, each generating continuously, that unreliability compounds —
you can't easily run a dose-response study ("what happens between mild and
strong Agreeableness?") on top of natural-language instructions, because
"mild" and "strong" aren't well-defined in prompt space.

## The approach

CAA steers by directly perturbing the model's internal activations instead of
its instructions. For each trait, `data/prompts/*.jsonl` holds forced-choice
items: a trait-agnostic behavioral stem plus two contrasting first-person
responses, with exactly one flagged as the "high" expression of the trait
(`data/prompts/schema.py::PromptItem`). `steering/compute_caa.py` runs the
base model over every item's high and low option, records the hidden state at
the answer token for a chosen decoder layer, and averages `high − low` across
all items to get a single direction vector per layer — the direction in
activation space that separates "behaves high on this trait" from "behaves
low on this trait." At runtime, `steering/hooks.py::SteeringController`
registers PyTorch forward hooks on those layers and adds `alpha * vector`
into the residual stream while the model generates, where `alpha` is the
agent's persona coefficient for that trait.

This is a deliberate two-tier strategy, not a single design applied
everywhere: local Hugging Face models get the full activation-space
treatment; Gemini (a black-box API with no access to internals) falls back to
`steering/prompt_steering.py`, which maps alpha magnitude/sign to canned
natural-language trait descriptions prepended to the prompt. The mechanistic
version is preferred wherever it's available; the prompt-based version is a
documented, deliberate degradation for API-only models, not an oversight —
though see [explanation-known-gaps.md](explanation-known-gaps.md#gemini-persona-steering-silently-no-ops)
for a real bug in how that fallback is currently wired up.

## Why vectors are normalized (repeatedly)

Every trait vector gets normalized to unit length at multiple points:
once when computed (`compute_trait_vectors`), again defensively when loaded
into a `SteeringController`, and again in the legacy metadata-file loading
path in `orchestrator/cli.py`. This isn't redundancy for its own sake — raw
activation-difference magnitudes vary hugely by layer (in the checked-in
`E.meta.json`, layer 6's norm is ≈2.4 versus layer 30's ≈38.7), and without
normalization the same `alpha` value would produce wildly different effective
steering strength depending on which layer you picked. Normalizing means
`alpha` is the *only* dose knob, comparable across traits, layers, and (in
principle) models. The pre-normalization norm is kept purely as diagnostic
metadata, not as an active parameter.

## Why layer choice is empirical, per trait

`configs/steering.layers.yaml` assigns 2-3 specific decoder layers per trait
rather than one global choice, on the theory that different behavioral
signals separate best at different depths (early/mid layers for affective
tone, later layers for procedural/planning behavior, per the file's own
per-trait `description` fields). `steering/layer_sweep.py` makes this
empirical rather than purely heuristic: it evaluates a saved vector's
dot-product sign-agreement against a *held-out* prompt file, per candidate
layer, and writes the winning layers back into the vector's own
`.meta.json` as `preferred_layers`. Keeping the winning layers inside the
metadata (rather than a separate config) means a re-run of `layer_sweep.py`
automatically improves future runtime loads without touching any other file.

## Why prompt tokens are masked out of steering

`SteeringController` only adds the residual delta to *generated continuation*
tokens, not the tokens that make up the prompt itself (system instructions,
memory, plan suggestions). The rationale: injecting a steering vector into
the model's processing of its own instructions risks corrupting comprehension
of what it's being asked to do. Steering is meant to bias how the model
*continues*, not how it *reads*.

## What this design does not do

An earlier draft of this document claimed persona vectors were
"orthogonalized per trait to reduce entanglement." That's not implemented —
each trait's vector is computed and normalized independently, with no
cross-trait projection-removal step. If two traits' vectors happen to be
correlated in activation space, steering one will have some effect on the
other; nothing in this codebase currently corrects for that. See
[explanation-known-gaps.md](explanation-known-gaps.md) for this and other
gaps between what the docs used to claim and what the code does.

## Related

- [reference-modules.md](reference-modules.md#steering) — CAA runtime surface.
- [reference-config.md](reference-config.md#steering-vector-metadata-configssteeringlayersyaml) — `steering.layers.yaml` schema, including a live model/layer mismatch to check before extracting vectors.
- [howto-compute-steering-vectors.md](howto-compute-steering-vectors.md) — running the extraction/eval pipeline.
- [explanation-known-gaps.md](explanation-known-gaps.md) — the Gemini steering bug, the held-out-eval-file gap, and other steering-adjacent issues.
