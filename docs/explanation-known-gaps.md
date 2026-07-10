# Known gaps, live bugs, and dead configuration

This document exists because writing accurate reference docs for this
codebase surfaced a number of things that don't work the way the code
"reads" at a glance, or the way older docs described. Rather than scatter
one-line caveats across a dozen files, they're consolidated here — reference
docs link back to the relevant section instead of repeating the explanation.

Issues marked **Status: RESOLVED** have been fixed in code; the section text
is retained for historical context and to explain what was changed. Issues
marked **Status: PARTIALLY RESOLVED** have had partial fixes applied but
residual concerns remain. Issues marked **Status: N/A** reference files not
present in the current checkout.

## Probe preamble is injected twice

**Status: RESOLVED.** `orchestrator/runner.py` previously called
`probe_assignment.inject(base_context)` twice in a row in both the batched
and non-batched decision paths, duplicating the entire probe preamble
(question/scenario/instructions text) in the observation sent to the agent.
The duplicate `.inject()` call has been removed from both code paths; each
path now calls `.inject()` exactly once.

## `steering.eval` previously evaluated on training data

**Status: RESOLVED for E/A/C.** Each supported trait now has 20 authored
`*_eval.jsonl` items that are disjoint from its eight extraction items. The
split verifier checks IDs, exact content fingerprints, normalized text, and
near-duplicate question stems. `scripts/eval_vectors.sh` fails instead of
silently accepting train/eval overlap.

The evaluator also resets the continuation mask independently for both answer
options, decodes only newly generated tokens in transcript mode, and records
immutable model/tokenizer revisions. These sets are construct-grounded held-out
scenarios, not a separately normed psychometric instrument; they establish
data separation, not external validity. Openness and Neuroticism still lack
both vectors and held-out evaluation sets.

## Steering config describes a model that doesn't match the checked-in vectors

**Status: RESOLVED.** Extraction and evaluation now use
`configs/steering.layers.yaml::defaults.model` unless explicitly overridden.
The checked-in E/A/C vector metadata identifies
`Qwen/Qwen2.5-32B-Instruct`, matches the configured Qwen layers, and uses
portable artifact paths. Runtime loading verifies artifact IDs and hashes;
`HFBackend` additionally rejects model-name, layer-range, or hidden-width
mismatches before generation. An override remains possible for deliberate
cross-model extraction, but mismatched vectors can no longer enter a run
silently.

## `steering.eval`'s trait coverage is narrower than the persona model's

`steering.eval` accepts all five trait aliases, but its CLI and
`scripts/eval_vectors.sh` deliberately default to Extraversion,
Agreeableness, and Conscientiousness. Openness and Neuroticism are represented
by `PersonaCoeffs` and run-config `steering.coefficients`, but there are no
checked-in O/N extraction prompts, held-out sets, or CAA artifacts. Those data
and vectors—not another alias—are required before either trait can become an
activation-steered experimental arm.

## `docs/design.md`'s orthogonalization claim doesn't match the code

**Status: PARTIALLY RESOLVED.** An earlier version of the design doc stated
persona vectors are "orthogonalized per trait to reduce entanglement." The
original gap was that no orthogonalization step existed anywhere in the
steering pipeline. The design doc has since been corrected, and
`steering/compute_caa.py` now contains an `enforce_orthogonality()` function
that checks cross-trait cosine similarity (threshold 0.2) and **raises** if
candidate vectors are too aligned with existing traits' vectors. This is a
rejection-based orthogonality gate, not a projection-removal transform (it
doesn't modify vectors to make them orthogonal; it refuses to save vectors
that are too correlated). The explanation doc has been updated to reflect
this in [explanation-steering.md](explanation-steering.md). Note that prior
work (Bhandari et al. 2026) shows hard orthonormalization does NOT eliminate
behavioral cross-trait bleed even when geometric orthogonality is enforced,
so this gate is a necessary but not sufficient safeguard against trait
entanglement.

## Objective progress tracking uses strict string equality

**Status: RESOLVED.** `ObjectiveManager.process_action_log` previously only
counted `fill_field`/`scan` progress when `action_log.info` carried the exact
**string** `"1"` for the `unique`/`token_acquired` keys. A type-tolerant
`_is_flag_set()` helper has been added that accepts string `"1"`, int `1`,
and boolean `True` as truthy success markers, so objective progress now
increments correctly regardless of which type the action layer emits.

## `log_sink.close()` is never called on a successful run

**Status: RESOLVED.** `orchestrator/cli.py`'s `main()` previously called
`log_sink.close()` only on the exception path; the `finally` block ran only
`event_bridge.stop()`. `log_sink.close()` has been moved into the `finally`
block so it now runs on both success and failure paths, and the redundant
cleanup calls have been removed from the `except` block. Final buffer flush
errors are now surfaced, and `QueueRuntimeStats` (enqueued/dropped/error
counts) is inspected on every run exit path.

## Autoresearch: several "guardrails" are prose-only, not enforced

**Status: N/A — files absent.** The `experiments/autoresearch/` directory
does not contain the referenced files (`matrix.yaml`,
`autonomy_policy.yaml`, `run_matrix.py`, `anti_cheat.py`, `score.py`,
`program.md`, `README.md`, `RUNBOOK.md`) in the current checkout, and they
have no git history. The original findings are preserved below for reference
in case the autoresearch subsystem is re-introduced:

- `program.md` tells the operator not to modify `storage/log_sink.py`, probe
  scoring, report grading, or objective rules — but none of those files are
  in `matrix.yaml:boundaries.frozen_files`, so `anti_cheat.py`'s boundary
  manifest does not actually detect edits to them. This is the most
  safety-relevant gap of the group: a trial that edits one of those files
  mid-run will not be flagged invalid by anything automated.
- `autonomy_policy.yaml` declares `required_candidate_validity`,
  `required_human_decision_state`, and `max_capability_regression` as if they
  were enforced guardrails; only `allowed_auto_accept_proposals` and
  `forbidden_auto_accept_proposals` are actually read by `run_matrix.py`.
- `run_matrix.py --resume` has no `--no-resume` counterpart and defaults to
  `True` unconditionally — passing or omitting the flag has identical effect.
  `--force` is the actual way to force a clean re-run.

## Viewer: port-bind failures are invisible

Both the WebSocket bridge and the static HTTP server bind their sockets
inside background threads. If the port is already in use, the `OSError`
happens inside that thread, is not caught by the CLI's `try/except` around
viewer startup (which only guards thread *creation*, not the bind itself),
and the run proceeds silently as if the viewer had started — the CLI still
prints "Viewer: Open http://127.0.0.1:19123" even if the bind failed.
Similarly, `--viewer` import/init failures are only logged when `--live` is
also passed; run `--viewer` alone and a missing `websockets` package fails
with no visible message at all.

## Dead / aspirational configuration fields

Collected in one place since they're easy to trip over when writing a new
run config:

| Field | File | Status |
|---|---|---|
| `steering.vector_norm` | `configs/run.*.yaml` | Never read; actual norms come from vector metadata |
| `safety.toxicity_threshold` | `configs/run.*.yaml` | Never read; governor is pure substring matching |
| `optimization.batch_size` | `configs/run.fast.yaml` | Explicitly commented "(future feature)" |
| `steering.layers.yaml: defaults.prompt_masking/extraction/num_hidden_layers` | `configs/steering.layers.yaml` | Documentation-only; `defaults.model` is now consumed by extraction/evaluation |
| `personas.bigfive.yaml: ranges/percentiles/sampling.strategy/sampling.seed` | `configs/personas.bigfive.yaml` | Only `sampling.jitter` is actually read |
| `autonomy_policy.yaml: enabled_by_default/required_candidate_validity/required_human_decision_state/max_capability_regression` | `experiments/autoresearch/autonomy_policy.yaml` | N/A — file not present in current checkout |
| `matrix.yaml: validity.hard_discard` | `experiments/autoresearch/matrix.yaml` | N/A — file not present in current checkout |
| `matrix.yaml: workflow.*` | `experiments/autoresearch/matrix.yaml` | N/A — file not present in current checkout |

## Related

- [explanation-steering.md](explanation-steering.md)
- [reference-config.md](reference-config.md)
- [reference-modules.md](reference-modules.md)
- [reference-data-schema.md](reference-data-schema.md)
