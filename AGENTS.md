# Agents

This document expands on the agent lifecycle, instrumentation hooks, and the new telemetry paths introduced in the latest SIM update. It supplements the brief overview in `README.md` with details that are helpful when debugging persona behavior or when extending the probe/evaluation stack.

## Cognitive loop
Agents still follow the perceive → reflect → plan → act loop described in the original paper implementation, but we now persist every intermediate artifact:

1. **Perceive** — `Agent.perceive` stores observations inside `MemoryStore` with a length-derived importance score (`agents/memory.py`). Recency isn't stored at this point; it's recomputed fresh from `current_tick - event.tick` whenever memories are scored during retrieval.
2. **Reflect** — `reflect_and_plan` queries `MemoryRetriever`, produces a natural-language summary plus referenced memory IDs, and caches that bundle on the agent object for later logging (`agents/agent.py`).
3. **Plan** — `Planner.plan` returns a deterministic, advisory action suggestion with params and utterance guidance (`agents/planner.py`). It is context for inference, not the action that is automatically executed.
4. **Act** — `_build_prompt` asks the model for a structured JSON action, the language backend injects persona steering vectors, and the agent validates the selected action and params before execution. Invalid or unparsable model output falls back to the planner suggestion and records the fallback reason.

The cached summary, referenced events, plan suggestion, prompt text, token counts, `prompt_hash`, raw completion, decision source, and parse error all travel alongside every `ActionDecision`. `SimulationRunner` writes the cognitive fields into `ActionLog`; when interpretability capture is enabled it also writes replay-safe `InferenceEvent` rows with exact token IDs, immutable model/tokenizer revisions, sampling state, effective alphas, and vector hashes.

## Persona steering & safety
Agents derive trait alphas from their `PersonaCoeffs`, clamp them via `SafetyGovernor`, and hand them to `SteeringController` so the residual additions happen in the specified decoder layers (`agents/language_backend.py`, `steering/hooks.py`). The global `steering.strength` scalar in each run config multiplies every trait coefficient before the controller applies prompt-aware masking, keeping instructions/system messages untouched while the generated continuation receives the residuals. Layer choices, vector-store IDs, and training prompt files now live in `configs/steering.layers.yaml`; both the extraction script and runtime loader read the same metadata so there is no hidden `[12, 16, 20]` default.

Use `scripts/compute_vectors.sh` to regenerate vectors from the A/B prompt files and `scripts/eval_vectors.sh` to run the `steering.eval` harness. The evaluation report records accuracy deltas, log-prob gaps, and transcript samples with the same `steering.strength` dose you plan to use in-sim, making it easier to gate launches on fresh metrics.

MetricTracker registers each agent’s persona profile at startup and consumes the per-message steering snapshots to compute efficiency/collaboration aggregates by trait band and |alpha| bucket. This is surfaced in the tracker JSONL output and can be merged with the cognitive-trace logs to diagnose drift.

## Graph & macro metrics
During every tick the runner promotes social interactions into edges:

- `talk`/`work`/`research`/`gift` actions generate weighted `Edge` records tagged with the interaction kind. (`trade` exists as a function in `env/actions.py` but is disabled — it always returns `success=False` and isn't registered in the action router — so it never produces edges in practice.)
- Persona bands and active steering buckets are stored with each edge to support homophily/polarization studies.

`metrics/graphs.py` turns those edges into `GraphSnapshot` entries (persisted via `LogSink` to SQL/Parquet), and `metrics/social_dynamics.py` composes tick-level macro measurements (cooperation rates, wealth Gini, polarization, enforcement cost) that land in the new `metrics_snapshot` table. These feeds provide the quantitative backbone for RQ2/RQ4 without having to reconstruct graphs from scratch.

## Probe scheduling
A dedicated ProbeManager now orchestrates evaluation interventions:

- **Self-report probes** inject structured observations for randomly selected agents on a configurable cadence. Prompts include Likert questions mapped to the same IPIP traits used for steering. Responses are parsed into numeric scores and stored in ProbeLog entries along with the raw question bundle.
- **Behavioral probes** temporarily override world context (e.g., spawn scripted NPC actions, alter objectives, or enforce civic events) to elicit targeted behaviors. Rubric scorers attach to the resulting `ActionLog`s and emit probe outcomes so we can benchmark compliance against governance policies.

Both probe types log their metadata and derived scores to SQL/Parquet, which makes it easy to slice adherence metrics by persona trait or steering intensity.

## Research telemetry
For the Research Sprint environment the simulator now emits structured research logs:

- `ResearchFactLog` captures every fact discovery with agent ID, doc/fact IDs, correctness flags, and persona metadata.
- `CitationLog` records each cite action so citation diversity can be measured by trait/alpha buckets.
- `ReportGradeLog` stores the parsed grading output from `submit_report`, including fact coverage, valid citations, reward points, and prompt hashes/cognitive traces for context.

These logs replace the opaque JSON blobs previously embedded inside `ActionLog.info` and enable personas-by-performance analyses without additional ETL.

## Where to look next
- `agents/agent.py` for the full decision pipeline and cognitive trace helpers.
- `orchestrator/runner.py` for how decisions turn into env actions, logs, graph edges, and metric snapshots.
- `metrics/tracker.py`, `metrics/graphs.py`, and `metrics/social_dynamics.py` for persona-aware aggregation logic.
- `storage/log_sink.py` for the buffering/writing of the new cognitive, graph, metrics, probe, and research logs.

## Full documentation

This file is a supplement focused on the agent/telemetry angle. For complete
reference and how-to material, see [docs/README.md](docs/README.md) — in
particular [docs/reference-modules.md](docs/reference-modules.md) (full
agent/world/viewer public surface) and
[docs/explanation-known-gaps.md](docs/explanation-known-gaps.md) (known gaps
worth knowing about before debugging unexpected behavior).
