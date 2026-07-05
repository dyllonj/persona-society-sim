# System design

This is the high-level "why" document for the simulator's architecture. For
"how do I do X" and "what exactly does function Y accept," see the how-to and
reference docs linked throughout — this file stays focused on the reasoning
behind the major structural choices.

## Overview

Persona Society Sim hosts 30-300 LLM agents in a text-only town. Each agent
combines persona steering (CAA vectors or, for API models, prompt-based
descriptions), a perceive-reflect-plan-act memory loop, and goals derived
from trait coefficients and evolving objectives. Agents live inside a
lightweight world of locations, resources, and institutions; a scheduler
samples encounters each tick; every generated token bundle, applied steering
vector, and action is logged for downstream analysis.

## Why persona steering via CAA, not prompting

See [explanation-steering.md](explanation-steering.md) for the full
rationale. In short: activation-space steering gives a quantitative,
composable dose knob that doesn't compete with the rest of the prompt for
attention, which matters when running hundreds of agents that also carry
memory, dialogue history, and plan suggestions in their context. Local HF
models get the mechanistic version; the Gemini backend falls back to
prompt-based trait descriptions since API models expose no internals to hook.

## Why the memory/planning loop is heuristic, not another LLM call

`agents/planner.py::Planner.plan()` is a deterministic rule engine — no LLM
round-trip. It's presented to the *acting* LLM as an overridable default
("this is a heuristic suggestion, reject it if it conflicts with your
personality") rather than the final word. This keeps the planning step free,
instant, and reproducible for benchmarking objective/role compliance, while
still letting persona-steered generation have final say over what the agent
actually says or does. Similarly, "reflection" here means "retrieve and
concatenate relevant memories," not "generate a new abstract insight" the way
the original Generative Agents paper's LLM-driven reflection worked — a
deliberate simplification for a system with 3 rooms and modest agent counts,
where keyword+recency+importance scoring is cheap, fully inspectable, and
doesn't require an embedding model dependency. See
[reference-modules.md](reference-modules.md#agent-loop-agents) for the exact
retrieval-scoring formula.

## Why the orchestrator has a queue-backed "seam"

`orchestrator/queued_runtime.py` exists specifically as an incremental
migration path: bolt bounded-queue/worker-thread semantics onto the
*existing* synchronous `SimulationRunner` call sites (same method signatures)
rather than rewriting the runner into a fully async pipeline up front. This
lets larger runs (`configs/run.medium.yaml`: 100 agents, 500 ticks, Postgres)
avoid blocking the tick loop on slow DB writes or WebSocket sends, without a
risky big-bang rewrite. The two queues make different tradeoffs on purpose:
log records always block on a full queue (telemetry — the scientific record
of a run — must never be silently dropped), while viewer/TUI broadcasts drop
on a full queue (a missed UI frame is invisible and harmless). See
[reference-cli.md](reference-cli.md#queue-backed-runtime-scale-seam).

## Why safety is a lightweight backoff, not a hard gate

`safety/governor.py::SafetyGovernor` is a substring-match banned-phrase
filter, not a learned classifier — `toxicity_threshold` in the run config is
declared but not read anywhere, explicitly a placeholder for a future
classifier. On a match, it nudges every active persona coefficient toward
zero by a configurable fraction (`governor_backoff`) rather than hard-stopping
generation, and logs the event. This is a soft, self-correcting mechanism:
a persona that keeps tripping the filter gradually loses steering strength
rather than the run halting. The backoff fraction is tuned gentler at scale
(`0.15` in the 100-agent/500-tick medium config vs `0.2` elsewhere) because a
fixed multiplicative backoff compounds across more agents and ticks — too
aggressive a value would crush persona expression over a long run.

## Why structured tables for research telemetry, not opaque JSON

`ResearchFactLog`, `CitationLog`, and `ReportGradeLog` exist as first-class
Pydantic models/SQL tables rather than free-form JSON stuffed into
`ActionLog.info`. This lets `metrics/research.py` compute fact-coverage,
citation-diversity, and grading drift *per persona-trait cohort* using typed
field access — directly serving the "does persona steering change behavior on
structured tasks" research question without an ETL step. The same reasoning
extends to `graph_snapshot`/`metrics_snapshot` being written per-tick
*per-cohort* rather than one global row: cohort slicing happens once, at
write time, instead of being reconstructed downstream. See
[reference-data-schema.md](reference-data-schema.md).

## Why SQLite/Postgres *and* Parquet

SQL gives transactional, queryable, live-during-run storage; Parquet gives a
columnar, dependency-light format for downstream analysis (`scripts/analyze_simulation.py`
and `scripts/view_results.py` read Parquet exclusively, never the DB, because
that keeps analysis notebooks from needing a live DB connection). Both are
optional and fail closed — `--mock-model` dry runs work with neither
configured.

## Core components at a glance

1. **Steering** — `steering/compute_caa.py` extracts vectors from
   `data/prompts/*.jsonl`; `configs/steering.layers.yaml` is the metadata
   source of truth for layers/vector-store IDs; `steering/hooks.py` applies
   them at generation time with prompt-token masking. See
   [explanation-steering.md](explanation-steering.md).
2. **Agents** — `agents/agent.py` runs perceive → reflect/plan → act;
   `agents/memory.py`/`agents/retrieval.py` implement keyword+recency+importance
   memory scoring; `agents/planner.py` is the heuristic decision tree. See
   [reference-modules.md](reference-modules.md#agent-loop-agents).
3. **World + scheduler** — `env/world.py`, `env/actions.py`, `env/economy.py`,
   `env/institutions.py` model the town; `orchestrator/scheduler.py` samples
   encounters; `orchestrator/runner.py` executes ticks. See
   [reference-modules.md](reference-modules.md#world-model-env).
4. **Telemetry + storage** — Pydantic schemas in `schemas/` mirror SQL tables
   in `storage/db.py`; `storage/log_sink.py` dual-writes to SQL + Parquet.
   See [reference-data-schema.md](reference-data-schema.md).
5. **Safety + metrics** — `safety/governor.py` is the backoff filter above;
   `metrics/graphs.py` and `metrics/social_dynamics.py` compute per-tick
   social/macro metrics.
6. **Runner CLI** — `python3 -m orchestrator.cli <config>`. See
   [reference-cli.md](reference-cli.md).

## Data flow

```
trait prompts + configs/steering.layers.yaml
  -> steering/compute_caa.py
  -> data/vectors/*.npy + .meta.json
  -> metadata-aware loader (orchestrator/cli.py)
  -> steering hooks (steering/hooks.py)
  -> agent generation

agent observation -> memory store -> scheduler -> env -> logs -> metrics
```

## 3D viewer protocol

`viewer/ws_bridge.py` starts `ws://127.0.0.1:8765/ws` and emits JSON events:
`init` (world + agent roster), `tick` (positions + stats), `action`, `chat`,
`processing` (agent-is-thinking), `meta_broadcast`. `viewer/static/index.html`
(Three.js) renders a radial layout with room anchors and agent spheres colored
by persona. This is explicitly a prototype meant to be replaced by a real
engine (Unity/Godot/Unreal) implementing the same protocol later — see
[reference-modules.md](reference-modules.md#viewer-viewer) for current
limitations (CDN dependency, no auth, unbounded log growth, silent port-bind
failures).

## Extensibility hooks

- Add new traits by appending prompt JSONL + a `configs/steering.layers.yaml`
  entry — but note `steering/eval.py`'s `TRAIT_ALIASES` currently only covers
  E/A/C, so evaluating a new O/N trait needs a small code change too (see
  [explanation-known-gaps.md](explanation-known-gaps.md)).
- Validate prompt files with `python -m data.prompts.schema validate data/prompts/*.jsonl`;
  convert legacy `situation`/`positive`/`negative` files with the `convert` sub-command.
- Plug alternative environments by extending `env/world.py::World.configure_environment`.
- Swap storage backends by extending the `storage/db.py` interface.

## Related

- [explanation-steering.md](explanation-steering.md) — CAA design rationale in depth.
- [explanation-known-gaps.md](explanation-known-gaps.md) — where the implementation currently diverges from this design.
- [docs/eval.md](eval.md) — evaluation protocols for the research questions this architecture is meant to support.
