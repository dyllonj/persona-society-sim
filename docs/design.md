# System Design

## Overview
Persona Society Sim hosts 30–300 LLM agents in a text-only town. Each agent combines:
- **Persona steering** via contrastive activation addition vectors stored per trait + layer.
- **Memory loop** (observation → reflection → planning) with summarization heuristics.
- **Local goals** derived from trait coefficients, town events, and evolving plans.

Agents live inside a lightweight world containing locations, resources, institutions, and scheduled events. A scheduler samples encounters (co-location, directed messages, noticeboard, projects) each tick. Logging captures every generated token bundle, applied steering vector, and action for downstream analysis.

## Core components
1. **Steering pipeline**
   - `data/prompts/*.jsonl` store forced-choice IPIP-inspired pairs per trait using the shared stem + A/B option schema (see `data/prompts/schema.py`). Each prompt marks the high-trait option explicitly so extraction and evaluation can check directional fidelity.
   - `steering/compute_caa.py` loads prompts, runs base model forward passes, subtracts the low option from the high option per decoder layer, and normalizes the residual before persisting it.
   - `configs/steering.layers.yaml` defines the vector root, layer choices, vector-store IDs, and prompt files for every trait. Both `scripts/compute_vectors.sh` and the runtime loader rely on this metadata so the legacy `[12, 16, 20]` default is gone.
   - `steering/vector_store.py` persists vectors as `.npy` with metadata for reproducibility.
   - `steering/hooks.py` attaches PyTorch forward hooks to add \(\alpha_t v_{t,\ell}\) at runtime while masking prompt tokens so only generated continuations receive the residual injection.

2. **Agent architecture**
   - `agents/agent.py` implements the perceive → retrieve memories → reflect → plan → act pipeline.
   - `agents/memory.py` stores `MemoryEvent`, `Reflection`, and `Plan` entries with recency × importance scoring.
   - `agents/planner.py` converts plans into `env.actions` requests plus utterances.

3. **World + scheduler**
   - `env/world.py` tracks locations, noticeboards, economy state, and calendars.
   - `env/actions.py` validates and executes moves, conversations, trades, proposals, sanctions, etc.
   - `env/economy.py` models a simple civic exchange (quotes, clearing, wealth updates).
   - `orchestrator/scheduler.py` samples encounters each tick; `orchestrator/runner.py` executes ticks, logs outputs, and handles safety governors.

4. **Telemetry + storage**
   - Pydantic schemas in `schemas/` enforce structured logging. They map directly onto SQL tables defined in `storage/db.py`.
   - `storage/log_sink.py` fans out logs to SQL + Parquet, enabling incremental snapshotting per tick.

5. **Safety + metrics**
   - `metrics/graphs.py` constructs social graphs per tick and computes centrality/assortativity.
   - `metrics/social_dynamics.py` tracks cooperation, polarization, productivity, and well-being proxies.
   - `safety/governor.py` clamps coefficients and emits `SafetyEvent`s when heuristics fire.
   - `docs/eval.md` details evaluation protocols for the research questions.

6. **Runner CLI**
   - `python3 -m orchestrator.cli <config>` loads YAML configs, instantiates agents/world, and streams logs via `SimulationRunner`.
   - Optional: start a WebSocket bridge and web viewer with `--viewer` to visualize rooms and agent movement in 3D.
   - `--mock-model` flag runs without HF weights for fast development; defaults to HF backend once vectors exist.
   - `steering.strength` in the run config globally scales persona coefficients (set `< 1.0` for subtle steering, `> 1.0` for aggressive personas).

## Data flow
```
trait prompts + configs/steering.layers.yaml -> steering/compute_caa.py -> data/vectors/*.npy + .meta.json -> metadata-aware loader -> steering hooks -> agent generation
agent observation -> memory store -> scheduler -> env -> logs -> metrics

## 3D viewer integration (prototype → engine)

- Bridge: `viewer/ws_bridge.py` starts `ws://127.0.0.1:8765/ws` and accepts JSON events:
  - `{"type":"init", "world": {"locations": {...}}, "agents": [{agent_id, display_name, persona_coeffs, location_id}]}`
  - `{"type":"tick", "tick": int, "positions": {agent_id: room_id}, "stats": {collab_ratio, duration_ms}}`
  - `{"type":"action", "tick": int, "agent_id": str, "action_type": str, "params": {...}}`
  - `{"type":"chat", "tick": int, "from_agent": str, "room_id": str, "content": str}`
- Viewer: `viewer/static/index.html` (Three.js) renders a radial layout with room anchors and agent spheres.
- Engine path: replicate the same WebSocket protocol in Unity/Godot/Unreal to render a richer world (NavMesh, animations, UI). The simulation remains the source of truth.
```

## Safety + governance
- Coefficient clamps (default |α| ≤ 1.5) with automatic backoff when safety events fire.
- Persona vectors orthogonalized per trait to reduce entanglement.
- Content filters + red-team probes (hooks for custom governance modules to be added later).

## Extensibility hooks
- Add new traits by appending prompt JSONL + vector generation config.
- Use `python -m data.prompts.schema validate data/prompts/*.jsonl` when editing prompts; the CLI enforces the stem + option schema and can convert legacy positive/negative entries via the `convert` sub-command.
- Plug alternative environments by implementing `WorldLike` protocol in `env/world.py`.
- Swap storage backends by extending `storage/db.py` interface.
