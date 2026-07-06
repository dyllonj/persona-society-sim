# Module reference: agents, world, viewer

Reference for the public surface of the core Python packages. For the
steering/CAA pipeline specifically, see [explanation-steering.md](explanation-steering.md)
(design rationale) — this doc covers the runtime pieces that *consume*
steering vectors, plus the parts of the simulation that have nothing to do
with steering at all.

## Steering

- `steering/hooks.py::SteeringController(model, trait_vectors, vector_norms=None)` — registers PyTorch forward hooks on the decoder layers referenced by any trait vector. `.set_alphas(alphas, prompt_length=None)` / `.set_batched_alphas(...)` set per-trait coefficients for the next generate call; `.register()`/`.remove()` attach/detach the hooks; `.needed_layers` is the sorted union of layers across all loaded traits. Every vector is re-normalized to unit length on construction regardless of what was passed in.
- `steering/vector_store.py::VectorStore(root).load(vector_store_id, layers=None)` — loads a saved `.npy` + `.meta.json` bundle; raises if the metadata's recorded `vector_store_id` doesn't match what you asked for.
- `steering/prompt_steering.py::get_steering_prompt(alphas)` — the Gemini-backend fallback; maps full trait names (not short codes — see the integration bug below) with `|alpha| >= 0.1` to canned natural-language descriptions.
- `steering/compute_caa.py`, `steering/eval.py`, `steering/layer_sweep.py` — the offline extraction/evaluation/layer-selection scripts; see [howto-compute-steering-vectors.md](howto-compute-steering-vectors.md) for how to run them and [explanation-steering.md](explanation-steering.md) for why they're shaped this way.

## Agent loop (`agents/`)

Each `Agent` runs **perceive → reflect/plan → act** once per tick.

### `agents/agent.py`

- `Agent(run_id, state: AgentState, language_backend, memory: MemoryStore, retriever: MemoryRetriever, planner: Planner, safety_governor: SafetyGovernor, max_new_tokens=120, reflect_every_n_ticks=1, suppress_alphas=False)`
- `.persona_alphas() -> Dict[str, float]` — per-trait alpha = clamp(base persona coefficient + any active override); all zero if `suppress_alphas`.
- `.perceive(observation: str, tick: int) -> None` — stores the observation in `MemoryStore` with an importance score that is purely a function of text length (`min(1.0, 0.3 + 0.05*len(words))`) — there is no semantic-salience scoring.
- `.reflect_and_plan(tick, ...) -> PlanSuggestion` — retrieves relevant memories via `MemoryRetriever.summarize()` and asks `Planner.plan()` for a suggested action. Only re-runs on ticks where `tick % reflect_every_n_ticks == 0`; otherwise reuses the cached suggestion if the agent hasn't moved and no more than 1 tick has elapsed since the last plan.
- `.act(observation, tick, ...) -> ActionDecision` — the main entry point. Builds a prompt that presents the planner's suggestion as a *rejectable heuristic default* ("If this action conflicts with your personality, YOU MUST REJECT IT"), calls the language backend, sanitizes the output, runs it through `SafetyGovernor`, and returns an `ActionDecision` (action type, params, utterance, prompt text/hash, token counts, steering snapshot, safety event, cognitive-trace fields).
- If the LLM's sanitized output comes back empty, `act()` silently falls back to the planner's heuristic utterance rather than an empty string.

### `agents/memory.py`, `agents/retrieval.py`

- `MemoryStore.add_event/add_reflection/add_plan` — append-only; **no eviction, TTL, or pruning**. Memory grows unbounded over a long run; `relevant_events` does a full linear scan every call.
- `MemoryStore.relevant_events(query, current_tick, ...)` scores each memory as `keyword_overlap + focus_bonus + importance + recency + 0.5*trait_resonance + deterministic_jitter`. Recency decays linearly (`1 - 0.01*ticks_ago`, floored at 0.1 — never fully excluded). `trait_resonance` is a dot product between a memory's keyword-tagged Big-Five traits and the acting agent's own persona — this is a bias mechanism, not embedding-based semantic retrieval; there is no vector similarity search anywhere in this pipeline.
- "Reflection" in this codebase is **not an LLM call** — `MemoryRetriever.summarize()` just concatenates the top-scored memories' text, and the stored `Reflection` record is a templated string (`"Focus: {goals}"`), not a generated insight. This differs from the original Generative Agents paper's LLM-driven reflection step.

### `agents/planner.py`

`Planner.plan(...)` is a **deterministic rule engine, not an LLM call** — this keeps plan generation cheap, reproducible, and testable independent of LLM variance. Precedence order when multiple rules could fire:

1. `force_collaboration` hint override (from the meta-orchestrator's silence detection)
2. Active-objective-specific plan (special-cased for policy/navigation/research objectives)
3. Institutional-rule-derived plan
4. A once-per-reflection-cycle "quick sync" alignment plan
5. Observation-keyword-triggered plan (e.g. mentions of "library"/"cite"/"report")
6. Role-bias cyclical plan (research/policy/navigation roles cycle through role-appropriate actions)
7. Fallback: keyword-match against the agent's current goal

The plan is deliberately framed to the acting LLM as *overridable* — see `agents/agent.py` above.

### `agents/language_backend.py`, `agents/gemini_backend.py`

- `HFBackend` — local Hugging Face model, steers via `steering.hooks.SteeringController` (activation addition). `generate_batch()` groups requests by `max_new_tokens` and steers the whole batch in one forward pass for throughput.
- `GeminiBackend` — calls the Gemini API; steers by **prepending natural-language persona instructions** (`steering.prompt_steering.get_steering_prompt`) since activation injection isn't possible against a black-box API. Any API exception is caught and turned into the literal text `"[Error: ...]"` rather than propagating — downstream sanitize/safety code will process that string as if it were agent dialogue.
- `MockBackend` — deterministic stub for tests; echoes the alpha values into its output text.

**Known integration bug**: `Agent.persona_alphas()` returns short trait codes (`E`, `A`, `C`, `O`, `N`). `HFBackend`/`SteeringController` consume these correctly. But `GeminiBackend.generate()` passes these same short codes into `prompt_steering.get_steering_prompt()`, which looks keys up by *full capitalized trait names* (`"Extraversion"`, etc. — `trait.capitalize()` on `"E"` yields `"E"`, which is never in the lookup table). **Persona steering silently no-ops for every Gemini-backed run driven through the normal `Agent.act()` pipeline.** `scripts/verify_gemini_steering.py` doesn't catch this because it manually constructs full-name alpha dicts, bypassing `Agent.persona_alphas()` entirely. See [explanation-known-gaps.md](explanation-known-gaps.md#gemini-persona-steering-silently-no-ops).

## World model (`env/`)

### `env/world.py`

`World` holds a small fixed graph of `Location`s (`town_square`, `community_center`,
`library` by default), each with capacity, 2D coords, resources, and occupants.
Key methods: `move_agent` (no capacity enforcement — capacity is
descriptive/prompt-context only), `sample_context(agent_id)` (builds the
plain-text observation string fed to agents each tick — not structured JSON),
`configure_environment(env_name, difficulty)` (switches between the
`research`/`policy`/`nav` scenarios), `research_access`/`add_citation`/`grade_report`
(the research-scenario corpus/citation/grading loop), `record_checklist_field`/`policy_plan_ready`
(the policy-scenario checklist), `acquire_scan_token` (the nav-scenario token pool).

`World.institutional_guidance()` demotes commerce/civic-tagged rules to
`"advisory"` priority when `environment == "research"` — the same
`InstitutionManager` rule set is reused across scenarios, with the
environment only changing how rules are *interpreted*.

### `env/actions.py`

`ACTION_ROUTER` dispatches `move`, `talk`, `work`, `gift`, `scan`, `fill_field`,
`propose_plan`, `submit_plan`, `research`, `cite`, `submit_report`. **`trade` is
implemented but disabled** — `trade()` always returns `success=False,
{"error": "disabled"}` and isn't registered in `ACTION_ROUTER` at all, even
though `Economy.apply_trade` (below) is fully functional. `scripts/migrate_remove_trade_records.py`
exists specifically to strip old trade-era rows out of a dump directory —
see [howto-migrate-legacy-data.md](howto-migrate-legacy-data.md).

Other notable behaviors: `move` to your current location is a no-op that
still reports success; `research`/`cite` work from any location (not just the
library — being elsewhere only adds an informational note); `scan` fails
gracefully once a room's finite token pool is exhausted; `fill_field`/`scan`
each only count as "unique" progress once per field/room per agent.

### `env/economy.py`, `env/institutions.py`

`Economy` balances are floored at zero and never go negative; `gift` checks
the balance explicitly rather than relying on the floor. `InstitutionManager`
rules start `active=False` until `enact_rule()` is called — agents can create
new rules at runtime (e.g. a completed policy plan becomes an institutional
rule via `World.enact_plan_rule`).

## Orchestration (`orchestrator/`)

See [reference-cli.md](reference-cli.md) for the CLI surface. Internals worth
knowing when reading `runner.py`:

- **`Scheduler`** groups co-located, still-available agents into 2-3 person encounters each tick, capped at `--max-events`; it's a fresh random sample every tick with no cross-tick fairness.
- **`MetaOrchestrator`** is a scripted (non-learned) coordinator: it cycles through a fixed reminder list, applies role-specific hint/reminder text from the playbook, and force-nudges agents in rooms that have been silent for **more than 3 ticks** (hardcoded, not config-driven).
- **`ObjectiveManager`** keeps every agent perpetually assigned one active `Objective` (auto-reassigning on completion/failure), tracking progress by scanning successful `ActionLog`s against the objective's `requirements` dict. Two requirement keys (`fill_field`, `scan`) count progress via a type-tolerant `_is_flag_set()` helper that accepts string `"1"`, int `1`, and boolean `True` as success markers (previously a strict string-equality check; see [explanation-known-gaps.md](explanation-known-gaps.md#objective-progress-tracking-uses-strict-string-equality)).
- **`ProbeManager`** schedules at most one self-report/behavioral probe per agent at a time, on a per-kind cooldown. Probe text is injected once per active probe via `ProbeAssignment.inject()` (a previous double-inject bug has been fixed; see [explanation-known-gaps.md](explanation-known-gaps.md#probe-preamble-is-injected-twice)).
- **`SafetyGovernor`** is a substring-match banned-phrase filter (default list: "hate speech", "bomb recipe", "harm yourself", "kill", "attack plan"), not a learned toxicity classifier — `safety.toxicity_threshold` in the run config is declared but never read. On a hit, it nudges every currently-active persona alpha toward zero by `governor_backoff` and logs a `SafetyEvent`.
- **Queue-backed runtime** (`orchestrator/queued_runtime.py`) is an explicit "migration seam" per its own module docstring — see [reference-cli.md](reference-cli.md#queue-backed-runtime-scale-seam).

## Viewer (`viewer/`)

Three independent, mostly-alternative ways to watch a run — pick based on
where you're running from:

| Mode | Flag | Best for |
|---|---|---|
| Live console | `--live` | Any terminal, log files, `tee`/tailing. Checks `sys.stdout.isatty()` before enabling color. |
| ASCII TUI | `--tui` | SSH sessions or low-resource machines with no browser. Takes over the terminal (alternate screen buffer via `rich`). |
| 3D web viewer | `--viewer` | Browser-based spatial visualization. Explicitly a prototype ("designed to be replaced by a full engine... later" per the README) — single static HTML file, Three.js loaded from a CDN (needs internet access), no auth/TLS. |

Details:

- **Console** (`orchestrator/console_logger.py`) prints both actions and dialogue, with truncation controlled by `--full-messages`.
- **TUI** (`viewer/ascii_tui.py`) only logs **chat** to its scrolling log panel — actions are explicitly not rendered there (`_handle_action` is a no-op). No truncation control; long lines simply overflow the fixed-height panel. `--no-color` does not affect it.
- **3D viewer** (`viewer/ws_bridge.py` + `viewer/http_server.py` + `viewer/static/index.html`) streams JSON events over `ws://127.0.0.1:8765/ws` (`init`, `processing`, `action`, `chat`, `tick`, `meta_broadcast`) and serves the page at `http://127.0.0.1:19123`. Agent sphere color is fixed from the agent's persona at `init` time and never recomputed. The browser tab auto-reconnects every 2s if the WebSocket drops. The `#log` panel has no entry cap — long sessions accumulate DOM nodes indefinitely (contrast with the TUI's capped 20-line log).

Port binds happen inside background threads; if a port is already in use, the
bind failure occurs inside that thread and is not surfaced to the CLI — the
run proceeds as if the viewer started successfully. See
[explanation-known-gaps.md](explanation-known-gaps.md) for this and other
viewer edge cases.

## Related

- [reference-cli.md](reference-cli.md) — flags controlling everything above.
- [reference-config.md](reference-config.md) — config schemas.
- [reference-data-schema.md](reference-data-schema.md) — what gets logged/persisted.
- [explanation-known-gaps.md](explanation-known-gaps.md) — every live bug and dead-config surface found while writing this documentation, consolidated.
