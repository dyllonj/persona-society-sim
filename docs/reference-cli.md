# CLI reference: `orchestrator.cli`

Reference for every flag accepted by the simulation entry point:

```bash
python3 -m orchestrator.cli <config.yaml> [flags]
```

`<config.yaml>` is required and must be a path to one of the `configs/run.*.yaml`
files (or your own file following the same schema — see
[reference-config.md](reference-config.md)).

## Model / backend selection

| Flag | Type / default | Effect |
|---|---|---|
| `--mock-model` | flag, off | Use `MockBackend` (deterministic, no model weights) instead of a real language backend. Fastest way to smoke-test a config. |
| `--no-steering` | flag, off | Disable persona steering entirely. Sets effective alpha strength to `0.0`, skips loading vectors from disk, and runs every agent with neutral (zeroed) trait coefficients. Overrides `steering.enabled` in the config either way. |
| `--vector-dir` | path, `data/vectors` | Root directory to search for `.npy`/`.meta.json` steering vector bundles when steering is enabled. |

If `--mock-model` is not passed, the CLI loads a local Hugging Face model
(`model_name` from the config) and steers it via activation addition (CAA) —
see [reference-modules.md](reference-modules.md#steering) and
[explanation-steering.md](explanation-steering.md).

## Scenario and environment

| Flag | Type / default | Effect |
|---|---|---|
| `--env` | `research` \| `policy` \| `nav`, default `research` | Selects the built-in scenario. `research`: gather facts at the library and `submit_report`. `policy`: fill checklist fields, draft a summary, `submit_plan`. `nav`: visit unique rooms and `scan` tokens. |
| `--difficulty` | int, default `3` | Scenario-specific difficulty knob: target fact count (research), required checklist fields (policy), or discovery tokens per agent (nav). |
| `--max-events` | int, default `16` | Maximum number of encounters (room groupings, 1-3 agents each) the scheduler creates per tick. This bounds encounters, not agents directly — roughly `3 * max_events` is the per-tick agent-action ceiling. Agents left over once the cap is hit simply get no encounter (and take no action) that tick; there's no rotation/fairness guarantee for who gets skipped. |

## Viewing / output

| Flag | Type / default | Effect |
|---|---|---|
| `--live` | flag, off | Stream tick/action/dialogue output to stdout via `ConsoleLogger`. |
| `--no-color` | flag, off | Disable ANSI colors in `--live` console output. Has **no effect** on `--tui` (the TUI's colors come from `rich`'s own terminal detection, not this flag). |
| `--full-messages` | flag, off | Disable dialogue truncation in `--live` console output (default truncates to ~120 chars / 3 lines). Console-only; the TUI has no truncation control at all. |
| `--viewer` | flag, off | Start a WebSocket bridge (`ws://127.0.0.1:8765/ws`) and static file server (`http://127.0.0.1:19123`) for the 3D Three.js viewer. Both are hardcoded to `127.0.0.1` with no host/port override flags. If startup fails, the failure is only logged when `--live` is also set — otherwise it fails silently and the run proceeds with no viewer. |
| `--tui` | flag, off | Use the ASCII terminal dashboard (`rich`-based, alternate screen buffer) instead of the web viewer. **Forcibly disables `--live`** regardless of whether you passed it, to avoid both fighting over the terminal. Requires `rich`; unlike `--viewer`, an import failure here is an uncaught crash, not a soft warning. |

`--viewer` and `--tui` share one `event_bridge` slot: if you pass both, `--tui` wins (it's checked second and unconditionally overwrites the assignment), but the web viewer's WS/HTTP server threads are still left running on their ports with nothing broadcasting to them. See [reference-modules.md](reference-modules.md#viewer-viewer) for details on all three viewing modes.

## Queue-backed runtime (scale seam)

These flags only matter once you pass `--queued-runtime`; the default runtime does all logging/broadcasting synchronously on the main tick loop.

| Flag | Type / default | Effect |
|---|---|---|
| `--queued-runtime` | flag, off | Move log-sink writes and viewer/TUI broadcasts onto background worker threads via bounded queues, instead of doing that I/O inline in the tick loop. Explicitly a migration seam, not the final architecture. |
| `--async-log-flush` | flag, off | Only with `--queued-runtime`. Don't block the per-tick `flush()` call waiting for the queued write to land — trades durability-ordering guarantees for throughput. |
| `--log-queue-size` | int, default `10000` | Max size of the log-sink's internal queue. Enqueues **block** (not drop) when full — telemetry is never silently lost, but a slow DB/Parquet writer can stall the tick loop. |
| `--event-queue-size` | int, default `10000` | Max size of the viewer/TUI event queue. Broadcasts are **dropped** (counted, not blocking) when full — viewer updates are best-effort. |
| `--decision-workers` | int, default `1` | Size of a thread pool used to compute agent decisions (`Agent.act()` calls) concurrently. Does **not** parallelize world-state mutation — the runner still applies decisions with synchronous barriers, so this speeds up concurrent LLM calls only. |
| `--batch-decisions-per-encounter` | flag, off | Compute all participants' decisions for an encounter together before applying any of them, instead of decide-then-apply one at a time. Tradeoff: participants no longer see each other's utterances from the same encounter (they all act off the same pre-encounter transcript). |

## What happens on error

The full build-and-run sequence (viewer/TUI setup through `runner.run(...)`) is wrapped in one `try/except/finally`.

- **On success**: `finally` runs both `event_bridge.stop()` and `log_sink.close()`, ensuring the background log worker thread is cleanly shut down and any final buffer flush errors are surfaced (previously `log_sink.close()` was only called on the exception path — see [explanation-known-gaps.md](explanation-known-gaps.md#log_sinkclose-is-never-called-on-a-successful-run)).
- **On failure**: the `except` block prints the traceback and exits `1`; `finally` then runs the same cleanup (`event_bridge.stop()` + `log_sink.close()`) as the success path.

See [explanation-known-gaps.md](explanation-known-gaps.md) for the historical context (previously, partial/queued telemetry was not durably flushed even on a clean run).

## Related

- [reference-config.md](reference-config.md) — full schema of `configs/run.*.yaml` and the metadata files it references.
- [howto-run-simulations.md](howto-run-simulations.md) — task-oriented walkthroughs using these flags.
- [tutorial-getting-started.md](tutorial-getting-started.md) — first run, step by step.
