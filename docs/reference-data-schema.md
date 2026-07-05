# Data schema reference

*(Renamed/expanded from the old `parquet_schema.md`, which only documented 5
of the 10 tables actually written. This file supersedes it.)*

Every simulation run dual-writes structured telemetry: once to a SQL database
(SQLite or Postgres, via `storage/db.py`) and once to Parquet (one file per
tick per record kind, via `storage/log_sink.py`), both driven by the same
Pydantic models in `schemas/logs.py`. Both writes happen inside `LogSink`;
records are buffered in memory and only hit disk/DB on `flush(tick)` — a crash
mid-tick loses that tick's unflushed buffer from **both** stores.

## Active tables (written every run)

All ten live under `{parquet_dir}/<name>/` as `<name>_t{tick:05d}.parquet`
(only written for ticks that produced ≥1 record of that kind), and as a
same-named SQL table when `logging.db_url` is set.

| Table | Key columns | Written on |
|---|---|---|
| `actions` | `action_id`, `run_id`, `tick`, `agent_id`, `action_type`, `params` (JSON), `outcome`, `info` (JSON), `prompt_text`, `prompt_hash`, `plan_metadata` (JSON), `reflection_summary`, `reflection_implications` (JSON list) | every executed action |
| `messages` | `msg_id`, `run_id`, `tick`, `channel`, `from_agent`, `to_agent`, `room_id`, `content`, `tokens_in`, `tokens_out`, `temperature`, `top_p`, `steering_snapshot` (JSON), `layers_used` (JSON list) | every `talk` action |
| `safety` | `event_id`, `run_id`, `tick`, `agent_id`, `kind`, `severity`, `applied_alpha_delta` (JSON) | only when `SafetyGovernor` flags a generation |
| `graph_snapshots` | `run_id`, `tick`, `trait_key`, `edges` (JSON list of `{src,dst,weight,kind}`), `centrality` (JSON), `band_metadata` (JSON) | every tick, once per active trait cohort + one `"global"` row |
| `metrics_snapshots` | `run_id`, `tick`, `trait_key`, `band_metadata`, `cooperation_rate`, `gini_wealth`, `polarization_modularity`, `conflicts`, `rule_enforcement_cost`, `prompt_duplication_rate`, `plan_reuse_rate` | every tick, same cohort structure as graph_snapshots |
| `research_facts` | `log_id`, `run_id`, `tick`, `agent_id`, `doc_id`, `fact_id`, `fact_answer`, `target_answer`, `correct`, `trait_key`, `trait_band`, `alpha_value`, `alpha_bucket` | one row per revealed fact on a `research` action |
| `citations` | `log_id`, `run_id`, `tick`, `agent_id`, `doc_id`, `trait_key`, `trait_band`, `alpha_value`, `alpha_bucket` | one row per `cite` action |
| `report_grades` | `log_id`, `run_id`, `tick`, `agent_id`, `targets_total`, `facts_correct`, `citations_valid`, `reward_points`, `trait_key`, `trait_band`, `alpha_value`, `alpha_bucket` | one row per `submit_report` action |
| `probe_logs` | `log_id`, `run_id`, `tick`, `agent_id`, `probe_id`, `question`, `prompt_text`, `response_text`, `trait`, `score`, `parser_hint` | one row per answered Likert self-report probe |
| `behavior_probes` | `log_id`, `run_id`, `tick`, `agent_id`, `probe_id`, `scenario`, `prompt_text`, `response_text`, `outcome`, `parser_hint` | one row per answered scripted behavioral probe |

**`trait_key`**: `null` in the Pydantic model becomes the literal string
`"global"` at write time (`LogSink._normalize`), representing the
population-wide cohort. Otherwise it's a `"TRAIT:BAND"` key (e.g. `"A:low"`)
from `metrics/persona_bands.py`, identifying whichever trait had the largest
combined (base + steering-delta) magnitude for that action, banded at ±1.5.

**JSON columns are always plain strings**, never native Postgres arrays/JSONB
— `LogSink._normalize` runs every dict/list field through `json.dumps` before
either the SQL insert or the Parquet write, regardless of what the DDL
declares (`msg_log.layers_used` is typed `INT[]` in the DDL but is actually
stored as a JSON string like `"[1, 2]"`). Downstream consumers must
`json.loads()` these columns explicitly (see `scripts/analyze_simulation.py`
for the pattern).

## DDL-only tables (created, never populated)

`storage/db.py`'s `Database.init()` issues `CREATE TABLE IF NOT EXISTS` for
these too, but no code path in this repo currently inserts into them:

- `agent_state`, `memory_event` — agent state and memory live only in-process (`AgentState`, `MemoryStore`); never persisted here.
- `econ_txn`, `vote_log`, `sanction_log` — leftovers from a removed marketplace/trade feature. `scripts/migrate_remove_trade_records.py` is the evidence of that removal (it deletes `action_type='trade'` rows and filters `trade` edges out of `graph_snapshot.edges` JSON blobs).
- `steering_vector_store` — steering vector metadata is instead persisted to a JSONL sidecar (`data/vectors/index.jsonl`) by `steering/vector_store.py`, not to SQL.
- `run_config`, `run_summary` — run configuration is read from YAML and never round-tripped into these tables.

Don't expect a `SELECT * FROM agent_state` to return anything — if you need
current agent state, read the in-memory objects during a run or add the
missing write path yourself.

## Separate output: `MetricTracker` (not part of `LogSink`)

`metrics/tracker.py`'s `MetricTracker` writes its own files at end-of-run, to
the same directory tree as the Parquet dumps (`log_sink.parquet_dir`, or
`metrics/` by default) — a second, independent persistence path that does
**not** go through `storage/db.py`:

- `run_{run_id}.jsonl` — line 1 is `{"summary": {...}}` (tick-level collab ratio, trait-band aggregates, alpha-magnitude bucket aggregates, research coverage); subsequent lines are one JSON object per agent (`total_actions`, `efficiency`, `collab_ratio`, `time_to_submit`, etc).
- `run_{run_id}_agents.parquet` — per-agent summary rows.
- `run_{run_id}_trait_aggregates.parquet` — per `"trait:band"` cohort rows (agent count, efficiency, collab ratio, submit rate, etc).
- `run_{run_id}_alpha_aggregates.parquet` — per-trait average steering magnitude, bucketed into `<0.5` / `0.5-1.5` / `>1.5`.

Note `MetricTracker`'s own trait-banding (`_band_for_value`, hardcoded ±1.5)
duplicates `metrics/persona_bands.py::BAND_THRESHOLDS` rather than importing
it — if the threshold is ever changed in one place, the other silently drifts
out of sync.

## Schema evolution

There is no migration/versioning framework — `Database.init()` only issues
`CREATE TABLE IF NOT EXISTS`, with no `ALTER TABLE` path. Schema changes are
handled by one-off scripts instead; `scripts/migrate_remove_trade_records.py`
is the template for that pattern (rewrite SQLite tables in place, row-filter
and rewrite affected Parquet files). If you're removing or renaming a field
in `schemas/logs.py`, write a similar script rather than assuming old dumps
stay compatible.

## Related

- [howto-analyze-runs.md](howto-run-simulations.md#inspecting-a-completed-run) — reading these tables back out.
- [reference-modules.md](reference-modules.md) — what produces each record kind.
- [explanation-known-gaps.md](explanation-known-gaps.md) — telemetry failure modes (metrics collection is wrapped in bare `except: pass` throughout the runner, so gaps in these tables can occur silently).
