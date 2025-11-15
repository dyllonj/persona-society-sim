# Parquet schema reference

Simulation dumps now include two additional datasets per tick alongside the existing
`actions/` and `messages/` directories:

| Directory | Schema | Notes |
|-----------|--------|-------|
| `graph_snapshots/` | `run_id`, `tick`, `trait_key`, `edges`, `centrality`, `band_metadata` | `edges` is a JSON list of `Edge` objects (`src`, `dst`, `weight`, `kind`). `trait_key` equals `"global"` for the population-wide snapshot and otherwise uses the `"TRAIT:BAND"` key produced by `metrics.persona_bands`. |
| `metrics_snapshots/` | `run_id`, `tick`, `trait_key`, `band_metadata`, `cooperation_rate`, `gini_wealth`, `polarization_modularity`, `conflicts`, `rule_enforcement_cost` | `band_metadata` captures the thresholds used to derive the `trait_key`. |
| `research_facts/` | `log_id`, `run_id`, `tick`, `agent_id`, `doc_id`, `fact_id`, `fact_answer`, `target_answer`, `correct`, `trait_key`, `trait_band`, `alpha_value`, `alpha_bucket` | Each row represents a revealed fact during a research action along with cohort metadata. |
| `citations/` | `log_id`, `run_id`, `tick`, `agent_id`, `doc_id`, `trait_key`, `trait_band`, `alpha_value`, `alpha_bucket` | Emits one row per cite action. |
| `report_grades/` | `log_id`, `run_id`, `tick`, `agent_id`, `targets_total`, `facts_correct`, `citations_valid`, `reward_points`, `trait_key`, `trait_band`, `alpha_value`, `alpha_bucket` | Structured grading output for `submit_report`. |

Both directories emit one Parquet file per tick (e.g., `graph_snapshots_t00010.parquet`).
Each file may contain multiple rows when multiple trait cohorts produced data for that tick.
Downstream notebooks can treat `trait_key="global"` as the aggregate view and filter for
other keys to study cohorts.

See `metrics/persona_bands.py` for the categorization logic and
`metrics/tick_instrumentation.py` for the tick-level collection pipeline.
