# Parquet schema reference

Simulation dumps now include two additional datasets per tick alongside the existing
`actions/` and `messages/` directories:

| Directory | Schema | Notes |
|-----------|--------|-------|
| `graph_snapshots/` | `run_id`, `tick`, `trait_key`, `edges`, `centrality`, `band_metadata` | `edges` is a JSON list of `Edge` objects (`src`, `dst`, `weight`, `kind`). `trait_key` equals `"global"` for the population-wide snapshot and otherwise uses the `"TRAIT:BAND"` key produced by `metrics.persona_bands`. |
| `metrics_snapshots/` | `run_id`, `tick`, `trait_key`, `band_metadata`, `cooperation_rate`, `gini_wealth`, `polarization_modularity`, `conflicts`, `rule_enforcement_cost` | `band_metadata` captures the thresholds used to derive the `trait_key`. |

Both directories emit one Parquet file per tick (e.g., `graph_snapshots_t00010.parquet`).
Each file may contain multiple rows when multiple trait cohorts produced data for that tick.
Downstream notebooks can treat `trait_key="global"` as the aggregate view and filter for
other keys to study cohorts.

See `metrics/persona_bands.py` for the categorization logic and
`metrics/tick_instrumentation.py` for the tick-level collection pipeline.
