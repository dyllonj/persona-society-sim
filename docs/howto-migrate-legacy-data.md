# How to migrate legacy data

Two independent migrations you may need, depending on when your prompt files
or simulation dumps were created.

## Migrating legacy prompt schema files

If you have prompt files using the old `situation`/`positive`/`negative` keys
instead of the current forced-choice `option_a`/`option_b` schema:

```bash
python -m data.prompts.schema convert old_prompts.jsonl data/prompts/newtrait.jsonl

# swap which option is treated as "high" if the legacy file's polarity was reversed
python -m data.prompts.schema convert old_prompts.jsonl data/prompts/newtrait.jsonl --swap-options

python -m data.prompts.schema validate data/prompts/*.jsonl
```

Then regenerate and evaluate vectors from the converted files — see
[howto-compute-steering-vectors.md](howto-compute-steering-vectors.md).

## Migrating steering vector metadata to the current format

If you're picking up an old checkout that still used the legacy `[12, 16, 20]`
hardcoded layer default:

1. Set `vector_root` in `configs/steering.layers.yaml` to where your `.meta.json`/`.npy` bundles live.
2. Under `traits`, declare `vector_store_id`, `prompt_file`, and `layers` per trait — this file is the single source of truth for both extraction and runtime loading now; there is no hidden fallback layer list anymore (`steering/compute_caa.py` raises if `--layers` is omitted).
3. Regenerate: `./scripts/compute_vectors.sh` (see [howto-compute-steering-vectors.md](howto-compute-steering-vectors.md) for the model/layer mismatch to check first).
4. Evaluate: `./scripts/eval_vectors.sh` with `STEERING_ALPHA` matching your intended `steering.strength`.
5. Confirm your run config's `steering.metadata_files.vectors` points at `configs/steering.layers.yaml` and `metadata_files.personas` points at `configs/personas.bigfive.yaml`.
6. Do a `--mock-model` dry run to confirm vectors load via the metadata-aware loader before committing to a full HF run.

Re-run steps 3-4 any time the prompt files or `steering.layers.yaml` change,
and check the resulting `data/vectors/*.meta.json` diff into your experiment
log for provenance.

## Removing legacy trade/marketplace records from an old dump

An earlier version of the simulator supported a `trade` action and logged
`econ_txn` rows plus `trade`-kind graph edges. That action is now disabled at
the action-router level (`env/actions.py`), but old dump directories created
before the removal may still contain `trade` rows/edges that current schema
validation will reject if loaded through the Pydantic models. Clean them up
with:

```bash
python scripts/migrate_remove_trade_records.py --sqlite <log.db> --parquet-dir <dump_root>
```

This deletes `action_type='trade'` rows from `action_log` and all rows from
`econ_txn` in SQLite, and row-filters/rewrites the equivalent Parquet files —
including stripping `trade`-kind edges out of `graph_snapshot.edges` JSON
blobs. Run this before pointing any current-schema analysis tooling at an old
dump; there's no automatic detection of pre-migration data, so this is a
manual step you need to remember.

## Related

- [howto-compute-steering-vectors.md](howto-compute-steering-vectors.md)
- [reference-data-schema.md](reference-data-schema.md) — current table schemas, including the DDL-only tables left over from the trade feature.
- [explanation-known-gaps.md](explanation-known-gaps.md)
