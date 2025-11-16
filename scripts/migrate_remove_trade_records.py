"""Remove deprecated trade interactions from persisted logs.

This utility cleans up SQLite tables and Parquet shards that still
reference the deprecated ``trade`` action or graph edges with
``kind == "trade"``. Run it before upgrading so downstream notebooks no
longer expect marketplace data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def _load_pyarrow():
    if importlib.util.find_spec("pyarrow") is None:
        raise SystemExit("pyarrow is required to clean Parquet dumps")
    import pyarrow as pa  # type: ignore
    import pyarrow.compute as pc  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    return pa, pc, pq


def _filter_trade_edges(edges_raw: str | List[Dict]) -> Tuple[List[Dict], bool]:
    parsed = edges_raw
    if isinstance(edges_raw, str):
        parsed = json.loads(edges_raw)
    filtered = [edge for edge in parsed if edge.get("kind") != "trade"]
    return filtered, filtered != parsed


def _clean_sqlite(db_path: Path, dry_run: bool) -> Dict[str, int]:
    summary = {"actions_removed": 0, "econ_rows_removed": 0, "graph_rows_rewritten": 0}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM action_log WHERE action_type = 'trade'")
    summary["actions_removed"] = cursor.fetchone()[0] or 0
    if summary["actions_removed"] and not dry_run:
        cursor.execute("DELETE FROM action_log WHERE action_type = 'trade'")

    cursor.execute("SELECT COUNT(*) FROM econ_txn")
    summary["econ_rows_removed"] = cursor.fetchone()[0] or 0
    if summary["econ_rows_removed"] and not dry_run:
        cursor.execute("DELETE FROM econ_txn")

    cursor.execute("SELECT rowid, edges FROM graph_snapshot WHERE edges LIKE '%trade%'")
    for rowid, edges_raw in cursor.fetchall():
        if not edges_raw:
            continue
        filtered, changed = _filter_trade_edges(edges_raw)
        if changed:
            summary["graph_rows_rewritten"] += 1
            if not dry_run:
                cursor.execute(
                    "UPDATE graph_snapshot SET edges = ? WHERE rowid = ?",
                    (json.dumps(filtered), rowid),
                )

    if not dry_run:
        conn.commit()
    conn.close()
    return summary


def _clean_parquet(parquet_dir: Path, dry_run: bool) -> Dict[str, int]:
    pa, pc, pq = _load_pyarrow()
    summary = {"action_rows_removed": 0, "graph_files_rewritten": 0}

    for path in parquet_dir.rglob("*.parquet"):
        table = pq.read_table(path)
        modified = False

        if "action_type" in table.schema.names:
            keep_mask = pc.not_equal(table["action_type"], "trade")
            filtered = table.filter(keep_mask)
            removed = table.num_rows - filtered.num_rows
            if removed > 0:
                summary["action_rows_removed"] += removed
                table = filtered
                modified = True

        if "edges" in table.schema.names:
            edges_py = table["edges"].to_pylist()
            rewritten = False
            cleaned_edges: List[str | None] = []
            for edges_raw in edges_py:
                if edges_raw is None:
                    cleaned_edges.append(None)
                    continue
                filtered, changed = _filter_trade_edges(edges_raw)
                cleaned_edges.append(json.dumps(filtered))
                rewritten = rewritten or changed
            if rewritten:
                column = pa.array(cleaned_edges, type=table.schema.field("edges").type)
                table = table.set_column(table.schema.get_field_index("edges"), "edges", column)
                summary["graph_files_rewritten"] += 1
                modified = True

        if modified and not dry_run:
            pq.write_table(table, path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove deprecated trade logs from persisted data.")
    parser.add_argument("--sqlite", type=Path, nargs="*", help="SQLite database file(s) to clean")
    parser.add_argument("--parquet-dir", type=Path, help="Root directory containing Parquet dumps", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Report counts without modifying files")
    args = parser.parse_args()

    if not args.sqlite and not args.parquet_dir:
        raise SystemExit("Provide at least one --sqlite database or --parquet-dir to clean")

    if args.sqlite:
        for db_path in args.sqlite:
            summary = _clean_sqlite(db_path, args.dry_run)
            print(f"[sqlite] {db_path}: removed {summary['actions_removed']} trade actions, "
                  f"{summary['econ_rows_removed']} econ rows; rewrote {summary['graph_rows_rewritten']} graph snapshots")

    if args.parquet_dir:
        summary = _clean_parquet(args.parquet_dir, args.dry_run)
        print(
            f"[parquet] {args.parquet_dir}: removed {summary['action_rows_removed']} trade action rows; "
            f"rewrote {summary['graph_files_rewritten']} graph shard(s)"
        )


if __name__ == "__main__":
    main()
