#!/usr/bin/env python3
"""Quick viewer for simulation parquet results."""

import argparse
from pathlib import Path
import pyarrow.parquet as pq


def view_file(filepath: Path, num_rows: int = 5):
    """Display contents of a parquet file."""
    table = pq.read_table(filepath)
    print(f"\n{'='*80}")
    print(f"File: {filepath.name}")
    print(f"{'='*80}")
    print(f"Rows: {table.num_rows}, Columns: {table.num_columns}")
    print(f"\nSchema:")
    print(table.schema)

    # Convert to Python dict for display
    data = {col: table.column(col).to_pylist() for col in table.column_names}

    print(f"\nFirst {min(num_rows, table.num_rows)} rows:")
    for i in range(min(num_rows, table.num_rows)):
        print(f"\n--- Row {i+1} ---")
        for col in table.column_names:
            val = data[col][i]
            if isinstance(val, str) and len(val) > 100:
                print(f"  {col}: {val[:100]}...")
            else:
                print(f"  {col}: {val}")


def main():
    parser = argparse.ArgumentParser(description="View parquet simulation results")
    parser.add_argument("file", type=Path, help="Parquet file to view")
    parser.add_argument("-n", "--rows", type=int, default=5, help="Number of rows to display")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: File {args.file} not found")
        return

    view_file(args.file, args.rows)


if __name__ == "__main__":
    main()
