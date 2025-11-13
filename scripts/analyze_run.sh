#!/usr/bin/env bash
set -euo pipefail

RUN_ID=${1:-debug32}
DUMP_DIR="storage/dumps/${RUN_ID}"

echo "Analyzing simulation run: ${RUN_ID}"
echo "==========================================="

# Check if dump directory exists
if [ ! -d "$DUMP_DIR" ]; then
    echo "ERROR: Dump directory not found: $DUMP_DIR"
    echo "Run a simulation first to generate data."
    exit 1
fi

# Run comprehensive analysis
echo ""
echo "Running comprehensive analysis..."
python3 scripts/analyze_simulation.py --dump-dir "$DUMP_DIR"

echo ""
echo "Verifying steering vectors..."
python3 scripts/verify_steering.py --messages-dir "$DUMP_DIR/messages"

echo ""
echo "==========================================="
echo "Analysis complete! Check storage/analysis/ for plots."
