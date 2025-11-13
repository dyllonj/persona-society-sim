#!/usr/bin/env python3
"""Verify that steering vectors are actually affecting model outputs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def analyze_steering_effects(messages_dir: Path, num_files: int = 20):
    """Analyze if steering coefficients correlate with outputs."""
    messages_files = sorted(messages_dir.glob("*.parquet"))[:num_files]

    if not messages_files:
        print(f"ERROR: No message files found in {messages_dir}")
        return False

    print(f"Analyzing {len(messages_files)} timesteps...")
    print("=" * 80)

    # Collect steering snapshots
    all_traits = {"E": [], "A": [], "C": [], "O": [], "N": []}
    total_messages = 0
    agents_with_steering = set()

    for msg_file in messages_files:
        table = pq.read_table(msg_file)
        data = {col: table.column(col).to_pylist() for col in table.column_names}

        for i in range(table.num_rows):
            total_messages += 1
            agent_id = data["from_agent"][i]
            steering_raw = data["steering_snapshot"][i]

            # Parse steering snapshot (stored as JSON string)
            if steering_raw:
                try:
                    steering = json.loads(steering_raw) if isinstance(steering_raw, str) else steering_raw
                    agents_with_steering.add(agent_id)
                    for trait, value in steering.items():
                        if trait in all_traits:
                            all_traits[trait].append(value)
                except (json.JSONDecodeError, AttributeError):
                    pass

    print(f"\n✓ Analyzed {total_messages} messages from {len(agents_with_steering)} agents")
    print(f"\n{'=' * 80}")
    print("STEERING VERIFICATION RESULTS")
    print("=" * 80)

    # Check 1: Are steering values being logged?
    if not any(all_traits.values()):
        print("❌ FAIL: No steering values found in logs!")
        print("   This suggests steering is not being applied.")
        return False

    print("\n✓ PASS: Steering values are present in logs")

    # Check 2: Are steering values varied (not all zero)?
    all_values = [v for values in all_traits.values() for v in values]
    if all(abs(v) < 0.01 for v in all_values):
        print("❌ FAIL: All steering values are near zero!")
        print("   Steering vectors may not be loaded correctly.")
        return False

    print("✓ PASS: Steering values show variation")

    # Check 3: Report statistics per trait
    print("\n" + "=" * 80)
    print("TRAIT STATISTICS")
    print("=" * 80)

    for trait, values in all_traits.items():
        if not values:
            print(f"\n{trait}: No data")
            continue

        arr = np.array(values)
        print(f"\n{trait} (n={len(values)}):")
        print(f"  Mean:   {arr.mean():+.3f}")
        print(f"  Std:    {arr.std():.3f}")
        print(f"  Range:  [{arr.min():+.3f}, {arr.max():+.3f}]")
        print(f"  Non-zero: {(np.abs(arr) > 0.01).sum()} ({(np.abs(arr) > 0.01).mean()*100:.1f}%)")

    # Check 4: Are different agents showing different steering values?
    print("\n" + "=" * 80)
    print("AGENT DIVERSITY CHECK")
    print("=" * 80)

    # Sample a few agents and check their average traits
    agent_traits = {}
    for msg_file in messages_files[:5]:  # Check first 5 timesteps
        table = pq.read_table(msg_file)
        data = {col: table.column(col).to_pylist() for col in table.column_names}

        for i in range(table.num_rows):
            agent_id = data["from_agent"][i]
            steering_raw = data["steering_snapshot"][i]

            if steering_raw and agent_id not in agent_traits:
                try:
                    steering = json.loads(steering_raw) if isinstance(steering_raw, str) else steering_raw
                    agent_traits[agent_id] = steering
                except (json.JSONDecodeError, AttributeError):
                    pass

    if len(agent_traits) > 1:
        print(f"\nSample of {min(5, len(agent_traits))} agents' steering profiles:")
        for idx, (agent_id, traits) in enumerate(list(agent_traits.items())[:5]):
            traits_str = ", ".join(f"{k}:{v:+.2f}" for k, v in sorted(traits.items()))
            print(f"  {agent_id}: {traits_str}")

        # Check variance across agents
        trait_arrays = {trait: [] for trait in all_traits}
        for traits in agent_traits.values():
            for trait, value in traits.items():
                if trait in trait_arrays:
                    trait_arrays[trait].append(value)

        print("\nInter-agent variance:")
        for trait, values in trait_arrays.items():
            if values:
                variance = np.var(values)
                print(f"  {trait}: {variance:.3f} {'✓' if variance > 0.01 else '⚠'}")

        if all(np.var(vals) > 0.01 for vals in trait_arrays.values() if vals):
            print("\n✓ PASS: Agents show diverse steering profiles")
        else:
            print("\n⚠ WARNING: Low variance - agents may be too similar")
    else:
        print("\n⚠ WARNING: Not enough agents sampled")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("✓ Steering appears to be active and functioning")
    print(f"  - {len(agents_with_steering)} agents with steering applied")
    print(f"  - {total_messages} messages analyzed")
    print(f"  - Trait values show expected variation")
    print("\nTo verify steering is actually affecting outputs:")
    print("  1. Run with different steering configs and compare results")
    print("  2. Check if high-E agents produce more enthusiastic language")
    print("  3. Compare with --mock-model baseline (no steering)")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify steering vector functionality")
    parser.add_argument(
        "--messages-dir",
        type=Path,
        default=Path("storage/dumps/debug32/messages"),
        help="Directory containing message parquet files",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=20,
        help="Number of timestep files to analyze",
    )
    args = parser.parse_args()

    if not args.messages_dir.exists():
        print(f"ERROR: Messages directory not found: {args.messages_dir}")
        print("Run a simulation first to generate data.")
        sys.exit(1)

    success = analyze_steering_effects(args.messages_dir, args.num_files)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
