#!/usr/bin/env python3
"""Comprehensive post-simulation analysis and visualization."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns


def _read_parquet_dir(directory: Path):
    records = []
    if not directory.exists():
        return records
    for file_path in sorted(directory.glob("*.parquet")):
        table = pq.read_table(file_path)
        data = {col: table.column(col).to_pylist() for col in table.column_names}
        for i in range(table.num_rows):
            records.append({col: data[col][i] for col in table.column_names})
    return records


def load_simulation_data(dump_dir: Path):
    """Load all parquet data from a simulation run."""
    print(f"Loading data from {dump_dir}...")
    actions = _read_parquet_dir(dump_dir / "actions")
    messages = _read_parquet_dir(dump_dir / "messages")
    graph_snapshots = _read_parquet_dir(dump_dir / "graph_snapshots")
    metrics_snapshots = _read_parquet_dir(dump_dir / "metrics_snapshots")
    print(
        "  Loaded %d actions, %d messages, %d graph snapshots, %d metrics snapshots"
        % (len(actions), len(messages), len(graph_snapshots), len(metrics_snapshots))
    )
    return actions, messages, graph_snapshots, metrics_snapshots


def _auto_detect_metric_log(dump_dir: Path) -> Optional[Path]:
    metrics_dir = dump_dir / "metrics"
    if not metrics_dir.exists():
        return None
    candidates = sorted(metrics_dir.glob("run_*.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def load_metric_tracker_summary(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if not first_line:
        return None
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    return payload.get("summary")


def analyze_metric_tracker_summary(summary: dict, source: Path) -> None:
    print("\n" + "=" * 80)
    print("METRIC TRACKER SUMMARY")
    print("=" * 80)
    print(f"Source: {source}")

    trait_aggregates = summary.get("trait_band_aggregates", {}) or {}
    if trait_aggregates:
        print("\nTop trait bands by total actions:")
        top_bands = sorted(
            trait_aggregates.items(),
            key=lambda item: item[1].get("total_actions", 0),
            reverse=True,
        )[:5]
        for key, stats in top_bands:
            eff = stats.get("efficiency", 0.0)
            collab = stats.get("collab_ratio", 0.0)
            actions = stats.get("total_actions", 0)
            submit_rate = stats.get("submit_rate", 0.0)
            print(
                f"  {key:10s}: actions={actions}, eff={eff:.3f}, collab={collab:.3f}, submit={submit_rate:.3f}"
            )
    else:
        print("\nNo trait band aggregates were recorded.")

    alpha_buckets = summary.get("alpha_buckets", {}) or {}
    if alpha_buckets:
        print("\nSteering magnitude buckets:")
        labels = list(summary.get("alpha_bucket_labels", {}).keys())
        if not labels and alpha_buckets:
            sample = next(iter(alpha_buckets.values()))
            labels = list(sample.get("bucket_counts", {}).keys())
        for trait, stats in sorted(alpha_buckets.items()):
            counts = stats.get("bucket_counts", {})
            bucket_str = ", ".join(f"{label}:{counts.get(label, 0)}" for label in labels)
            avg = stats.get("avg_magnitude", 0.0)
            samples = stats.get("samples", 0)
            print(f"  {trait:5s}: avg|alpha|={avg:.3f} ({samples} samples) -> {bucket_str}")
    else:
        print("\nNo steering bucket data recorded.")


def analyze_action_distribution(actions):
    """Analyze distribution of action types."""
    action_types = Counter(a["action_type"] for a in actions)
    outcomes = Counter(a["outcome"] for a in actions)

    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION")
    print("=" * 80)
    print(f"\nTotal actions: {len(actions)}")
    print("\nAction types:")
    for action_type, count in action_types.most_common():
        pct = count / len(actions) * 100
        print(f"  {action_type:15s}: {count:5d} ({pct:5.1f}%)")

    print("\nOutcomes:")
    for outcome, count in outcomes.most_common():
        pct = count / len(actions) * 100
        print(f"  {outcome:15s}: {count:5d} ({pct:5.1f}%)")

    return action_types, outcomes


def analyze_agent_activity(actions, messages):
    """Analyze per-agent activity levels."""
    agent_actions = Counter(a["agent_id"] for a in actions)
    agent_messages = Counter(m["from_agent"] for m in messages)

    print("\n" + "=" * 80)
    print("AGENT ACTIVITY")
    print("=" * 80)
    print(f"\nTotal agents: {len(agent_actions)}")
    print(f"\nMost active agents (by actions):")
    for agent_id, count in agent_actions.most_common(5):
        print(f"  {agent_id}: {count} actions, {agent_messages.get(agent_id, 0)} messages")

    print(f"\nLeast active agents:")
    for agent_id, count in list(agent_actions.most_common())[-5:]:
        print(f"  {agent_id}: {count} actions, {agent_messages.get(agent_id, 0)} messages")

    return agent_actions, agent_messages


def analyze_token_usage(messages):
    """Analyze token consumption."""
    total_in = sum(m["tokens_in"] for m in messages)
    total_out = sum(m["tokens_out"] for m in messages)

    print("\n" + "=" * 80)
    print("TOKEN USAGE")
    print("=" * 80)
    print(f"\nTotal messages: {len(messages)}")
    print(f"Total input tokens:  {total_in:,}")
    print(f"Total output tokens: {total_out:,}")
    print(f"Total tokens:        {total_in + total_out:,}")
    print(f"\nAverage per message:")
    print(f"  Input:  {total_in / len(messages):.1f} tokens")
    print(f"  Output: {total_out / len(messages):.1f} tokens")

    return total_in, total_out


def analyze_steering_profiles(messages):
    """Analyze steering coefficient distributions."""
    # Collect all steering snapshots
    agent_steering = defaultdict(list)

    for msg in messages:
        steering_raw = msg.get("steering_snapshot")
        if steering_raw:
            try:
                steering = json.loads(steering_raw) if isinstance(steering_raw, str) else steering_raw
                agent_id = msg["from_agent"]
                agent_steering[agent_id].append(steering)
            except (json.JSONDecodeError, AttributeError):
                pass

    print("\n" + "=" * 80)
    print("STEERING ANALYSIS")
    print("=" * 80)

    # Average steering per agent
    agent_avg_steering = {}
    for agent_id, snapshots in agent_steering.items():
        avg = {}
        for trait in ["E", "A", "C", "O", "N"]:
            values = [s[trait] for s in snapshots if trait in s]
            if values:
                avg[trait] = np.mean(values)
        agent_avg_steering[agent_id] = avg

    print(f"\nAgents with steering: {len(agent_avg_steering)}")
    print("\nSample agent profiles:")
    for agent_id in sorted(agent_avg_steering.keys())[:5]:
        traits_str = ", ".join(
            f"{k}:{v:+.2f}" for k, v in sorted(agent_avg_steering[agent_id].items())
        )
        print(f"  {agent_id}: {traits_str}")

    return agent_avg_steering


def analyze_temporal_dynamics(actions, messages):
    """Analyze how activity changes over time."""
    actions_by_tick = defaultdict(int)
    messages_by_tick = defaultdict(int)
    tokens_by_tick = defaultdict(lambda: {"in": 0, "out": 0})

    for action in actions:
        actions_by_tick[action["tick"]] += 1

    for msg in messages:
        tick = msg["tick"]
        messages_by_tick[tick] += 1
        tokens_by_tick[tick]["in"] += msg["tokens_in"]
        tokens_by_tick[tick]["out"] += msg["tokens_out"]

    print("\n" + "=" * 80)
    print("TEMPORAL DYNAMICS")
    print("=" * 80)

    ticks = sorted(actions_by_tick.keys())
    if ticks:
        print(f"\nTick range: {min(ticks)} to {max(ticks)} ({len(ticks)} ticks)")
        print(f"Actions per tick: {len(actions) / len(ticks):.1f} avg")
        print(f"Messages per tick: {len(messages) / len(ticks):.1f} avg")

    return actions_by_tick, messages_by_tick, tokens_by_tick


def create_visualizations(
    dump_dir: Path, action_types, agent_avg_steering, actions_by_tick, messages_by_tick
):
    """Create visualization plots."""
    output_dir = dump_dir.parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True)

    sns.set_style("whitegrid")

    # 1. Action type distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    action_names = list(action_types.keys())
    action_counts = list(action_types.values())
    ax.bar(action_names, action_counts, color="steelblue")
    ax.set_xlabel("Action Type")
    ax.set_ylabel("Count")
    ax.set_title("Action Type Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "action_distribution.png", dpi=150)
    print(f"\n✓ Saved: {output_dir / 'action_distribution.png'}")
    plt.close()

    # 2. Steering heatmap
    if agent_avg_steering:
        fig, ax = plt.subplots(figsize=(10, 8))
        agents = sorted(agent_avg_steering.keys())[:20]  # Top 20 agents
        traits = ["E", "A", "C", "O", "N"]

        matrix = []
        for agent in agents:
            row = [agent_avg_steering[agent].get(trait, 0) for trait in traits]
            matrix.append(row)

        sns.heatmap(
            matrix,
            xticklabels=traits,
            yticklabels=agents,
            cmap="RdBu_r",
            center=0,
            vmin=-1.5,
            vmax=1.5,
            ax=ax,
        )
        ax.set_title("Agent Steering Profiles (First 20 Agents)")
        ax.set_xlabel("Trait")
        ax.set_ylabel("Agent ID")
        plt.tight_layout()
        plt.savefig(output_dir / "steering_heatmap.png", dpi=150)
        print(f"✓ Saved: {output_dir / 'steering_heatmap.png'}")
        plt.close()

    # 3. Activity over time
    if actions_by_tick and messages_by_tick:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ticks = sorted(actions_by_tick.keys())
        action_counts = [actions_by_tick[t] for t in ticks]
        message_counts = [messages_by_tick[t] for t in ticks]

        ax1.plot(ticks, action_counts, color="steelblue", linewidth=2)
        ax1.set_ylabel("Actions per Tick")
        ax1.set_title("Simulation Activity Over Time")
        ax1.grid(True, alpha=0.3)

        ax2.plot(ticks, message_counts, color="darkgreen", linewidth=2)
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Messages per Tick")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "activity_over_time.png", dpi=150)
        print(f"✓ Saved: {output_dir / 'activity_over_time.png'}")
        plt.close()

    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("storage/dumps/debug32"),
        help="Directory containing simulation dumps",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating plots"
    )
    parser.add_argument(
        "--metric-log",
        type=Path,
        help="Optional path to MetricTracker JSONL output (auto-detected from dump_dir if omitted)",
    )
    args = parser.parse_args()

    if not args.dump_dir.exists():
        print(f"ERROR: Dump directory not found: {args.dump_dir}")
        return

    # Load data
    actions, messages, graph_snapshots, metrics_snapshots = load_simulation_data(args.dump_dir)

    # Run analyses
    action_types, outcomes = analyze_action_distribution(actions)
    agent_actions, agent_messages = analyze_agent_activity(actions, messages)
    total_in, total_out = analyze_token_usage(messages)
    agent_avg_steering = analyze_steering_profiles(messages)
    actions_by_tick, messages_by_tick, tokens_by_tick = analyze_temporal_dynamics(
        actions, messages
    )
    if graph_snapshots or metrics_snapshots:
        print("\nAdditional datasets available:")
        print(f"  Graph snapshots:  {len(graph_snapshots)} records")
        print(f"  Metrics snapshots: {len(metrics_snapshots)} records")

    metric_log_path = args.metric_log or _auto_detect_metric_log(args.dump_dir)
    if metric_log_path:
        summary = load_metric_tracker_summary(metric_log_path)
        if summary:
            analyze_metric_tracker_summary(summary, metric_log_path)
        else:
            print(f"\n⚠ Warning: MetricTracker log not readable at {metric_log_path}")

    # Generate visualizations
    if not args.no_plots:
        try:
            create_visualizations(
                args.dump_dir,
                action_types,
                agent_avg_steering,
                actions_by_tick,
                messages_by_tick,
            )
        except Exception as e:
            print(f"\n⚠ Warning: Could not generate plots: {e}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
