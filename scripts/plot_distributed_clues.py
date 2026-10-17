"""Render dependency-free SVG figures for completed distributed-clue batches.

Figures follow Tufte defaults: gray data with one accent, direct labels instead
of legends, range-frame axes without gridlines, serif text, an off-white
background with a dark-mode variant, and titles that state the finding. Each
dot is one task instance (environment seed), averaged over replicates and
residents, because instances, not messages, are the independent unit.

  python scripts/plot_distributed_clues.py effect  BATCH... --output docs/assets/fig-communication-effect.svg
  python scripts/plot_distributed_clues.py guesses BATCH... --output docs/assets/fig-partial-information-guesses.svg
  python scripts/plot_distributed_clues.py conditions --row "Label=BATCH:condition" ... --output fig.svg --title "..."
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_distributed_clues import (
    bootstrap_ci, episode_accuracy, guess_diagnostics, load_batches, per_seed_means, plurality_vote,
)

WIDTH, LEFT, RIGHT = 900, 250, 60
STYLE = """
  .bg { fill: #fffff8; }
  .t { font-family: "ET Book", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif; fill: #111; }
  .title { font-size: 21px; }
  .sub { font-size: 14px; fill: #666; }
  .label { font-size: 15px; }
  .value { font-size: 14px; fill: #111; paint-order: stroke; stroke: #fffff8; stroke-width: 5px; stroke-linejoin: round; }
  .note { font-size: 13px; fill: #666; }
  .tick { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 11px; fill: #777; }
  .axis { stroke: #bbb; stroke-width: 1; }
  .ref { stroke: #999; stroke-width: 1; stroke-dasharray: 4 4; }
  .dot { fill: #666; fill-opacity: 0.55; }
  .dot-hi { fill: #c0392b; fill-opacity: 0.6; }
  .mean { stroke: #111; stroke-width: 2.5; }
  .mean-hi { stroke: #c0392b; stroke-width: 2.5; }
  .ci { stroke: #111; stroke-width: 1.2; }
  .bar { fill: #7a7a7a; }
  .bar-hi { fill: #c0392b; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #151515; }
    .t, .value { fill: #ddd; }
    .value { stroke: #151515; }
    .sub, .note { fill: #999; }
    .tick { fill: #888; }
    .axis { stroke: #555; }
    .ref { stroke: #777; }
    .dot { fill: #999; }
    .dot-hi { fill: #fc8d62; }
    .mean, .ci { stroke: #ddd; }
    .mean-hi { stroke: #fc8d62; }
    .bar { fill: #888; }
    .bar-hi { fill: #fc8d62; }
  }
"""


def pct(value):
    return f"{100 * value:.0f}%"


def svg(height, title, subtitle, body, description):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">\n'
            f'<title id="title">{escape(title)}</title>\n<desc id="desc">{escape(description)}</desc>\n'
            f'<style>{STYLE}</style>\n<rect class="bg" width="{WIDTH}" height="{height}"/>\n'
            f'<text class="t title" x="24" y="38">{escape(title)}</text>\n'
            f'<text class="t sub" x="24" y="62">{escape(subtitle)}</text>\n' + "\n".join(body) + "\n</svg>\n")


def x_of(value):
    return LEFT + value * (WIDTH - LEFT - RIGHT)


def strip_rows(rows, top, row_height, reference=None):
    """One row per group: instance dots, mean bar, 95% instance-bootstrap CI, direct value label."""
    body = []
    for index, (label, values, highlight) in enumerate(rows):
        y = top + index * row_height + row_height / 2
        mean, ci = statistics.fmean(values), bootstrap_ci(values)
        body.append(f'<text class="t label" x="{LEFT - 16}" y="{y + 5:.1f}" text-anchor="end">{escape(label)}</text>')
        if ci:
            body.append(f'<line class="ci" x1="{x_of(ci[0]):.1f}" x2="{x_of(ci[1]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        # Tied instances share one dot with a count, so rows never collide.
        ties = defaultdict(int)
        for value in values:
            ties[round(value, 3)] += 1
        for value, count in sorted(ties.items()):
            body.append(f'<circle class="{"dot-hi" if highlight else "dot"}" cx="{x_of(value):.1f}" cy="{y:.1f}" r="{3.2 + min(count, 16) ** 0.5:.1f}"/>')
            if count > 1:
                body.append(f'<text class="tick" x="{x_of(value):.1f}" y="{y + 24:.1f}" text-anchor="middle">×{count}</text>')
        body.append(f'<line class="{"mean-hi" if highlight else "mean"}" x1="{x_of(mean):.1f}" x2="{x_of(mean):.1f}" y1="{y - 14:.1f}" y2="{y + 14:.1f}"/>')
        anchor, dx = ("end", -10) if mean > 0.85 else ("start", 10)
        span = f" (CI {pct(ci[0])}–{pct(ci[1])})" if ci and ci[0] != ci[1] else ""
        body.append(f'<text class="t value" x="{x_of(mean) + dx:.1f}" y="{y - 18:.1f}" text-anchor="{anchor}">{pct(mean)}{span}</text>')
    bottom = top + len(rows) * row_height
    if reference is not None:
        value, text = reference
        body.append(f'<line class="ref" x1="{x_of(value):.1f}" x2="{x_of(value):.1f}" y1="{top:.1f}" y2="{bottom:.1f}"/>')
        body.append(f'<text class="t note" x="{x_of(value) + 6:.1f}" y="{bottom - 2:.1f}">{escape(text)}</text>')
    body.append(f'<line class="axis" x1="{x_of(0):.1f}" x2="{x_of(1):.1f}" y1="{bottom + 8:.1f}" y2="{bottom + 8:.1f}"/>')
    for tick in (0, 0.25, 0.5, 0.75, 1):
        body.append(f'<text class="tick" x="{x_of(tick):.1f}" y="{bottom + 26:.1f}" text-anchor="middle">{pct(tick)}</text>')
    body.append(f'<text class="tick" x="{x_of(0.5):.1f}" y="{bottom + 44:.1f}" text-anchor="middle">answer accuracy per task instance</text>')
    return body, bottom + 44


def effect(batches, output):
    rows, _ = load_batches(batches)
    on = [r for r in rows if r["condition"] == "communication"]
    off = [r for r in rows if r["condition"] == "no_communication"]
    on_seed, off_seed = per_seed_means(on, episode_accuracy), per_seed_means(off, episode_accuracy)
    vote_seed = per_seed_means(off, lambda r: plurality_vote(r)[0])
    chance = guess_diagnostics(batches)["uniform_guess_expected_accuracy"]
    groups = [("Isolated agents", list(off_seed.values()), False),
              ("Majority vote of isolated agents", list(vote_seed.values()), False),
              ("Agents who can share evidence", list(on_seed.values()), True)]
    n = len(on_seed)
    title = f"Sharing evidence lifts accuracy from {pct(statistics.fmean(off_seed.values()))} to {pct(statistics.fmean(on_seed.values()))}"
    subtitle = f"{n} fresh task instances, Qwen3-8B 4-bit, 3 residents each; dot = one instance, bar = mean, line = 95% CI"
    body, bottom = strip_rows(groups, 96, 92, (chance, f"uniform guess {pct(chance)}"))
    body.append(f'<text class="t note" x="24" y="{bottom + 30:.1f}">Isolated agents fall below the uniform-guess reference because they choose early-listed candidates (next figure).</text>')
    body.append(f'<text class="t note" x="24" y="{bottom + 48:.1f}">Voting cannot recover information nobody holds. Circle area grows with the number of tied instances.</text>')
    desc = (f"Strip plot over {n} task instances. Mean answer accuracy: isolated agents {pct(statistics.fmean(off_seed.values()))}, "
            f"majority vote {pct(statistics.fmean(vote_seed.values()))}, communicating agents {pct(statistics.fmean(on_seed.values()))}. "
            f"Uniform-guess reference {pct(chance)}.")
    output.write_text(svg(bottom + 68, title, subtitle, body, desc))


def guesses(batches, output):
    diag = guess_diagnostics(batches)
    counts, total = diag["candidate_position_counts"], diag["partial_information_submissions"]
    consistent = diag["consistent_with_held_clues"]
    labels = ["1st listed candidate", "2nd", "3rd", "4th"]
    top, bar_h, gap = 100, 34, 18
    body = []
    scale = lambda share: LEFT + share * (WIDTH - LEFT - RIGHT) / 0.8
    for i, label in enumerate(labels):
        share = counts.get(i, 0) / total
        y = top + i * (bar_h + gap)
        body.append(f'<text class="t label" x="{LEFT - 16}" y="{y + bar_h / 2 + 5:.1f}" text-anchor="end">{label}</text>')
        if share:
            body.append(f'<rect class="{"bar-hi" if i == 0 else "bar"}" x="{LEFT}" y="{y}" width="{scale(share) - LEFT:.1f}" height="{bar_h}"/>')
        body.append(f'<text class="t value" x="{scale(share) + 8:.1f}" y="{y + bar_h / 2 + 5:.1f}">{pct(share)}</text>')
    bottom = top + 4 * (bar_h + gap) - gap
    body.append(f'<line class="ref" x1="{scale(0.25):.1f}" x2="{scale(0.25):.1f}" y1="{top - 8}" y2="{bottom + 4}"/>')
    body.append(f'<text class="t note" x="{scale(0.25) + 6:.1f}" y="{top - 14}">uniform guess among 4 candidates: 25% each</text>')
    title = f"Without the missing clues, agents take the first listed candidate {pct(counts.get(0, 0) / total)} of the time"
    subtitle = (f"{total} final answers given with partial information; {pct(consistent)} satisfied every clue the agent held")
    body.append(f'<text class="t note" x="24" y="{bottom + 40:.1f}">Plans are listed in ID order. Position bias, not rule-breaking, explains below-chance accuracy without communication.</text>')
    desc = (f"Horizontal bars: share of partial-information answers by candidate position. "
            + ", ".join(f"{labels[i]} {pct(counts.get(i, 0) / total)}" for i in range(4)) + ". Uniform reference 25%.")
    output.write_text(svg(bottom + 60, title, subtitle, body, desc))


def conditions(specs, output, title, subtitle):
    groups = []
    for spec in specs:
        label, _, source = spec.partition("=")
        batch, _, condition = source.rpartition(":")
        rows, _ = load_batches([Path(batch)])
        chosen = [r for r in rows if r["condition"] == condition]
        groups.append((label, list(per_seed_means(chosen, episode_accuracy).values()), False))
    worst = min(range(len(groups)), key=lambda i: statistics.fmean(groups[i][1]))
    groups = [(l, v, i == worst) for i, (l, v, _) in enumerate(groups)]
    body, bottom = strip_rows(groups, 96, 84)
    desc = "; ".join(f"{l}: mean {pct(statistics.fmean(v))} over {len(v)} instances" for l, v, _ in groups)
    output.write_text(svg(bottom + 24, title, subtitle, body, desc))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=("effect", "guesses", "conditions"))
    parser.add_argument("batches", nargs="*", type=Path)
    parser.add_argument("--row", action="append", default=[], help='"Label=BATCH:condition" (conditions figure)')
    parser.add_argument("--title", default="")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.figure == "effect":
        effect(args.batches, args.output)
    elif args.figure == "guesses":
        guesses(args.batches, args.output)
    else:
        conditions(args.row, args.output, args.title, args.subtitle)
    print(args.output)


if __name__ == "__main__":
    main()
