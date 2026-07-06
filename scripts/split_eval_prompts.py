"""Split persona contrast prompts into train/eval JSONL files.

The splitter is intentionally deterministic and validates disjointness by both
prompt id and full prompt fingerprint.  This keeps held-out prompt files from
quietly sharing stems or duplicated records with training data.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from data.prompts.schema import PromptItem, load_prompt_items


DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_SEED = 1337


@dataclass(frozen=True)
class SplitVerification:
    """Overlap checks for two prompt splits."""

    train_count: int
    eval_count: int
    overlapping_ids: tuple[str, ...]
    overlapping_fingerprints: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.overlapping_ids and not self.overlapping_fingerprints


@dataclass(frozen=True)
class SplitResult:
    """Summary returned after writing a train/eval prompt split."""

    trait: str
    source_path: Path
    train_path: Path
    eval_path: Path
    train_count: int
    eval_count: int
    verification: SplitVerification


def prompt_fingerprint(record: PromptItem | Mapping[str, object]) -> str:
    """Return a stable content fingerprint for overlap detection."""

    if isinstance(record, PromptItem):
        payload = record.to_dict()
    else:
        payload = dict(record)
    selected = {
        "question_text": str(payload.get("question_text", "")).strip(),
        "option_a": str(payload.get("option_a", "")).strip(),
        "option_b": str(payload.get("option_b", "")).strip(),
        "option_a_is_high": bool(payload.get("option_a_is_high")),
        "option_b_is_high": bool(payload.get("option_b_is_high")),
    }
    return json.dumps(selected, sort_keys=True, ensure_ascii=False)


def _record_id(record: PromptItem | Mapping[str, object]) -> str:
    if isinstance(record, PromptItem):
        return record.id
    return str(record.get("id", ""))


def verify_disjoint(
    train_records: Sequence[PromptItem | Mapping[str, object]],
    eval_records: Sequence[PromptItem | Mapping[str, object]],
) -> SplitVerification:
    """Verify that two prompt collections share no ids or full fingerprints."""

    train_ids = {_record_id(record) for record in train_records}
    eval_ids = {_record_id(record) for record in eval_records}
    train_fingerprints = {prompt_fingerprint(record) for record in train_records}
    eval_fingerprints = {prompt_fingerprint(record) for record in eval_records}
    return SplitVerification(
        train_count=len(train_records),
        eval_count=len(eval_records),
        overlapping_ids=tuple(sorted(train_ids & eval_ids)),
        overlapping_fingerprints=tuple(sorted(train_fingerprints & eval_fingerprints)),
    )


def split_prompt_items(
    records: Sequence[PromptItem],
    *,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    eval_count: int | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[list[PromptItem], list[PromptItem]]:
    """Split prompt items into train/eval groups.

    Records sharing an ``id`` are kept in the same split.  This is stricter
    than row-level sampling and prevents repeated stems from leaking into the
    held-out set.
    """

    if not records:
        raise ValueError("Cannot split an empty prompt collection")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")

    groups: dict[str, list[PromptItem]] = {}
    for record in records:
        groups.setdefault(record.id, []).append(record)

    group_keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    group_count = len(group_keys)
    if group_count < 2:
        raise ValueError("Need at least two distinct prompt ids to create a split")

    if eval_count is None:
        eval_group_count = max(1, round(group_count * eval_fraction))
    else:
        eval_group_count = eval_count
    if eval_group_count < 1:
        raise ValueError("eval_count must select at least one prompt id")
    if eval_group_count >= group_count:
        raise ValueError("eval split would leave no training prompt ids")

    eval_keys = set(group_keys[:eval_group_count])
    train_records: list[PromptItem] = []
    eval_records: list[PromptItem] = []
    for key in sorted(groups):
        target = eval_records if key in eval_keys else train_records
        target.extend(groups[key])

    verification = verify_disjoint(train_records, eval_records)
    if not verification.ok:
        raise ValueError(
            "Prompt split is not disjoint: "
            f"ids={verification.overlapping_ids}, "
            f"fingerprints={verification.overlapping_fingerprints}"
        )
    return train_records, eval_records


def write_prompt_items(path: Path | str, records: Sequence[PromptItem]) -> None:
    """Write prompt items as JSONL in the canonical A/B schema."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def split_prompt_file(
    source_path: Path | str,
    *,
    train_path: Path | str | None = None,
    eval_path: Path | str | None = None,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    eval_count: int | None = None,
    seed: int = DEFAULT_SEED,
    train_suffix: str = "_train",
    eval_suffix: str = "_eval",
    overwrite: bool = False,
) -> SplitResult:
    """Load one prompt JSONL file, split it, write outputs, and verify overlap."""

    source = Path(source_path)
    trait = source.stem
    destination_dir = source.parent
    train_dest = Path(train_path) if train_path else destination_dir / f"{trait}{train_suffix}.jsonl"
    eval_dest = Path(eval_path) if eval_path else destination_dir / f"{trait}{eval_suffix}.jsonl"

    if train_dest == source or eval_dest == source:
        raise ValueError("Split outputs must not overwrite the source prompt file")
    existing = [path for path in (train_dest, eval_dest) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing split file(s): {names}")

    records = load_prompt_items(source)
    train_records, eval_records = split_prompt_items(
        records,
        eval_fraction=eval_fraction,
        eval_count=eval_count,
        seed=seed,
    )
    verification = verify_disjoint(train_records, eval_records)
    if not verification.ok:
        raise ValueError(f"Split verification failed for {source}")

    write_prompt_items(train_dest, train_records)
    write_prompt_items(eval_dest, eval_records)
    return SplitResult(
        trait=trait,
        source_path=source,
        train_path=train_dest,
        eval_path=eval_dest,
        train_count=len(train_records),
        eval_count=len(eval_records),
        verification=verification,
    )


def _discover_traits(prompt_dir: Path) -> list[str]:
    traits = []
    for path in sorted(prompt_dir.glob("*.jsonl")):
        if path.stem.endswith(("_train", "_eval")):
            continue
        traits.append(path.stem)
    return traits


def _print_result(result: SplitResult) -> None:
    print(
        f"{result.trait}: train={result.train_count} -> {result.train_path}; "
        f"eval={result.eval_count} -> {result.eval_path}"
    )


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split data/prompts/{trait}.jsonl into train/eval JSONL files"
    )
    parser.add_argument("traits", nargs="*", help="Trait file stems to split")
    parser.add_argument("--prompt-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--eval-fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--eval-count", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-suffix", default="_train")
    parser.add_argument("--eval-suffix", default="_eval")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report planned split sizes without writing files",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    traits = list(args.traits) or _discover_traits(args.prompt_dir)
    if not traits:
        raise FileNotFoundError(f"No prompt JSONL files found in {args.prompt_dir}")

    for trait in traits:
        source = args.prompt_dir / f"{trait}.jsonl"
        records = load_prompt_items(source)
        train_records, eval_records = split_prompt_items(
            records,
            eval_fraction=args.eval_fraction,
            eval_count=args.eval_count,
            seed=args.seed,
        )
        verification = verify_disjoint(train_records, eval_records)
        if not verification.ok:
            raise ValueError(f"Split verification failed for {source}")
        if args.dry_run:
            print(f"{trait}: train={len(train_records)}; eval={len(eval_records)}")
            continue
        result = split_prompt_file(
            source,
            eval_fraction=args.eval_fraction,
            eval_count=args.eval_count,
            seed=args.seed,
            train_suffix=args.train_suffix,
            eval_suffix=args.eval_suffix,
            overwrite=args.overwrite,
        )
        _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
