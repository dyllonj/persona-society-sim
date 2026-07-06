"""Validate judge parsing against static human-rated examples."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from steering.llm_judge import (
    DEFAULT_TRAITS,
    StaticJudgeClient,
    parse_likert_scores,
    score_text_with_judge,
)


@dataclass(frozen=True)
class HumanRatedExample:
    example_id: str
    generated_text: str
    human_scores: Mapping[str, int]
    judge_output: str


@dataclass(frozen=True)
class ExampleAgreement:
    example_id: str
    exact_traits: int
    total_traits: int
    absolute_error: float
    parsed_scores: Mapping[str, int]


@dataclass(frozen=True)
class JudgeValidationReport:
    examples: Sequence[ExampleAgreement]
    exact_trait_rate: float
    mean_absolute_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "examples": [asdict(example) for example in self.examples],
            "exact_trait_rate": self.exact_trait_rate,
            "mean_absolute_error": self.mean_absolute_error,
        }


DEFAULT_EXAMPLES: tuple[HumanRatedExample, ...] = (
    HumanRatedExample(
        example_id="sociable_helper",
        generated_text=(
            "I will bring people together, make sure the newcomer feels welcome, "
            "and keep the plan moving."
        ),
        human_scores={"E": 5, "A": 4, "C": 4, "O": 3, "N": 2},
        judge_output='{"E": 5, "A": 4, "C": 4, "O": 3, "N": 2}',
    ),
    HumanRatedExample(
        example_id="anxious_critic",
        generated_text=(
            "This could go wrong in several ways, and I do not trust the group "
            "to notice the details."
        ),
        human_scores={"E": 2, "A": 2, "C": 4, "O": 3, "N": 5},
        judge_output=(
            "Extraversion: 2\nAgreeableness: 2\nConscientiousness: 4\n"
            "Openness: 3\nNeuroticism: 5"
        ),
    ),
)


def evaluate_examples(
    examples: Sequence[HumanRatedExample] = DEFAULT_EXAMPLES,
    *,
    tolerance: int = 0,
) -> JudgeValidationReport:
    """Parse static judge outputs and compare them to human-rated scores."""

    agreements: list[ExampleAgreement] = []
    total_abs = 0.0
    total_traits = 0
    exact_traits = 0
    for example in examples:
        parsed = parse_likert_scores(example.judge_output, DEFAULT_TRAITS)
        errors = [
            abs(parsed[trait] - int(example.human_scores[trait]))
            for trait in DEFAULT_TRAITS
        ]
        exact = sum(error <= tolerance for error in errors)
        total = len(errors)
        agreements.append(
            ExampleAgreement(
                example_id=example.example_id,
                exact_traits=exact,
                total_traits=total,
                absolute_error=round(sum(errors) / total, 3),
                parsed_scores=parsed,
            )
        )
        total_abs += sum(errors)
        total_traits += total
        exact_traits += exact
    return JudgeValidationReport(
        examples=agreements,
        exact_trait_rate=round(exact_traits / total_traits, 3) if total_traits else 0.0,
        mean_absolute_error=round(total_abs / total_traits, 3) if total_traits else 0.0,
    )


def run_static_client_smoke() -> Mapping[str, int]:
    """Exercise the mockable client path without making a model call."""

    client = StaticJudgeClient([DEFAULT_EXAMPLES[0].judge_output])
    result = score_text_with_judge(
        client,
        DEFAULT_EXAMPLES[0].generated_text,
        judge_model="gpt-4o-mini",
        target_model="meta-llama/Llama-3.1-8B-Instruct",
        traits=DEFAULT_TRAITS,
    )
    return result.scores


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate static human-rated examples against judge helpers"
    )
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--client-smoke",
        action="store_true",
        help="Also exercise StaticJudgeClient through score_text_with_judge",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = evaluate_examples(tolerance=args.tolerance)
    if args.client_smoke:
        run_static_client_smoke()
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Judge validation: "
            f"exact_trait_rate={report.exact_trait_rate}, "
            f"mean_absolute_error={report.mean_absolute_error}"
        )
        for example in report.examples:
            print(
                f"{example.example_id}: exact={example.exact_traits}/"
                f"{example.total_traits}, mae={example.absolute_error}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
