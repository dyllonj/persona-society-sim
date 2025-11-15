"""Utilities to sanitize LLM outputs before logging or acting.

This keeps agent messages free of meta-instructions, UI artifacts,
and accidental prompt leakage while preserving intent.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable


logger = logging.getLogger(__name__)


_META_PATTERNS: Iterable[re.Pattern[str]] = [
    # Common prompt/meta leakage
    re.compile(r"(?i)\bplease write the next response\b.*"),
    re.compile(r"(?i)\bend of (?:response|mission)\b.*"),
    re.compile(r"(?i)\bmission (?:complete|accomplished)\b.*"),
    re.compile(r"(?i)\bdelete file:.*"),
    re.compile(r"(?i)\bshut down:.*"),
    re.compile(r"(?i)\b\(edited to follow rules\).*"),
    re.compile(r"(?i)\bnote:\s*.*"),
    # UI affordances and scaffolds
    re.compile(r"(?i)\bshow (?:more|less)\b.*"),
    re.compile(r"(?i)^next:\s*\(.*\).*$"),
    re.compile(r"(?i)^here'?s a sample response:.*$"),
    re.compile(r"(?i)^agent-?\d{3}'?s response:.*$"),
]


def _strip_quotes(text: str) -> str:
    # Remove leading/trailing ASCII or smart quotes
    return text.strip().strip('"').strip("'").strip("“").strip("”").strip()


def _remove_hashtag_noise(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        # Drop lines that are mostly hashtags/labels
        if line.count("#") >= 3:
            continue
        cleaned.append(line)
    return cleaned


def _remove_separators(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        # Drop long dash or dot separators
        if re.fullmatch(r"[-•\s]{5,}", line) or re.fullmatch(r"[.\s]{5,}", line):
            continue
        cleaned.append(line)
    return cleaned


def _dedupe_consecutive(lines: list[str]) -> list[str]:
    out: list[str] = []
    last: str | None = None
    for line in lines:
        if line == last:
            continue
        out.append(line)
        last = line
    return out


_REPEATED_SENTENCE_PATTERN = re.compile(
    r"(?P<sentence>[^.!?]+[.!?])(?P<tail>(?:\s+(?P=sentence)){2,})",
    re.MULTILINE,
)


def _collapse_repeated_phrases(text: str) -> tuple[str, bool]:
    """Collapse identical sentences repeated more than twice into two occurrences."""

    triggered = False

    def _repl(match: re.Match[str]) -> str:
        nonlocal triggered
        triggered = True
        sentence = match.group("sentence")
        tail = match.group("tail")
        sep_match = re.match(r"\s+", tail)
        sep = sep_match.group(0) if sep_match else " "
        return f"{sentence}{sep}{sentence}"

    collapsed = _REPEATED_SENTENCE_PATTERN.sub(_repl, text)
    return collapsed, triggered


def sanitize_agent_output(text: str) -> str:
    """Clean up LLM output for in-sim use and display.

    - Removes obvious meta-instructions and UI scaffolding
    - Strips enclosing quotes
    - Drops hashtag spam and separator lines
    - Deduplicates consecutive duplicate lines
    - Collapses excessive whitespace
    """
    if not text:
        return text

    # Fast path: strip and remove outer quotes
    text = _strip_quotes(text)

    # Remove meta lines/patterns
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    keep: list[str] = []
    for ln in lines:
        if any(p.search(ln) for p in _META_PATTERNS):
            continue
        keep.append(ln)

    # Remove hashtag noise and separators, then dedupe
    keep = _remove_hashtag_noise(keep)
    keep = _remove_separators(keep)
    keep = _dedupe_consecutive(keep)

    # Rejoin and collapse excessive internal spaces
    cleaned = "\n".join(keep)
    cleaned = re.sub(r"\s{3,}", "  ", cleaned).strip()

    cleaned, collapsed = _collapse_repeated_phrases(cleaned)
    if collapsed:
        logger.info("Collapsed repeated sentences in agent output for persona review")

    return cleaned

