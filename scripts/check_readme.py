"""Check README.md against the limits of the Analytics Cup rules.

The rules allow at most 1000 words and at most two figures or tables in total. The count
here is conservative: every whitespace-separated token counts as a word, including code
blocks, link targets and markup. Em dashes and double hyphens outside code are rejected too
(project convention).

Run from the repository root: ``uv run python scripts/check_readme.py``. Exit code 1 on a
violation. Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAX_WORDS = 1000
MAX_FIGURES_AND_TABLES = 2

_FENCE = re.compile(r"^(```|~~~)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b", re.IGNORECASE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HTML_TABLE = re.compile(r"<table\b", re.IGNORECASE)


def count_words(text: str) -> int:
    """Number of whitespace-separated tokens, code blocks and markup included."""
    return len(text.split())


def _prose_lines(text: str) -> list[str]:
    """Lines outside fenced code blocks."""
    lines, fenced = [], False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return lines


def count_figures(text: str) -> int:
    """Markdown and HTML images outside code blocks."""
    return sum(len(_IMAGE.findall(line)) for line in _prose_lines(text))


def count_tables(text: str) -> int:
    """Markdown tables (a header rule that contains a pipe) and HTML tables outside code blocks."""
    lines = _prose_lines(text)
    pipes = sum(1 for line in lines if "|" in line and _TABLE_RULE.match(line))
    return pipes + sum(len(_HTML_TABLE.findall(line)) for line in lines)


def check(text: str) -> list[str]:
    """Violations of the limits as readable messages. Empty if the text passes."""
    problems = []
    words = count_words(text)
    if words > MAX_WORDS:
        problems.append(f"{words} words, the limit is {MAX_WORDS}")
    figures, tables = count_figures(text), count_tables(text)
    if figures + tables > MAX_FIGURES_AND_TABLES:
        problems.append(
            f"{figures} figures and {tables} tables, the limit is {MAX_FIGURES_AND_TABLES} in total"
        )
    prose = re.sub(r"`[^`\n]*`", "", "\n".join(_prose_lines(text)))  # command flags are code
    if "—" in text or re.search(r"(?<![-<!])--(?![->])", prose):
        problems.append("em dash or double hyphen in the text")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", nargs="?", type=Path, default=Path("README.md"))
    args = parser.parse_args(argv)
    text = args.path.read_text("utf-8")
    problems = check(text)
    print(
        f"{args.path}: {count_words(text)} of {MAX_WORDS} words, "
        f"{count_figures(text)} figures, {count_tables(text)} tables"
    )
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
