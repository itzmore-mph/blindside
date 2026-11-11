"""The README is the judged document: its limits and its code example are tested."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_readme  # noqa: E402

README = (ROOT / "README.md").read_text("utf-8")


def test_readme_respects_the_submission_limits():
    assert check_readme.check(README) == []
    assert check_readme.count_figures(README) == 2
    assert (ROOT / "LICENSE.md").is_file()


def test_readme_figures_exist():
    targets = re.findall(r"!\[[^\]]*\]\(([^)]*)\)", README)
    assert len(targets) == 2
    for target in targets:
        assert (ROOT / target).is_file(), target


def test_readme_links_point_to_existing_files():
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]*)\)", README):
        if not target.startswith("http"):
            assert (ROOT / target).is_file(), target


def test_readme_example_runs_and_prints_what_it_claims(capsys: pytest.CaptureFixture[str]):
    blocks = re.findall(r"```python\n(.*?)```", README, flags=re.DOTALL)
    assert len(blocks) == 1
    exec(compile(blocks[0], "README.md", "exec"), {})
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["p_blind_side, lower bound: 1.00", "p_blind_side, upper bound: 0.73"]


def test_limits_are_enforced():
    assert check_readme.check("word " * 1000) == []
    assert "1001 words" in check_readme.check("word " * 1001)[0]
    image, table = "![a](a.png)\n", "| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert check_readme.check(image * 2) == []
    assert check_readme.check(image + table) == []
    assert "limit is 2 in total" in check_readme.check(image * 2 + table)[0]
    assert "limit is 2 in total" in check_readme.check(image * 2 + '<img src="b.png">')[0]
    # Images and tables inside a code block are examples, not figures.
    assert check_readme.check(image * 2 + "```\n" + image + table + "```\n") == []
    assert check_readme.check("a — b") == ["em dash or double hyphen in the text"]
    assert check_readme.check("a -- b") == ["em dash or double hyphen in the text"]
    assert check_readme.check("```\nuv run x --fast\n```\n`--fast` <!-- note -->") == []
