"""The README is the judged document: its figures, links and code example are tested."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

README = (ROOT / "README.md").read_text("utf-8")


def test_license_file_has_the_required_name():
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
