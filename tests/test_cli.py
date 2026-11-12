"""Tests for the command line interface. No test here downloads or reads data."""

from pathlib import Path

import pandas as pd
import pytest

from blindside import cli
from blindside.data.download import DownloadError


def test_reproduce_modes_are_parsed():
    parser = cli.build_parser()
    args = parser.parse_args(["reproduce"])
    assert (args.fast, args.with_test, args.output_dir) == (False, False, None)
    args = parser.parse_args(["reproduce", "--fast", "--output-dir", "out"])
    assert args.fast and args.output_dir == Path("out")
    assert parser.parse_args(["reproduce", "--with-test"]).with_test


def test_fast_and_with_test_exclude_each_other(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args(["reproduce", "--fast", "--with-test"])
    assert error.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_main_dispatches_to_the_default_directories(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []
    monkeypatch.setattr(cli, "reproduce_fast", lambda out: calls.append(("fast", out)))
    monkeypatch.setattr(
        cli,
        "reproduce_full",
        lambda out, with_test=False, with_scene=False: calls.append(
            ("full", out, with_test, with_scene)
        ),
    )
    assert cli.main(["reproduce", "--fast"]) == 0
    assert cli.main(["reproduce"]) == 0
    assert cli.main(["reproduce", "--with-test", "--output-dir", "x"]) == 0
    assert cli.main(["reproduce", "--with-scene"]) == 0
    assert calls == [
        ("fast", cli.FAST_OUTPUT),
        ("full", cli.DEFAULT_OUTPUT, False, False),
        ("full", Path("x"), True, False),
        ("full", cli.DEFAULT_OUTPUT, False, True),
    ]
    # The scene needs full matches, and it is not one of the two README figures.
    with pytest.raises(SystemExit):
        cli.main(["reproduce", "--fast", "--with-scene"])
    assert cli.SCENE_FIGURE not in {cli.FIGURE_1, cli.FIGURE_2}
    # The fast path must never write into the directory of the README figures.
    assert cli.FAST_OUTPUT != cli.DEFAULT_OUTPUT
    assert {cli.FAST_FIGURE_1, cli.FAST_FIGURE_2}.isdisjoint({cli.FIGURE_1, cli.FIGURE_2})


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (DownloadError("Download of http://x failed after 5 attempts"), "run the command again"),
        (FileNotFoundError("data/x not found"), "data/x not found"),
        (RuntimeError("camera side of match 1 could not be inferred reliably"), "camera side"),
    ],
)
def test_failures_end_with_one_message_and_exit_code_1(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    expected: str,
):
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(cli, "reproduce_full", fail)
    assert cli.main(["reproduce"]) == 1
    captured = capsys.readouterr().err
    assert captured.startswith("error: ") and expected in captured
    assert "Traceback" not in captured


def test_printed_tables_carry_the_reported_numbers():
    agreement = pd.DataFrame(
        {
            "feature": ["p_blind_side_head", "p_blind_side_head"],
            "bound": ["lower", "upper"],
            "group": ["all", "all"],
            "n": [1875, 1875],
            "pose_blind": [0.205, 0.205],
            "xy_blind": [0.455, 0.455],
            "kappa": [0.28, 0.28],
            "kappa_low": [0.24, 0.24],
            "kappa_high": [0.31, 0.32],
        }
    )
    lines = cli._agreement_lines(agreement)
    assert len(lines) == 3
    assert "n = 1,875" in lines[2] and "kappa +0.28 (+0.24 to +0.31)" in lines[2]

    results = {
        "n": 1875,
        "events": 217,
        "primary_rows": pd.DataFrame(
            {
                "bound": ["lower", "upper"],
                "loss0": [0.3104, 0.3104],
                "loss1": [0.3106, 0.3106],
                "gain": [-0.0002, -0.0002],
                "low": [-0.0004, -0.0004],
                "high": [0.0, 0.0],
            }
        ),
    }
    text = "\n".join(cli._test_lines(results))
    assert "217 possession losses" in text
    assert "-0.0002 (-0.0004 to +0.0000)" in text
    assert "No detectable increment" in text
