"""The walkthrough notebook is committed with its outputs, so that it reads in the browser."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "examples" / "walkthrough.ipynb").read_text("utf-8"))
CODE_CELLS = [cell for cell in NOTEBOOK["cells"] if cell["cell_type"] == "code"]


def test_every_code_cell_was_executed_without_an_error():
    assert CODE_CELLS
    for cell in CODE_CELLS:
        assert cell["execution_count"] is not None
        assert all(output["output_type"] != "error" for output in cell["outputs"])


def test_the_three_figures_are_stored_as_images():
    images = [
        output
        for cell in CODE_CELLS
        for output in cell["outputs"]
        if "image/png" in output.get("data", {})
    ]
    assert len(images) == 3


def test_outputs_hold_no_local_paths_and_linked_files_exist():
    text = json.dumps(NOTEBOOK)
    assert "Users" not in text and "/home/" not in text
    for target in ("docs/figures/reliability_composite.png", "docs/figures/proxy_agreement.png",
                   "docs/figures/match_scene_cone.png", "docs/TACTICAL_INSIGHTS.md"):  # fmt: skip
        assert f"../{target}" in text
        assert (ROOT / target).is_file()
