"""Selection rule and drawing of the tactical case study. Synthetic engagements only."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Wedge

from blindside.analysis.case_study import AGREE_BLIND, DISAGREE, case_study_candidates
from blindside.viz.case_study import plot_tactical_case_study


def _engagements() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Seven engagements. Rows 0 and 1 fit panel A, rows 4 and 5 panel B, the rest fail one rule."""
    test = pd.DataFrame(
        {
            "carrier_speed": [1.5, 1.2, 0.4, 1.5, 4.5, 6.0, 4.5],
            "approach_angle_heading_deg": [150.0, 170.0, 170.0, 60.0, 160.0, 170.0, 170.0],
            "interplayer_distance_start": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0],
            "carrier_head_flip": False,
            "carrier_torso_flip": False,
        }
    )
    p = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    off_head = np.array([10.0, 20.0, 5.0, 5.0, 170.0, 150.0, 175.0])
    feature = pd.DataFrame(
        {"p_blind_side_head": p, "p_blind_side_torso": p, "presser_off_head_deg": off_head}
    )
    return test, {"lower": feature, "upper": feature.copy()}


def test_candidates_follow_the_rule_and_are_sorted_by_clarity():
    test, features = _engagements()
    candidates = case_study_candidates(test, features)
    # Row 2 is too slow, row 3 is not blind for the proxy.
    assert candidates[DISAGREE].index.tolist() == [1, 0]
    assert candidates[DISAGREE]["score"].tolist() == [150.0, 140.0]
    # Row 6 is too far away. Row 4 scores 330, row 5 scores 320.
    assert candidates[AGREE_BLIND].index.tolist() == [4, 5]


def test_candidates_need_both_bounds_both_segments_and_no_flip():
    test, features = _engagements()
    features["upper"].loc[1, "p_blind_side_torso"] = 0.3
    features["lower"].loc[4, "p_blind_side_head"] = 0.85
    test.loc[5, "carrier_head_flip"] = True
    candidates = case_study_candidates(test, features)
    assert candidates[DISAGREE].index.tolist() == [0]
    assert candidates[AGREE_BLIND].empty


def _panels() -> pd.DataFrame:
    row = {
        "carrier_x": 10.0, "carrier_y": -5.0, "presser_x": 13.0, "presser_y": -1.0,
        "travel_dx": -0.5, "travel_dy": -0.5, "carrier_speed": 1.4,
        "approach_angle_heading_deg": 172.0, "head_deg": 50.0, "head_sd_lower_deg": 2.0,
        "head_sd_upper_deg": 12.0, "p_head_lower": 0.0, "p_head_upper": 0.02,
        "title": "A  Slow carrier", "source": "synthetic",
    }  # fmt: skip
    return pd.DataFrame([row, {**row, "p_head_lower": 1.0, "p_head_upper": 1.0, "head_deg": 230.0}])


def test_figure_has_one_pitch_panel_per_engagement_with_both_verdicts():
    others = pd.DataFrame({"x": [8.0, 15.0], "y": [0.0, -8.0], "pressing_team": [True, False]})
    figure = plot_tactical_case_study(_panels(), [others, None], [(105.0, 68.0)] * 2)
    assert len(figure.axes) == 2
    texts = ["\n".join(t.get_text() for t in axis.texts) for axis in figure.axes]
    assert "XY proxy: BLIND\nHead pose: IN VIEW" in texts[0]
    assert "XY proxy: BLIND\nHead pose: BLIND" in texts[1]
    assert "0.00 to 0.02 across both bounds" in texts[0]
    for axis in figure.axes:
        # Visual field, two bands at each of its two edges, and the dashed field of the proxy.
        assert sum(isinstance(patch, Wedge) for patch in axis.patches) == 6
        low, high = axis.get_xlim()
        assert high - low < 30.0  # zoomed on the engagement, not the full pitch
    # Large type for a screen share.
    assert min(t.get_fontsize() for axis in figure.axes for t in axis.texts if t.get_text()) >= 13


def test_a_standing_carrier_gets_no_direction_of_travel():
    panels = _panels().iloc[[0]].assign(travel_dx=0.0, travel_dy=0.0)
    figure = plot_tactical_case_study(panels, [None], [(105.0, 68.0)])
    assert sum(isinstance(patch, Wedge) for patch in figure.axes[0].patches) == 5


def test_inputs_must_match_the_panels():
    with pytest.raises(ValueError, match="one entry per panel"):
        plot_tactical_case_study(_panels(), [None], [(105.0, 68.0)] * 2)
    with pytest.raises(ValueError, match="one entry per panel"):
        plot_tactical_case_study(_panels().iloc[:0], [], [])
