"""The XY proxy and the speed filter that says when it can stand in for body pose."""

import numpy as np
import pandas as pd
import pytest

import blindside
from blindside.analysis.consistency import SPEED_EDGES
from blindside.features.pose_features import FOV_HALF_ANGLE_DEG
from blindside.features.xy_proxy import (
    XY_PROXY_ANGLE_DEG,
    XY_RELIABILITY_CLASSES,
    XY_RELIABLE_FROM_MPS,
    XY_UNRELIABLE_BELOW_MPS,
    classify_xy_reliability,
    xy_blind_side_proxy,
)


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (0.0, "unreliable"),
        (1.99, "unreliable"),
        (2.0, "intermediate"),
        (3.99, "intermediate"),
        (4.0, "reliable"),
        (9.5, "reliable"),
        (float("inf"), "reliable"),
        (float("nan"), "unknown"),
    ],
)
def test_scalar_speed_gives_a_string(speed: float, expected: str):
    label = classify_xy_reliability(speed)
    assert isinstance(label, str)
    assert label == expected


def test_array_speed_keeps_the_shape():
    speeds = np.array([[0.5, 2.0], [4.0, np.nan]])
    labels = classify_xy_reliability(speeds)
    assert labels.shape == speeds.shape
    assert labels.tolist() == [["unreliable", "intermediate"], ["reliable", "unknown"]]


def test_series_and_list_are_accepted_and_filter_a_table():
    table = pd.DataFrame({"carrier_speed": [1.0, 3.0, 5.0, 7.0, None]})
    table["xy_reliability"] = classify_xy_reliability(table["carrier_speed"])
    assert table.loc[table["xy_reliability"] == "reliable", "carrier_speed"].tolist() == [5.0, 7.0]
    assert classify_xy_reliability([1.0]).tolist() == ["unreliable"]
    assert classify_xy_reliability([]).shape == (0,)


def test_negative_speed_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        classify_xy_reliability(-0.1)
    with pytest.raises(ValueError, match="negative"):
        classify_xy_reliability([1.0, -2.0])


def test_thresholds_are_edges_of_the_speed_bands_of_the_agreement_table():
    assert XY_UNRELIABLE_BELOW_MPS in SPEED_EDGES
    assert XY_RELIABLE_FROM_MPS in SPEED_EDGES
    assert XY_UNRELIABLE_BELOW_MPS < XY_RELIABLE_FROM_MPS
    assert XY_RELIABILITY_CLASSES == ("unreliable", "intermediate", "reliable")


def test_proxy_is_blind_only_above_the_threshold_and_never_for_a_standing_carrier():
    assert XY_PROXY_ANGLE_DEG == FOV_HALF_ANGLE_DEG
    angles = [0.0, 100.0, 100.1, 180.0, np.nan]
    assert xy_blind_side_proxy(angles).tolist() == [False, False, True, True, False]
    assert xy_blind_side_proxy(95.0, threshold_deg=90.0)


def test_helpers_are_part_of_the_public_api():
    assert blindside.classify_xy_reliability(4.2) == "reliable"
    assert blindside.xy_blind_side_proxy([170.0]).tolist() == [True]
