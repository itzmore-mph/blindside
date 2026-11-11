"""Tests for the pose features on synthetic skeletons with known facing angles."""

import json

import numpy as np
import pandas as pd
import pytest

from blindside.features.engagements import pose_available
from blindside.features.pose_features import (
    FEATURE_JOINTS,
    POSE_FEATURES,
    blind_side_probability,
    carrier_attacking_angle,
    carrier_openness,
    pose_feature_table,
    presser_alignment_deg,
    read_engagement_poses,
    select_pose_frames,
)
from skeleton import make_skeleton, mirror_y

PRESSER, CARRIER = 1, 2
P90_CM = 6.0


def _pose_row(frame: int, player: int, joints: dict[str, np.ndarray], xy: tuple[float, float]):
    row = {"match_id": 7, "frame": frame, "player_id": player, "x": xy[0], "y": xy[1]}
    for name in FEATURE_JOINTS:
        row[f"{name}_x"], row[f"{name}_y"] = joints[name]
        row[f"{name}_p90"] = P90_CM
    return row


def _scene(
    carrier_facing_deg: float,
    presser_bearing_deg: float,
    presser_facing_deg: float = 0.0,
    side: str = "right_to_left",
    swap_carrier: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrier at the origin, presser 3 m away at ``presser_bearing_deg``."""
    bearing = np.radians(presser_bearing_deg)
    presser_xy = (3.0 * np.cos(bearing), 3.0 * np.sin(bearing))
    carrier = make_skeleton(np.radians(carrier_facing_deg))
    if swap_carrier:  # wrong left/right labels: the label-based facing points backwards
        carrier = mirror_y(make_skeleton(-np.radians(carrier_facing_deg)), swap_labels=False)
    presser = make_skeleton(np.radians(presser_facing_deg), presser_xy)
    poses = pd.DataFrame(
        [_pose_row(50, CARRIER, carrier, (0.0, 0.0)), _pose_row(50, PRESSER, presser, presser_xy)]
    )
    engagements = pd.DataFrame(
        {
            "match_id": [7],
            "pose_frame": [50],
            "presser_id": [PRESSER],
            "carrier_id": [CARRIER],
            "attacking_side": [side],
        }
    )
    return engagements, poses


def _samples(angle_deg: float, n: int = 1) -> np.ndarray:
    return np.full((200, n), np.radians(angle_deg))


@pytest.mark.parametrize(
    ("bearing", "expected"),
    [(0.0, 0.0), (99.0, 0.0), (101.0, 1.0), (180.0, 1.0), (-101.0, 1.0), (-99.0, 0.0)],
)
def test_blind_side_probability_without_noise(bearing, expected):
    presser = 5.0 * np.array([[np.cos(np.radians(bearing)), np.sin(np.radians(bearing))]])
    p = blind_side_probability(_samples(0.0), np.zeros((1, 2)), presser)
    assert p[0] == expected


def test_blind_side_probability_is_rotation_invariant_and_wraps():
    # Carrier faces 170 degrees, presser at -170 degrees: 20 degrees off, across the wrap.
    presser = np.array([[np.cos(np.radians(-170.0)), np.sin(np.radians(-170.0))]])
    assert blind_side_probability(_samples(170.0), np.zeros((1, 2)), presser)[0] == 0.0
    assert blind_side_probability(_samples(170.0), np.zeros((1, 2)), presser, 15.0)[0] == 1.0


def test_blind_side_probability_is_a_share_of_samples():
    samples = np.radians(np.r_[np.zeros(150), np.full(50, 180.0)])[:, None]
    presser = np.array([[5.0, 0.0]])
    assert blind_side_probability(samples, np.zeros((1, 2)), presser)[0] == pytest.approx(0.25)


def test_blind_side_probability_undefined_inputs():
    samples = _samples(0.0, 2)
    samples[3, 1] = np.nan
    presser = np.array([[0.0, 0.0], [1.0, 0.0]])  # first presser stands on the carrier
    assert np.isnan(blind_side_probability(samples, np.zeros((2, 2)), presser)).all()
    with pytest.raises(ValueError):
        blind_side_probability(samples, np.zeros((2, 2)), presser, 180.0)


def test_presser_alignment():
    carrier = np.array([[4.0, 0.0]] * 3)
    presser = np.zeros((3, 2))
    samples = np.radians(np.array([[0.0, 180.0, -90.0]])).repeat(200, axis=0)
    np.testing.assert_allclose(
        presser_alignment_deg(samples, presser, carrier), [0.0, 180.0, 90.0], atol=1e-9
    )


def test_carrier_openness_and_attacking_direction():
    # The pressing team attacks right to left, so the carrier attacks towards +x.
    angle = carrier_attacking_angle(
        np.array(["right_to_left", "left_to_right", None], dtype=object)
    )
    np.testing.assert_allclose(angle[:2], [0.0, np.pi])
    assert np.isnan(angle[2])
    samples = np.radians(np.array([[0.0, 0.0, 90.0]])).repeat(200, axis=0)
    openness = carrier_openness(samples, np.array([0.0, np.pi, 0.0]))
    np.testing.assert_allclose(openness, [1.0, -1.0, 0.0], atol=1e-9)


@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_feature_table_clear_cases(bound):
    behind, poses_behind = _scene(0.0, 180.0, presser_facing_deg=0.0)
    front, poses_front = _scene(0.0, 0.0, presser_facing_deg=180.0)
    out_behind = pose_feature_table(behind, poses_behind, bound=bound)
    out_front = pose_feature_table(front, poses_front, bound=bound)
    assert set(POSE_FEATURES) <= set(out_behind.columns)
    # Presser behind a carrier who faces +x and attacks +x. The presser faces the carrier.
    assert out_behind["p_blind_side_head"].iloc[0] == 1.0
    assert out_behind["p_blind_side_torso"].iloc[0] == 1.0
    assert out_behind["presser_alignment"].iloc[0] < 15.0
    assert out_behind["carrier_openness"].iloc[0] > 0.95
    assert out_behind["presser_off_head_deg"].iloc[0] == pytest.approx(180.0)
    assert out_front["p_blind_side_head"].iloc[0] == 0.0
    assert out_front["p_blind_side_torso"].iloc[0] == 0.0
    assert out_front["presser_alignment"].iloc[0] < 15.0


def test_bounds_bracket_the_probability_near_the_edge_of_the_visual_field():
    # Presser 110 degrees off the head direction: 10 degrees outside the visual field.
    engagements, poses = _scene(0.0, 110.0)
    lower = pose_feature_table(engagements, poses, bound="lower")
    upper = pose_feature_table(engagements, poses, bound="upper")
    assert lower["carrier_head_sd_deg"].iloc[0] < 4.0 < upper["carrier_head_sd_deg"].iloc[0]
    assert lower["p_blind_side_head"].iloc[0] > 0.99
    assert 0.6 < upper["p_blind_side_head"].iloc[0] < 0.95
    # The torso baseline is wider, so its upper bound is sharper than the head's.
    assert upper["p_blind_side_torso"].iloc[0] > upper["p_blind_side_head"].iloc[0]


def test_wider_visual_field_lowers_the_probability():
    engagements, poses = _scene(0.0, 100.0)
    p = [
        pose_feature_table(engagements, poses, half_angle_deg=h)["p_blind_side_head"].iloc[0]
        for h in (90.0, 100.0, 110.0)
    ]
    assert p[0] > p[1] > p[2]
    assert 0.3 < p[1] < 0.7


def test_swapped_labels_are_used_as_they_are():
    # The flip flag is reported elsewhere and never applied: swapped labels turn the facing by 180.
    engagements, poses = _scene(0.0, 180.0, swap_carrier=True)
    out = pose_feature_table(engagements, poses)
    assert out["p_blind_side_head"].iloc[0] == 0.0
    assert out["presser_off_torso_deg"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_feature_table_is_reproducible_and_shares_draws_between_calls():
    engagements, poses = _scene(0.0, 105.0)
    a = pose_feature_table(engagements, poses, seed=0)
    b = pose_feature_table(engagements, poses, seed=0)
    c = pose_feature_table(engagements, poses, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_missing_pose_gives_nan():
    engagements, poses = _scene(0.0, 180.0)
    out = pose_feature_table(engagements, poses[poses["player_id"] == PRESSER])
    assert out[list(POSE_FEATURES)].isna().all(axis=None)


def _keys(pairs: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["frame", "player_id"])


def _engagements(frames: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"frame_start": frames, "presser_id": PRESSER, "carrier_id": CARRIER})


def test_select_pose_frames_takes_the_nearest_and_breaks_ties_early():
    def both(frame: int) -> list[tuple[int, int]]:
        return [(frame, PRESSER), (frame, CARRIER)]

    # Tracking frame 20 is pose frame 50. Tracking frame 21 is pose frame 52.5.
    assert select_pose_frames(_engagements([20]), _keys(both(50) + both(49)))[0] == 50
    assert select_pose_frames(_engagements([20]), _keys(both(49) + both(51)))[0] == 49
    assert select_pose_frames(_engagements([20]), _keys(both(52) + both(47)))[0] == 52
    assert select_pose_frames(_engagements([21]), _keys(both(52) + both(53)))[0] == 52
    assert select_pose_frames(_engagements([21]), _keys(both(54) + both(51)))[0] == 51
    # Out of the window, or the two players never in the same frame.
    assert select_pose_frames(_engagements([20]), _keys(both(53)))[0] == -1
    assert select_pose_frames(_engagements([20]), _keys([(50, PRESSER), (51, CARRIER)]))[0] == -1


def test_select_pose_frames_agrees_with_the_inclusion_criterion():
    rng = np.random.default_rng(0)
    pairs = {
        (int(f), int(p))
        for f, p in zip(rng.integers(0, 300, 400), rng.integers(1, 3, 400), strict=True)
    }
    keys = _keys(sorted(pairs))
    engagements = _engagements(list(range(0, 120)))
    selected = select_pose_frames(engagements, keys)
    np.testing.assert_array_equal(selected >= 0, pose_available(engagements, keys))
    target = 2.5 * engagements["frame_start"].to_numpy()
    assert (np.abs(selected - target)[selected >= 0] <= 2).all()


def test_read_engagement_poses(tmp_path):
    def joints(offset: float) -> dict:
        names = [*FEATURE_JOINTS, "nose"]
        return {
            name: {"xyz": [offset + i, offset - i, 1.0], "p90_mae_cm": 5.0 + i}
            for i, name in enumerate(names)
        }

    frames = [
        {"frame": 10, "period": 1, "player_data": [
            {"player_id": PRESSER, "x": 1.0, "y": 2.0, "joints": joints(0.0)},
            {"player_id": CARRIER, "x": 3.0, "y": 4.0, "joints": None},
        ]},
        {"frame": 11, "period": 1, "player_data": [
            {"player_id": PRESSER, "x": 1.5, "y": 2.5, "joints": joints(10.0)},
            {"player_id": CARRIER, "x": 3.5, "y": 4.5, "joints": joints(20.0)},
        ]},
        {"frame": 12, "period": 1, "player_data": []},
    ]  # fmt: skip
    path = tmp_path / "pose.jsonl"
    path.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n")
    table = read_engagement_poses(path, {(10, PRESSER), (10, CARRIER), (11, CARRIER), (12, 9)}, 7)
    assert list(zip(table["frame"], table["player_id"], strict=True)) == [
        (10, PRESSER),
        (11, CARRIER),
    ]
    row = table.iloc[1]
    assert (row["match_id"], row["x"], row["y"]) == (7, 3.5, 4.5)
    assert (row["lShoulder_x"], row["lShoulder_y"], row["lShoulder_p90"]) == (20.0, 20.0, 5.0)
    assert (row["rEar_x"], row["rEar_y"], row["rEar_p90"]) == (25.0, 15.0, 10.0)
    assert read_engagement_poses(path, set(), 7).empty
