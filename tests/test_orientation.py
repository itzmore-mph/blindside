"""Tests for facing angles on synthetic skeletons with known orientation."""

import numpy as np
import pytest

from blindside.geometry.circular import angular_difference, circular_mean
from blindside.geometry.orientation import (
    SEGMENT_JOINTS,
    anatomical_forward,
    facing_angle,
    nose_direction,
    resolve_front_back,
    segment_width,
    vector_angle,
)
from skeleton import make_skeleton, mirror_y, rotate

DEG = np.pi / 180.0
FORWARD_JOINTS = ("nose", "lEar", "rEar", "lHeel", "rHeel", "lBigToe", "rBigToe")


def _facings(joints: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        segment: facing_angle(joints[left], joints[right])
        for segment, (left, right) in SEGMENT_JOINTS.items()
    }


def _forward(joints: dict[str, np.ndarray]) -> np.ndarray:
    return anatomical_forward(*(joints[name] for name in FORWARD_JOINTS))


@pytest.mark.parametrize("degrees", [0, 30, 90, 135, 180, -45, -90, -179])
def test_known_facing_angle(degrees: float) -> None:
    joints = make_skeleton(degrees * DEG, position=(12.0, -7.5))
    for segment, facing in _facings(joints).items():
        assert abs(angular_difference(facing, degrees * DEG)) < 1e-9, segment
    assert abs(angular_difference(_forward(joints), degrees * DEG)) < 1e-9
    nose = nose_direction(joints["nose"], joints["lEar"], joints["rEar"])
    assert abs(angular_difference(nose, degrees * DEG)) < 1e-9


def test_convention_matches_the_documented_example() -> None:
    # A player facing +x has the left shoulder at +y. Left-to-right points to -y,
    # rotated by +90 degrees it points to +x.
    assert facing_angle([0.0, 0.2], [0.0, -0.2]) == pytest.approx(0.0)
    assert facing_angle([-0.2, 0.0], [0.2, 0.0]) == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("rotation", [17.0, 90.0, 200.0, -123.0])
def test_rotating_the_skeleton_rotates_the_facing_angle(rotation: float) -> None:
    joints = rotate(make_skeleton(40 * DEG, position=(3.0, 4.0)), rotation * DEG)
    for facing in _facings(joints).values():
        assert abs(angular_difference(facing, (40 + rotation) * DEG)) < 1e-9


def test_mirrored_skeleton_with_correct_labels_faces_the_mirrored_direction() -> None:
    joints = mirror_y(make_skeleton(40 * DEG), swap_labels=True)
    for facing in _facings(joints).values():
        assert abs(angular_difference(facing, -40 * DEG)) < 1e-9
    resolved, flipped = resolve_front_back(_facings(joints)["torso"], _forward(joints))
    assert not flipped
    assert abs(angular_difference(resolved, -40 * DEG)) < 1e-9


def test_swapped_labels_point_backwards_and_are_resolved() -> None:
    joints = mirror_y(make_skeleton(40 * DEG), swap_labels=False)
    forward = _forward(joints)
    # The label-independent cue still sees the true direction of the mirrored body.
    assert abs(angular_difference(forward, -40 * DEG)) < 1e-9
    for facing in _facings(joints).values():
        assert abs(angular_difference(facing, (180 - 40) * DEG)) < 1e-9
        resolved, flipped = resolve_front_back(facing, forward)
        assert flipped
        assert abs(angular_difference(resolved, -40 * DEG)) < 1e-9


def test_forward_cue_ignores_left_right_labels() -> None:
    joints = make_skeleton(np.array([10.0, 100.0, -160.0]) * DEG)
    swapped = dict(joints)
    for a, b in (("lEar", "rEar"), ("lHeel", "rHeel"), ("lBigToe", "rBigToe")):
        swapped[a], swapped[b] = joints[b], joints[a]
    np.testing.assert_allclose(_forward(swapped), _forward(joints))


def test_resolve_front_back_keeps_angles_within_90_degrees_and_nan_references() -> None:
    facing = np.array([0.0, 0.0, 0.0, 0.0]) * DEG
    reference = np.array([80.0, 100.0, 180.0, np.nan]) * DEG
    resolved, flipped = resolve_front_back(facing, reference)
    np.testing.assert_array_equal(flipped, [False, True, True, False])
    np.testing.assert_allclose(np.abs(resolved), [0.0, np.pi, np.pi, 0.0])


def test_vectorised_over_frames_and_players() -> None:
    rng = np.random.default_rng(3)
    truth = rng.uniform(-np.pi, np.pi, size=(50, 22))
    position = rng.uniform(-50, 50, size=(50, 22, 2))
    joints = make_skeleton(truth, position)
    for segment, facing in _facings(joints).items():
        assert facing.shape == (50, 22), segment
        assert np.max(np.abs(angular_difference(facing, truth))) < 1e-9
    assert segment_width(joints["lHip"], joints["rHip"]).shape == (50, 22)
    np.testing.assert_allclose(segment_width(joints["lHip"], joints["rHip"]), 0.22)


def test_noisy_skeletons_are_unbiased() -> None:
    rng = np.random.default_rng(4)
    truth = 170 * DEG  # close to the wrap on purpose
    joints = make_skeleton(np.full(20_000, truth))
    noisy = {name: xy + rng.normal(0.0, 0.02, size=xy.shape) for name, xy in joints.items()}
    for segment, facing in _facings(noisy).items():
        error = angular_difference(facing, truth)
        assert abs(circular_mean(error)) < 0.5 * DEG, segment
        assert np.std(error) > 1 * DEG, segment  # the noise does reach the angle


def test_degenerate_and_missing_joints_give_nan() -> None:
    assert np.isnan(facing_angle([1.0, 2.0], [1.0, 2.0]))
    assert np.isnan(facing_angle([np.nan, 0.0], [1.0, 2.0]))
    assert np.isnan(vector_angle([0.0, 0.0]))
