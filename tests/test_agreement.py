"""Tests for the agreement of two binary labels with a cluster bootstrap."""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from blindside.analysis.agreement import agreement_table, cohen_kappa
from blindside.analysis.baseline import cluster_bootstrap_weights


def _labels(n: int = 600, flip: float = 0.2, seed: int = 0):
    """Two labels that disagree for a share ``flip`` of the rows, in clusters of two."""
    rng = np.random.default_rng(seed)
    a = rng.random(n) < 0.3
    b = np.where(rng.random(n) < flip, ~a, a)
    clusters = np.arange(n) // 2
    return a, b, clusters


def test_cohen_kappa_matches_scikit_learn():
    for flip in (0.05, 0.2, 0.5):
        a, b, _ = _labels(flip=flip)
        assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-12)


def test_cohen_kappa_limits():
    a, _, _ = _labels()
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    assert cohen_kappa(a, ~a) < 0.0
    independent = np.random.default_rng(1).random(200_000) < 0.4
    other = np.random.default_rng(2).random(200_000) < 0.2
    assert abs(cohen_kappa(independent, other)) < 0.01
    # A constant pair of labels has no chance-corrected agreement.
    constant = np.ones(50, dtype=bool)
    assert np.isnan(cohen_kappa(constant, constant))
    assert np.isnan(cohen_kappa(np.array([], dtype=bool), np.array([], dtype=bool)))


def test_cohen_kappa_weights_equal_repeated_rows():
    a, b, _ = _labels(n=200)
    weights = np.random.default_rng(3).integers(0, 4, size=(5, len(a)))
    weighted = cohen_kappa(a, b, weights)
    assert weighted.shape == (5,)
    for row, value in zip(weights, weighted, strict=True):
        repeated = np.repeat(np.arange(len(a)), row)
        assert value == pytest.approx(cohen_kappa_score(a[repeated], b[repeated]), abs=1e-12)


def test_agreement_table_counts_on_a_known_table():
    # 2x2 table: both blind 30, pose only 10, proxy only 20, neither 40.
    pose = np.r_[np.ones(30), np.ones(10), np.zeros(20), np.zeros(40)].astype(bool)
    proxy = np.r_[np.ones(30), np.zeros(10), np.ones(20), np.zeros(40)].astype(bool)
    weights = cluster_bootstrap_weights(np.arange(100), n_resamples=500)
    row = agreement_table(pose, proxy, weights).iloc[0]
    assert row["group"] == "all"
    assert row["n"] == 100
    assert row["pose_blind"] == pytest.approx(0.40)
    assert row["xy_blind"] == pytest.approx(0.50)
    assert row["agreement"] == pytest.approx(0.70)
    assert row["recall"] == pytest.approx(0.75)  # 30 of 40 pose-blind are proxy-blind
    assert row["precision"] == pytest.approx(0.60)  # 30 of 50 proxy-blind are pose-blind
    assert row["kappa"] == pytest.approx((0.70 - 0.50) / (1.0 - 0.50))
    for name in ("agreement", "kappa", "recall", "precision"):
        assert row[f"{name}_low"] < row[name] < row[f"{name}_high"]


def test_agreement_table_groups_keep_and_small_groups():
    a, b, clusters = _labels()
    weights = cluster_bootstrap_weights(clusters, n_resamples=300)
    index = np.arange(len(a))
    groups = {"first": index < 300, "second": index >= 300, "tiny": index < 20}
    keep = index % 3 != 0
    table = agreement_table(a, b, weights, groups, keep=keep, min_size=50)
    assert list(table["group"]) == ["all", "first", "second", "tiny"]
    assert table["n"].tolist() == [
        int(keep.sum()),
        int((keep & groups["first"]).sum()),
        int((keep & groups["second"]).sum()),
        int((keep & groups["tiny"]).sum()),
    ]
    first = table.iloc[1]
    subset = keep & groups["first"]
    assert first["agreement"] == pytest.approx(np.mean(a[subset] == b[subset]))
    assert first["kappa"] == pytest.approx(cohen_kappa_score(a[subset], b[subset]))
    # A group below the minimum size keeps its estimates and carries no interval.
    tiny = table.iloc[3]
    assert np.isfinite(tiny["agreement"])
    assert np.isnan(tiny["agreement_low"]) and np.isnan(tiny["kappa_high"])


def test_agreement_table_cluster_bootstrap_is_wider_than_row_bootstrap():
    # Rows of a cluster share their labels, so resampling clusters must widen the interval.
    rng = np.random.default_rng(4)
    a = np.repeat(rng.random(150) < 0.3, 4)
    b = np.where(np.repeat(rng.random(150) < 0.25, 4), ~a, a)
    clusters = np.repeat(np.arange(150), 4)
    by_cluster = agreement_table(a, b, cluster_bootstrap_weights(clusters)).iloc[0]
    by_row = agreement_table(a, b, cluster_bootstrap_weights(np.arange(len(a)))).iloc[0]
    assert (by_cluster["kappa_high"] - by_cluster["kappa_low"]) > 1.5 * (
        by_row["kappa_high"] - by_row["kappa_low"]
    )


def test_agreement_table_rejects_missing_labels():
    a, b, clusters = _labels(n=40)
    weights = cluster_bootstrap_weights(clusters, n_resamples=10)
    with pytest.raises(ValueError, match="shape"):
        agreement_table(a, b[:-1], weights)
