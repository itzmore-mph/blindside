"""Tests for the incremental value comparison on synthetic engagements."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from blindside.analysis.baseline import bootstrap_metrics, cluster_bootstrap_weights
from blindside.analysis.incremental import (
    BIN_LABELS,
    OutOfFold,
    bin_index,
    bin_table,
    bootstrap_mean,
    calibration_slope,
    choose_penalty,
    fit_logistic,
    grouped_folds,
    log_loss_gain,
    log_loss_terms,
    model1_coefficients,
    out_of_fold,
    predict_logistic,
    train_test_predictions,
)


def _data(n: int = 1200, pose_effect: float = 0.0, seed: int = 0):
    """Engagements in clusters of one to three, two matches, outcome from z and one pose feature."""
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n // 2), 2)[:n]
    clusters[rng.random(n) < 0.3] += n  # break some pairs into singletons
    strata = (np.arange(n) >= n // 2).astype(int)
    clusters = clusters + 10 * n * strata
    z = rng.normal(-2.0, 1.0, n)
    pose = rng.normal(size=(n, 4))
    eta = 0.3 + 0.9 * z + pose_effect * pose[:, 0]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return z, pose, y, clusters, strata


def test_fit_logistic_matches_scikit_learn():
    z, pose, y, _, _ = _data(pose_effect=0.8)
    x = np.column_stack([z, pose])
    for c in (0.01, 1.0, 100.0):
        reference = LogisticRegression(C=c, tol=1e-10, max_iter=5000).fit(x, y.astype(int))
        ours = fit_logistic(x, y, np.full(x.shape[1], 1.0 / c))
        np.testing.assert_allclose(ours[0], reference.intercept_[0], atol=2e-4)
        np.testing.assert_allclose(ours[1:], reference.coef_[0], atol=2e-4)
        np.testing.assert_allclose(
            predict_logistic(ours, x), reference.predict_proba(x)[:, 1], atol=1e-4
        )


def test_fit_logistic_penalises_only_the_named_coefficients():
    z, pose, y, _, _ = _data(pose_effect=0.8)
    x = np.column_stack([z, pose])
    free = fit_logistic(x, y)
    heavy = fit_logistic(x, y, np.r_[0.0, np.full(4, 1e6)])
    model0 = fit_logistic(z[:, None], y)
    assert abs(free[2]) > 0.5
    np.testing.assert_allclose(heavy[2:], 0.0, atol=1e-3)
    # With the pose coefficients shrunk to zero, the rest equals Model 0.
    np.testing.assert_allclose(heavy[:2], model0, atol=1e-3)


def test_fit_logistic_weights_equal_repeated_rows():
    z, pose, y, _, _ = _data(n=300)
    x = np.column_stack([z, pose])
    weight = np.random.default_rng(1).integers(0, 4, len(y))
    repeated = np.repeat(np.arange(len(y)), weight)
    penalty = np.r_[0.0, np.full(4, 2.0)]
    np.testing.assert_allclose(
        fit_logistic(x, y, penalty, weight.astype(float)),
        fit_logistic(x[repeated], y[repeated], penalty),
        atol=1e-7,
    )


def test_grouped_folds_keep_clusters_together_and_balance_strata():
    _, _, _, clusters, strata = _data()
    fold = grouped_folds(clusters, strata, 10, np.random.default_rng(0))
    assert set(fold) == set(range(10))
    for cluster in np.unique(clusters):
        assert len(set(fold[clusters == cluster])) == 1
    for stratum in (0, 1):
        per_fold = [len(np.unique(clusters[(fold == k) & (strata == stratum)])) for k in range(10)]
        assert max(per_fold) - min(per_fold) <= 1
    other = grouped_folds(clusters, strata, 10, np.random.default_rng(1))
    assert not np.array_equal(fold, other)


def test_out_of_fold_finds_an_informative_pose_feature():
    z, pose, y, clusters, strata = _data(pose_effect=1.0)
    oof = out_of_fold(z, pose, y, clusters, strata, n_repeats=2)
    weights = cluster_bootstrap_weights(clusters, strata, 300, seed=0)
    gain, low, high = log_loss_gain(y, oof, weights)
    assert gain > 0.02
    assert low > 0.0
    assert low < gain < high
    assert len(oof.chosen_c) == 20
    assert ((oof.model0 > 0) & (oof.model0 < 1)).all()


def test_out_of_fold_gives_no_gain_for_noise_features():
    z, pose, y, clusters, strata = _data(pose_effect=0.0)
    oof = out_of_fold(z, pose, y, clusters, strata, n_repeats=2)
    weights = cluster_bootstrap_weights(clusters, strata, 300, seed=0)
    gain, low, high = log_loss_gain(y, oof, weights)
    assert abs(gain) < 0.01
    assert low < 0.0 < high or gain < 0.0


def test_out_of_fold_is_deterministic_and_accepts_covariates():
    z, pose, y, clusters, strata = _data(n=400, pose_effect=0.5)
    a = out_of_fold(z, pose, y, clusters, strata, n_repeats=1)
    b = out_of_fold(z, pose, y, clusters, strata, n_repeats=1)
    np.testing.assert_array_equal(a.model1, b.model1)
    covariate = np.random.default_rng(3).normal(size=len(y))
    c = out_of_fold(z, pose, y, clusters, strata, covariates=covariate, n_repeats=1)
    assert not np.allclose(a.model0, c.model0)


def test_choose_penalty_prefers_strong_shrinkage_for_noise():
    z, pose, y, clusters, strata = _data(pose_effect=0.0)
    rng = np.random.default_rng(0)
    assert choose_penalty(z[:, None], pose, y, clusters, strata, rng) <= 0.1
    z, pose, y, clusters, strata = _data(pose_effect=1.5)
    assert choose_penalty(z[:, None], pose, y, clusters, strata, rng) >= 0.01


def test_train_test_predictions_fill_only_the_test_rows():
    z, pose, y, clusters, strata = _data(pose_effect=1.0)
    result = train_test_predictions(z, pose, y, clusters, strata == 0, strata == 1)
    assert np.isnan(result.model1[strata == 0]).all()
    assert np.isfinite(result.model1[strata == 1]).all()
    weights = cluster_bootstrap_weights(clusters, strata, 200, seed=0)
    gain, low, _ = log_loss_gain(y, result, weights)
    assert gain > 0.0 and low > 0.0


def test_log_loss_gain_matches_bootstrap_metrics():
    z, pose, y, clusters, strata = _data(n=400, pose_effect=1.0)
    rng = np.random.default_rng(0)
    oof = OutOfFold(rng.uniform(0.05, 0.5, len(y)), rng.uniform(0.05, 0.5, len(y)), np.array([]))
    weights = cluster_bootstrap_weights(clusters, strata, 200, seed=0)
    reference = bootstrap_metrics(
        y, {"model0": oof.model0, "model1": oof.model1}, weights, reference="model1"
    )
    row = reference[reference["metric"] == "log_loss_vs_model1"].iloc[0]
    np.testing.assert_allclose(
        log_loss_gain(y, oof, weights), row[["estimate", "ci_low", "ci_high"]].to_numpy(float)
    )
    assert log_loss_terms(y, oof.model0).mean() == pytest.approx(
        reference[(reference["model"] == "model0") & (reference["metric"] == "log_loss")][
            "estimate"
        ].iloc[0]
    )


def test_bootstrap_mean_on_a_subset():
    values = np.r_[np.ones(50), np.zeros(50)]
    clusters = np.arange(100)
    weights = cluster_bootstrap_weights(clusters, None, 500, seed=0)
    estimate, low, high = bootstrap_mean(values, weights)
    assert estimate == 0.5 and low < 0.5 < high
    estimate, low, high = bootstrap_mean(values, weights, values == 1.0)
    assert (estimate, low, high) == (1.0, 1.0, 1.0)
    assert np.isnan(bootstrap_mean(values, weights, np.zeros(100, dtype=bool))).all()


def test_calibration_slope():
    rng = np.random.default_rng(0)
    eta = rng.normal(-1.0, 1.5, 20_000)
    y = (rng.random(len(eta)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    assert calibration_slope(y, 1.0 / (1.0 + np.exp(-eta))) == pytest.approx(1.0, abs=0.05)
    assert calibration_slope(y, 1.0 / (1.0 + np.exp(-2.0 * eta))) == pytest.approx(0.5, abs=0.03)


def test_bin_index_edges():
    p = np.array([0.0, 0.099, 0.1, 0.5, 0.9, 0.901, 1.0, np.nan])
    np.testing.assert_array_equal(bin_index(p), [0, 0, 1, 1, 1, 2, 2, -1])


def test_bin_table_rates_excess_and_small_bins():
    n = 400
    p = np.r_[np.zeros(200), np.full(170, 0.5), np.ones(30)]
    y = np.zeros(n)
    y[:20] = 1.0  # 10% in view
    y[200:251] = 1.0  # 30% uncertain
    y[370:385] = 1.0  # 50% blind
    expected = np.full(n, 0.2)
    weights = cluster_bootstrap_weights(np.arange(n), None, 500, seed=0)
    table = bin_table(p, y, expected, weights)
    assert list(table["bin"]) == list(BIN_LABELS)
    assert list(table["n"]) == [200, 170, 30]
    np.testing.assert_allclose(table["rate"], [0.1, 0.3, 0.5])
    np.testing.assert_allclose(table["excess"], [-0.1, 0.1, 0.3])
    np.testing.assert_allclose(table["expected"], 0.2)
    assert (table["rate_low"][:2] < table["rate"][:2]).all()
    assert (table["excess_high"][:2] > table["excess"][:2]).all()
    # The interval of the excess is the interval of the rate shifted by a constant expectation.
    np.testing.assert_allclose(table["excess_low"][:2], table["rate_low"][:2] - 0.2)
    # Fewer than 50 engagements: count and rate, no interval.
    assert np.isnan(table.loc[2, ["rate_low", "rate_high", "excess_low", "excess_high"]]).all()


def test_bin_table_skips_missing_outcomes():
    p = np.r_[np.zeros(100), np.ones(100)]
    y = np.r_[np.full(100, 2.0), np.full(60, 4.0), np.full(40, np.nan)]
    weights = cluster_bootstrap_weights(np.arange(200), None, 100, seed=0)
    table = bin_table(p, y, np.full(200, np.nan), weights)
    assert list(table["n"]) == [100, 0, 60]
    np.testing.assert_allclose(table["rate"].to_numpy()[[0, 2]], [2.0, 4.0])


def test_model1_coefficients_recover_the_effect():
    z, pose, y, clusters, strata = _data(n=3000, pose_effect=1.0)
    weights = cluster_bootstrap_weights(clusters, strata, 100, seed=0)
    table, c = model1_coefficients(z, pose, y, clusters, strata, weights, ("a", "b", "c", "d"))
    assert list(table["feature"]) == ["intercept", "z_xy", "a", "b", "c", "d"]
    effect = table.set_index("feature").loc["a"]
    assert effect["ci_low"] < 1.0 < effect["ci_high"] or abs(effect["coefficient"] - 1.0) < 0.15
    assert effect["ci_low"] > 0.5
    noise = table.set_index("feature").loc["b"]
    assert noise["ci_low"] < 0.0 < noise["ci_high"] or abs(noise["coefficient"]) < 0.1
    assert c > 0
