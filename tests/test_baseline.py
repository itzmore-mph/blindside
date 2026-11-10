"""Tests for the baseline model, the metrics and the cluster bootstrap."""

import numpy as np
import pandas as pd
import pytest

from blindside.analysis.baseline import (
    C_GRID,
    auc,
    bootstrap_metrics,
    brier,
    cluster_bootstrap_weights,
    leave_one_match_out,
    log_loss,
    logit,
    predict_probability,
    tune_and_fit,
)

NUMERIC = ("signal", "noise")
CATEGORICAL = ("zone",)


@pytest.fixture(scope="module")
def engagements() -> pd.DataFrame:
    """Six matches. The outcome depends on ``signal`` and on the level of ``zone``."""
    rng = np.random.default_rng(3)
    n = 3000
    signal = rng.normal(size=n)
    zone = rng.choice(["a", "b"], size=n)
    probability = 1.0 / (1.0 + np.exp(-(-1.5 + 1.2 * signal + 0.8 * (zone == "b"))))
    table = pd.DataFrame(
        {
            "match_id": rng.integers(0, 6, n),
            "signal": signal,
            "noise": rng.normal(size=n),
            "zone": zone,
            "y": (rng.random(n) < probability).astype(float),
            "p_true": probability,
        }
    )
    table.loc[rng.random(n) < 0.05, "signal"] = np.nan
    return table


def test_metrics_on_known_values() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    assert log_loss(y, p) == pytest.approx(-(2 * np.log(0.8) + 2 * np.log(0.6)) / 4)
    assert brier(y, p) == pytest.approx((2 * 0.04 + 2 * 0.16) / 4)
    assert auc(y, p) == 1.0
    assert auc(y, 1.0 - p) == 0.0
    # Weights act like repeated observations.
    weight = np.array([2, 1, 0, 1])
    assert log_loss(y, p, weight) == pytest.approx(
        log_loss(np.array([1.0, 1.0, 0.0, 0.0]), np.array([0.8, 0.8, 0.2, 0.4]))
    )
    assert np.isnan(auc(y, p, np.array([1, 0, 1, 0])))
    np.testing.assert_allclose(logit(np.array([0.5, 0.75])), [0.0, np.log(3.0)])
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


def test_bootstrap_weights_resample_whole_clusters_within_strata() -> None:
    clusters = np.array(["a", "a", "b", "c", "c", "c", "d"])
    strata = np.array([1, 1, 1, 2, 2, 2, 2])
    weights = cluster_bootstrap_weights(clusters, strata, n_resamples=500, seed=1)
    assert weights.shape == (500, 7)
    # Observations of one cluster always share their weight.
    assert (weights[:, 0] == weights[:, 1]).all()
    assert (weights[:, 3] == weights[:, 4]).all() and (weights[:, 4] == weights[:, 5]).all()
    # Each stratum keeps its number of clusters in every resample.
    assert ((weights[:, 0] + weights[:, 2]) == 2).all()
    assert ((weights[:, 3] + weights[:, 6]) == 2).all()
    # A cluster is drawn once on average.
    assert weights[:, 2].mean() == pytest.approx(1.0, abs=0.1)
    np.testing.assert_array_equal(
        weights, cluster_bootstrap_weights(clusters, strata, n_resamples=500, seed=1)
    )


def test_cluster_bootstrap_is_wider_than_an_observation_bootstrap() -> None:
    rng = np.random.default_rng(0)
    cluster = np.repeat(np.arange(200), 5)
    # All five observations of a cluster share the outcome, so they carry one piece of information.
    y = np.repeat((rng.random(200) < 0.3).astype(float), 5)
    p = np.full(len(y), 0.3)
    clustered = bootstrap_metrics(y, {"m": p}, cluster_bootstrap_weights(cluster, n_resamples=400))
    naive = bootstrap_metrics(
        y, {"m": p}, cluster_bootstrap_weights(np.arange(len(y)), n_resamples=400)
    )

    def width(result: pd.DataFrame) -> float:
        row = result[result["metric"] == "brier"].iloc[0]
        return row["ci_high"] - row["ci_low"]

    assert width(clustered) > 1.7 * width(naive)


def test_tune_and_fit_recovers_the_signal(engagements: pd.DataFrame) -> None:
    model, tuning = tune_and_fit(engagements, "y", NUMERIC, CATEGORICAL, folds=3)
    assert len(tuning) == len(C_GRID) and tuning["chosen"].sum() == 1
    p = predict_probability(model, engagements, NUMERIC, CATEGORICAL)
    y = engagements["y"].to_numpy()
    constant = np.full(len(y), y.mean())
    assert log_loss(y, p) < log_loss(y, constant) - 0.05
    assert auc(y, p) > 0.7
    # Close to the probabilities that generated the data (5% of the signal is missing).
    assert np.corrcoef(p, engagements["p_true"])[0, 1] > 0.9


def test_bootstrap_metrics_reports_paired_differences(engagements: pd.DataFrame) -> None:
    y = engagements["y"].to_numpy()
    predictions = {
        "constant": np.full(len(y), y.mean()),
        "truth": engagements["p_true"].to_numpy(),
    }
    weights = cluster_bootstrap_weights(
        engagements.index.to_numpy(), engagements["match_id"].to_numpy(), n_resamples=200
    )
    result = bootstrap_metrics(y, predictions, weights, reference="constant")
    row = result[(result["model"] == "truth") & (result["metric"] == "log_loss_vs_constant")].iloc[
        0
    ]
    assert row["ci_low"] <= row["estimate"] <= row["ci_high"] < 0.0
    assert not (result["model"].eq("constant") & result["metric"].str.contains("_vs_")).any()
    assert (result["ci_low"] <= result["ci_high"]).all()


def test_leave_one_match_out_scores_every_match(engagements: pd.DataFrame) -> None:
    result = leave_one_match_out(engagements, "y", 1.0, NUMERIC, CATEGORICAL)
    assert sorted(result["match_id"]) == list(range(6))
    assert result["n"].sum() == len(engagements)
    assert (result["log_loss"] < result["log_loss_constant"]).all()
