"""Incremental value of pose features over the XY baseline.

Implements docs/PREREGISTRATION.md sections 7 to 11.

Two logistic regressions are compared on the pose matches with out-of-fold predictions:

- Model 0: outcome on the baseline logit ``z_xy`` (intercept and slope, no penalty).
- Model 1: the same plus the pose features, with an L2 penalty on the pose coefficients only.

scikit-learn's logistic regression penalises every coefficient, so the models are fitted
here with a small Newton solver that takes one penalty per coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from blindside.analysis.baseline import C_GRID, log_loss, logit

N_FOLDS = 10
N_REPEATS = 20
INNER_FOLDS = 5
#: Edges of the pre-registered bins of ``p_blind_side``: below, between (inclusive), above.
BIN_EDGES: tuple[float, float] = (0.1, 0.9)
BIN_LABELS: tuple[str, ...] = ("below 0.1 (in view)", "0.1 to 0.9 (uncertain)", "above 0.9 (blind)")
#: Bins with fewer engagements are reported without an interval.
MIN_BIN_SIZE = 50
_EPS = 1e-12


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -500.0, 500.0)))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    penalty: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> np.ndarray:
    """Logistic regression with one L2 penalty per coefficient, by Newton's method.

    Minimises ``sum_i w_i * loss_i + 0.5 * sum_j penalty_j * beta_j ** 2``. The intercept
    is never penalised. A penalty of ``1 / C`` on every coefficient reproduces
    scikit-learn's ``LogisticRegression(C=C)``.

    Parameters
    ----------
    x
        Features ``(n, k)`` without an intercept column.
    y
        Binary outcomes ``(n,)``.
    penalty
        Penalty per feature ``(k,)``, default no penalty.
    weight
        Observation weights ``(n,)``, default 1.

    Returns
    -------
    Array ``(k + 1,)``: intercept first, then the coefficients.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = x.shape
    design = np.column_stack([np.ones(n), x])
    lam = np.zeros(k + 1)
    if penalty is not None:
        lam[1:] = np.asarray(penalty, dtype=float)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=float)

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta
        loss = np.logaddexp(0.0, eta) - y * eta
        return float(w @ loss + 0.5 * lam @ beta**2)

    rate = np.clip(np.average(y, weights=w), _EPS, 1.0 - _EPS)
    beta = np.zeros(k + 1)
    beta[0] = np.log(rate / (1.0 - rate))
    current = objective(beta)
    for _ in range(max_iter):
        p = _sigmoid(design @ beta)
        gradient = design.T @ (w * (p - y)) + lam * beta
        hessian = (design * (w * p * (1.0 - p))[:, None]).T @ design + np.diag(lam + 1e-10)
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-6:
            candidate = beta - scale * step
            value = objective(candidate)
            if value <= current + 1e-12:
                break
            scale /= 2.0
        beta, current = candidate, value
        if np.max(np.abs(scale * step)) < tol:
            break
    return beta


def predict_logistic(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Probabilities from the output of :func:`fit_logistic`."""
    x = np.asarray(x, dtype=float)
    return _sigmoid(coefficients[0] + x @ coefficients[1:])


def grouped_folds(
    clusters: np.ndarray,
    strata: np.ndarray | None,
    n_folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fold index per observation: clusters stay together, folds are balanced within strata.

    Parameters
    ----------
    clusters
        Cluster label per observation (the pressed possession).
    strata
        Stratum label per observation (the match) or None. A cluster must lie in one stratum.
    """
    _, cluster_index = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(cluster_index.max()) + 1
    cluster_stratum = np.zeros(n_clusters, dtype=int)
    if strata is not None:
        _, stratum_index = np.unique(np.asarray(strata), return_inverse=True)
        cluster_stratum[cluster_index] = stratum_index
    cluster_fold = np.zeros(n_clusters, dtype=int)
    for stratum in np.unique(cluster_stratum):
        members = rng.permutation(np.flatnonzero(cluster_stratum == stratum))
        offset = int(rng.integers(n_folds))
        cluster_fold[members] = (np.arange(len(members)) + offset) % n_folds
    return cluster_fold[cluster_index]


def _standardise(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Impute with the training median and scale with the training mean and SD."""
    median = np.nanmedian(train, axis=0)
    train = np.where(np.isnan(train), median, train)
    other = np.where(np.isnan(other), median, other)
    mean, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (train - mean) / sd, (other - mean) / sd


def _fit_predict(
    base: np.ndarray,
    pose: np.ndarray | None,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    c: float | None = None,
) -> np.ndarray:
    """Fit on the training rows and predict the test rows. ``pose=None`` is Model 0."""
    if pose is None:
        return predict_logistic(fit_logistic(base[train], y[train]), base[test])
    pose_train, pose_test = _standardise(pose[train], pose[test])
    penalty = np.r_[np.zeros(base.shape[1]), np.full(pose.shape[1], 1.0 / c)]
    coefficients = fit_logistic(np.column_stack([base[train], pose_train]), y[train], penalty)
    return predict_logistic(coefficients, np.column_stack([base[test], pose_test]))


def choose_penalty(
    base: np.ndarray,
    pose: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    strata: np.ndarray | None,
    rng: np.random.Generator,
    c_grid: tuple[float, ...] = C_GRID,
    n_folds: int = INNER_FOLDS,
) -> float:
    """Value of ``C`` for the pose coefficients with the lowest grouped cross-validated log loss."""
    fold = grouped_folds(clusters, strata, n_folds, rng)
    losses = []
    for c in c_grid:
        p = np.empty(len(y))
        for k in range(n_folds):
            test = fold == k
            p[test] = _fit_predict(base, pose, y, ~test, test, c)
        losses.append(log_loss(y, p))
    return float(c_grid[int(np.argmin(losses))])


@dataclass(frozen=True)
class OutOfFold:
    """Out-of-fold probabilities of both models, averaged over the repeats.

    ``chosen_c`` holds the penalty chosen in every training fold (repeats times folds).
    """

    model0: np.ndarray
    model1: np.ndarray
    chosen_c: np.ndarray


def _base(z: np.ndarray, covariates: np.ndarray | None) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    if covariates is None:
        return z
    return np.column_stack([z, np.asarray(covariates, dtype=float).reshape(len(z), -1)])


def out_of_fold(
    z: np.ndarray,
    pose: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    strata: np.ndarray | None,
    covariates: np.ndarray | None = None,
    n_folds: int = N_FOLDS,
    n_repeats: int = N_REPEATS,
    c_grid: tuple[float, ...] = C_GRID,
    inner_folds: int = INNER_FOLDS,
) -> OutOfFold:
    """Repeated grouped cross-validation of Model 0 and Model 1.

    Parameters
    ----------
    z
        Baseline logit ``z_xy`` per engagement.
    pose
        Pose features ``(n, k)``. They are standardised and penalised inside each fold.
    y
        Binary outcome.
    clusters, strata
        Pressed possession and match of each engagement. Folds keep possessions together
        and are stratified by match.
    covariates
        Optional unpenalised covariates added to both models.
    n_repeats
        Repeats with seeds ``0 .. n_repeats - 1``. Predictions are averaged over repeats.
    """
    base = _base(z, covariates)
    pose = np.asarray(pose, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)
    strata = None if strata is None else np.asarray(strata)
    p0, p1 = np.zeros(len(y)), np.zeros(len(y))
    chosen = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(repeat)
        fold = grouped_folds(clusters, strata, n_folds, rng)
        for k in range(n_folds):
            test = fold == k
            train = ~test
            c = choose_penalty(
                base[train],
                pose[train],
                y[train],
                clusters[train],
                None if strata is None else strata[train],
                rng,
                c_grid,
                inner_folds,
            )
            chosen.append(c)
            p0[test] += _fit_predict(base, None, y, train, test)
            p1[test] += _fit_predict(base, pose, y, train, test, c)
    return OutOfFold(p0 / n_repeats, p1 / n_repeats, np.array(chosen))


def train_test_predictions(
    z: np.ndarray,
    pose: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    c_grid: tuple[float, ...] = C_GRID,
    inner_folds: int = INNER_FOLDS,
    seed: int = 0,
) -> OutOfFold:
    """Fit both models on the ``train`` rows and predict the ``test`` rows (leave-one-match-out).

    Returns arrays over all rows with NaN outside ``test``.
    """
    base = _base(z, None)
    pose = np.asarray(pose, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    c = choose_penalty(
        base[train], pose[train], y[train], np.asarray(clusters)[train], None, rng, c_grid,
        inner_folds,
    )  # fmt: skip
    p0, p1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    p0[test] = _fit_predict(base, None, y, train, test)
    p1[test] = _fit_predict(base, pose, y, train, test, c)
    return OutOfFold(p0, p1, np.array([c]))


def log_loss_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Negative log likelihood of each observation."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def bootstrap_mean(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean of per-observation values with a cluster bootstrap percentile interval.

    Parameters
    ----------
    values
        One value per observation, for example a paired difference of log loss terms.
    weights
        Output of :func:`blindside.analysis.baseline.cluster_bootstrap_weights`.
    mask
        Optional subset of observations. Resamples without any observation of the subset
        are dropped.

    Returns
    -------
    ``(estimate, ci_low, ci_high)``. The estimate is on the data as observed.
    """
    values = np.asarray(values, dtype=float)
    if mask is not None:
        values, weights = values[mask], weights[:, mask]
    if len(values) == 0 or np.isnan(values).any():
        return float("nan"), float("nan"), float("nan")
    total = weights.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = (weights @ values) / np.where(total > 0, total, np.nan)
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.nanpercentile(resampled, [tail, 100.0 - tail])
    return float(values.mean()), float(low), float(high)


def log_loss_gain(
    y: np.ndarray,
    oof: OutOfFold,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Primary metric: mean log loss of Model 0 minus Model 1. Positive means pose helps."""
    difference = log_loss_terms(y, oof.model0) - log_loss_terms(y, oof.model1)
    if mask is None:
        mask = np.isfinite(difference)
    return bootstrap_mean(difference, weights, mask)


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope of a logistic regression of the outcome on the predicted logit. 1 is ideal."""
    return float(fit_logistic(logit(p).reshape(-1, 1), y)[1])


def bin_index(p: np.ndarray, edges: tuple[float, float] = BIN_EDGES) -> np.ndarray:
    """Bin of each probability: 0 below the lower edge, 2 above the upper, else 1. -1 for NaN."""
    p = np.asarray(p, dtype=float)
    index = np.where(p < edges[0], 0, np.where(p > edges[1], 2, 1))
    return np.where(np.isnan(p), -1, index)


def bin_table(
    p_blind_side: np.ndarray,
    y: np.ndarray,
    expected: np.ndarray,
    weights: np.ndarray,
    edges: tuple[float, float] = BIN_EDGES,
    min_size: int = MIN_BIN_SIZE,
) -> pd.DataFrame:
    """Outcome rate per bin of ``p_blind_side``, raw and relative to Model 0.

    Parameters
    ----------
    p_blind_side
        Probability per engagement.
    y
        Outcome per engagement, binary or continuous. Rows with a missing outcome are skipped.
    expected
        Out-of-fold Model 0 probability per engagement, or NaN to skip the comparison.
    weights
        Cluster bootstrap weights.

    Returns
    -------
    One row per bin: ``bin``, ``n``, ``events`` (sum of ``y``), ``rate`` with ``rate_low``,
    ``rate_high``, ``expected`` (mean of ``expected``) and ``excess`` = observed minus
    expected with ``excess_low``, ``excess_high``. Intervals are NaN for bins below ``min_size``.
    """
    y = np.asarray(y, dtype=float)
    expected = np.asarray(expected, dtype=float)
    index = np.where(np.isnan(y), -1, bin_index(p_blind_side, edges))
    rows = []
    for k, label in enumerate(BIN_LABELS):
        mask = index == k
        n = int(mask.sum())
        row: dict[str, float | int | str] = {"bin": label, "n": n, "events": float(y[mask].sum())}
        rate = bootstrap_mean(y, weights, mask)
        excess = bootstrap_mean(y - expected, weights, mask)
        if n < min_size:
            rate, excess = (rate[0], np.nan, np.nan), (excess[0], np.nan, np.nan)
        row.update(rate=rate[0], rate_low=rate[1], rate_high=rate[2])
        row["expected"] = float(expected[mask].mean()) if n else float("nan")
        row.update(excess=excess[0], excess_low=excess[1], excess_high=excess[2])
        rows.append(row)
    return pd.DataFrame(rows)


def model1_coefficients(
    z: np.ndarray,
    pose: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    strata: np.ndarray | None,
    weights: np.ndarray,
    names: tuple[str, ...],
    c_grid: tuple[float, ...] = C_GRID,
    inner_folds: int = INNER_FOLDS,
    level: float = 0.95,
) -> tuple[pd.DataFrame, float]:
    """Model 1 fitted on all engagements, with cluster bootstrap intervals of its coefficients.

    The penalty is chosen once by grouped cross-validation on all rows and held fixed in the
    bootstrap. Pose features are standardised, so a coefficient is the change in log odds
    per standard deviation. Descriptive: the test of the increment is the out-of-fold comparison.

    Returns
    -------
    A table with ``feature``, ``coefficient``, ``ci_low``, ``ci_high``, and the chosen ``C``.
    """
    base = _base(z, None)
    pose = np.asarray(pose, dtype=float)
    y = np.asarray(y, dtype=float)
    c = choose_penalty(
        base, pose, y, clusters, strata, np.random.default_rng(0), c_grid, inner_folds
    )
    standardised, _ = _standardise(pose, pose)
    x = np.column_stack([base, standardised])
    penalty = np.r_[np.zeros(base.shape[1]), np.full(pose.shape[1], 1.0 / c)]
    point = fit_logistic(x, y, penalty)
    resampled = np.array([fit_logistic(x, y, penalty, w.astype(float)) for w in weights])
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(resampled, [tail, 100.0 - tail], axis=0)
    table = pd.DataFrame(
        {
            "feature": ["intercept", "z_xy", *names],
            "coefficient": point,
            "ci_low": low,
            "ci_high": high,
        }
    )
    return table, c
