"""XY baseline model and its evaluation (docs/PREREGISTRATION.md sections 7 to 9).

An L2-regularised logistic regression is tuned and fitted on the matches without body pose,
then frozen. Its metrics on other engagements are reported with a cluster bootstrap over
pressed possessions, because several engagements can target one possession.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from blindside.features.xy_baseline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, design_frame

#: Candidate values of ``C``, the inverse of the L2 penalty strength.
C_GRID: tuple[float, ...] = tuple(float(c) for c in np.logspace(-3, 2, 10))
TUNING_FOLDS = 5
N_BOOTSTRAP = 2000
_EPS = 1e-12

Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def make_pipeline(
    c: float = 1.0,
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> Pipeline:
    """Preprocessing and logistic regression as one estimator.

    Numeric features: median imputation with a missing indicator, then standardisation.
    Categorical features: one-hot encoding, levels unseen in training are ignored. All
    statistics come from the data the pipeline is fitted on.
    """
    transformers = []
    if numeric:
        numeric_steps = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_steps, list(numeric)))
    if categorical:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(categorical))
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("model", LogisticRegression(C=c, max_iter=2000)),
        ]
    )


def tune_and_fit(
    table: pd.DataFrame,
    outcome: str,
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
    c_grid: tuple[float, ...] = C_GRID,
    folds: int = TUNING_FOLDS,
) -> tuple[Pipeline, pd.DataFrame]:
    """Choose the penalty by grouped cross-validation over matches and fit on all rows.

    Parameters
    ----------
    table
        Training engagements with the feature columns, ``match_id`` and ``outcome``.
    outcome
        Name of the binary outcome column, without missing values.

    Returns
    -------
    The fitted pipeline with the chosen ``C``, and one row per candidate with its mean
    cross-validated log loss.
    """
    search = GridSearchCV(
        make_pipeline(numeric=numeric, categorical=categorical),
        {"model__C": list(c_grid)},
        scoring="neg_log_loss",
        cv=GroupKFold(n_splits=folds),
        refit=True,
    )
    search.fit(
        design_frame(table, numeric, categorical),
        table[outcome].to_numpy(dtype=int),
        groups=table["match_id"].to_numpy(),
    )
    tuning = pd.DataFrame(
        {
            "C": [params["model__C"] for params in search.cv_results_["params"]],
            "cv_log_loss": -search.cv_results_["mean_test_score"],
            "cv_log_loss_sd": search.cv_results_["std_test_score"],
        }
    )
    tuning["chosen"] = tuning["C"] == search.best_params_["model__C"]
    return search.best_estimator_, tuning


def predict_probability(
    model: Pipeline,
    table: pd.DataFrame,
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> np.ndarray:
    """Predicted probability of the outcome for each engagement."""
    return model.predict_proba(design_frame(table, numeric, categorical))[:, 1]


def logit(probability: np.ndarray) -> np.ndarray:
    """Log odds, with probabilities clipped away from 0 and 1."""
    p = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def log_loss(y: np.ndarray, p: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Weighted mean negative log likelihood of binary outcomes."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.average(loss, weights=weight))


def brier(y: np.ndarray, p: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Weighted mean squared difference between outcome and probability."""
    return float(np.average((np.asarray(p, dtype=float) - y) ** 2, weights=weight))


def auc(y: np.ndarray, p: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Weighted area under the ROC curve. NaN if only one class has weight."""
    keep = slice(None) if weight is None else weight > 0
    if len(np.unique(np.asarray(y)[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, sample_weight=weight))


METRICS: dict[str, Metric] = {"log_loss": log_loss, "auc": auc, "brier": brier}


def cluster_bootstrap_weights(
    clusters: np.ndarray,
    strata: np.ndarray | None = None,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = 0,
) -> np.ndarray:
    """Resampling weights of a cluster bootstrap.

    Clusters are drawn with replacement, within each stratum if ``strata`` is given. The
    weight of an observation is the number of times its cluster was drawn.

    Parameters
    ----------
    clusters
        Cluster label per observation.
    strata
        Stratum label per observation (for example the match). A cluster must lie in one stratum.

    Returns
    -------
    Integer array ``(n_resamples, observations)``.
    """
    rng = np.random.default_rng(seed)
    _, cluster_index = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(cluster_index.max()) + 1
    if strata is None:
        cluster_stratum = np.zeros(n_clusters, dtype=int)
    else:
        _, stratum_index = np.unique(np.asarray(strata), return_inverse=True)
        cluster_stratum = np.zeros(n_clusters, dtype=int)
        cluster_stratum[cluster_index] = stratum_index
    counts = np.zeros((n_resamples, n_clusters), dtype=np.int32)
    for stratum in np.unique(cluster_stratum):
        members = np.flatnonzero(cluster_stratum == stratum)
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        for r in range(n_resamples):
            counts[r, members] = np.bincount(draws[r], minlength=len(members))
    return counts[:, cluster_index]


def bootstrap_metrics(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: np.ndarray,
    reference: str | None = None,
    level: float = 0.95,
) -> pd.DataFrame:
    """Metrics of one or more prediction vectors with cluster bootstrap intervals.

    Parameters
    ----------
    y
        Binary outcomes.
    predictions
        ``{name: probabilities}``, all aligned with ``y``.
    weights
        Output of :func:`cluster_bootstrap_weights`. The same resamples are used for every
        prediction, so differences are paired.
    reference
        Name of a prediction. If given, each other prediction also gets rows
        ``{metric}_vs_{reference}``: its metric minus the metric of the reference.

    Returns
    -------
    One row per prediction and metric with ``estimate`` (on the data as observed) and the
    percentile interval ``ci_low``, ``ci_high``.
    """
    y = np.asarray(y, dtype=float)
    tail = 100.0 * (1.0 - level) / 2.0
    point = {
        (name, metric): function(y, p, None)
        for name, p in predictions.items()
        for metric, function in METRICS.items()
    }
    resampled = {
        key: np.array([METRICS[key[1]](y, predictions[key[0]], w) for w in weights])
        for key in point
    }
    rows = []
    for (name, metric), estimate in point.items():
        low, high = np.nanpercentile(resampled[(name, metric)], [tail, 100.0 - tail])
        rows.append(
            {"model": name, "metric": metric, "estimate": estimate, "ci_low": low, "ci_high": high}
        )
        if reference is not None and name != reference:
            difference = resampled[(name, metric)] - resampled[(reference, metric)]
            low, high = np.nanpercentile(difference, [tail, 100.0 - tail])
            rows.append(
                {
                    "model": name,
                    "metric": f"{metric}_vs_{reference}",
                    "estimate": estimate - point[(reference, metric)],
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def leave_one_match_out(
    table: pd.DataFrame,
    outcome: str,
    c: float,
    numeric: tuple[str, ...] = NUMERIC_FEATURES,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """Fit on all matches but one and score the held-out match, for every match.

    The penalty ``c`` is held fixed. The constant prediction of each fold is the outcome
    rate of its training matches.

    Returns
    -------
    One row per held-out match: ``n``, ``events``, ``rate``, the three metrics of the model
    and ``log_loss_constant``, ``brier_constant`` of the constant prediction.
    """
    match = table["match_id"].to_numpy()
    y = table[outcome].to_numpy(dtype=float)
    design = design_frame(table, numeric, categorical)
    rows = []
    for held_out in np.unique(match):
        test = match == held_out
        model = make_pipeline(c, numeric, categorical).fit(design[~test], y[~test].astype(int))
        p = model.predict_proba(design[test])[:, 1]
        constant = np.full(test.sum(), y[~test].mean())
        rows.append(
            {
                "match_id": int(held_out),
                "n": int(test.sum()),
                "events": int(y[test].sum()),
                "rate": float(y[test].mean()),
                "log_loss": log_loss(y[test], p),
                "auc": auc(y[test], p),
                "brier": brier(y[test], p),
                "log_loss_constant": log_loss(y[test], constant),
                "brier_constant": brier(y[test], constant),
            }
        )
    return pd.DataFrame(rows)
