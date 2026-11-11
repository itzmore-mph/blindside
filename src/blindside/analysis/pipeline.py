"""The steps between the cached tables and the reported results.

Engagement tables with the inclusion criteria of docs/PREREGISTRATION.md, the pose attached
to each included engagement, the pre-registered comparison of Model 0 and Model 1 with its
robustness checks, and the exploratory agreement of the XY proxy with the pose label.
The scripts in ``scripts/`` render these results into ``docs/``, and
``blindside reproduce`` draws the README figures from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from blindside.analysis.agreement import agreement_table
from blindside.analysis.baseline import (
    N_BOOTSTRAP,
    bootstrap_metrics,
    cluster_bootstrap_weights,
    logit,
    predict_probability,
    tune_and_fit,
)
from blindside.analysis.bracket import add_residuals, bracket_summary
from blindside.analysis.camera import cached_camera_estimate
from blindside.analysis.consistency import SPEED_EDGES, speed_band_labels
from blindside.analysis.incremental import (
    MIN_BIN_SIZE,
    OutOfFold,
    bin_index,
    bin_table,
    calibration_slope,
    log_loss_gain,
    log_loss_terms,
    model1_coefficients,
    out_of_fold,
    train_test_predictions,
)
from blindside.analysis.reliability import (
    MIN_ZONE_POSES,
    pitch_zone_summary,
    standard_xy,
    touchline_distance,
    velocity_agreement_by_band,
)
from blindside.data.download import download_pose_archive
from blindside.data.events import load_match_metadata
from blindside.data.manifest import pose_match_ids
from blindside.data.paths import cache_dir
from blindside.data.tracking import tracking_match_ids
from blindside.features.engagements import POSSESSION_LABELS, PRIMARY_OUTCOME, pose_available
from blindside.features.orientation_table import (
    OrientationConfig,
    cached_orientation_table,
    orientation_cache_path,
)
from blindside.features.pose_features import (
    BOUNDS,
    FOV_HALF_ANGLE_DEG,
    POSE_FEATURES,
    cached_engagement_poses,
    pose_feature_table,
    select_pose_frames,
)
from blindside.features.xy_baseline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    baseline_table,
    design_frame,
)
from blindside.geometry.uncertainty import P90_TO_SIGMA

USABLE_SD_DEG = 20.0
SLOW_CARRIER_MPS = 2.0
PHASE_TYPE = "team_out_of_possession_phase_type"
NO_LABELS = tuple(name for name in CATEGORICAL_FEATURES if name not in POSSESSION_LABELS)
NO_LEAK = tuple(name for name in NO_LABELS if name != PHASE_TYPE)
SECONDARY_BINARY = {"y_turnover_wide": "Wide turnover", "y_force_backward": "Forced backward"}
SEGMENTS = ("head", "torso", "hips")
BASELINE_NAMES = {
    "3 Baseline without SkillCorner labels": "without SkillCorner labels (check 3)",
    "7 Gradient-boosted baseline": "gradient-boosted (check 7)",
    "9 Baseline without phase type and labels": "without phase type and labels (check 9)",
}
#: Effectively no penalty, for the exploratory fit of Model 1 without shrinkage.
NO_PENALTY_C = 1e6
#: Exploratory binary pose label: blind if more than this share of the samples is blind.
LABEL_THRESHOLD = 0.5
BLIND_FEATURES = ("p_blind_side_head", "p_blind_side_torso")
ALL_ROW = "all"
CONFIDENT_ROW = "all, uncertain bin dropped"


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def orientation_table_path(match_id: int) -> Path:
    """Parquet path of the default orientation table of a pose match, built on first use."""
    archive = download_pose_archive(match_id)
    path = orientation_cache_path(archive, OrientationConfig())
    if not path.exists():
        cached_orientation_table(archive, match_id, OrientationConfig())
    return path


def _pose_match_table(match_id: int) -> pd.DataFrame:
    """Engagements of one pose match with the XY features and the flag ``has_pose``."""
    table = baseline_table(match_id)
    keys = pd.read_parquet(orientation_table_path(match_id), columns=["frame", "player_id"])
    table["has_pose"] = pose_available(table, keys)
    return table


def load_test_table() -> pd.DataFrame:
    """Engagement table of the pose matches only, before inclusion.

    The same rows as the second table of :func:`load_tables`, without touching the tracking
    of the matches that have no body pose.
    """
    pose_ids = set(pose_match_ids())
    return pd.concat(
        [_pose_match_table(m) for m in tracking_match_ids() if m in pose_ids], ignore_index=True
    )


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engagement tables of the training matches and of the pose matches, before inclusion."""
    pose_ids = set(pose_match_ids())
    train, test = [], []
    for match_id in tracking_match_ids():
        if match_id in pose_ids:
            test.append(_pose_match_table(match_id))
        else:
            train.append(baseline_table(match_id))
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)


def inclusion_flow(table: pd.DataFrame, with_pose: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the inclusion criteria in order. Returns the included rows and the flow table."""
    steps = [("All on-ball engagements", np.ones(len(table), dtype=bool))]
    if with_pose:
        steps.append(("1. Pose for presser and carrier at t0", table["has_pose"].to_numpy()))
    steps.append(("2. Primary outcome defined", table[PRIMARY_OUTCOME].notna().to_numpy()))
    steps.append(("3. Tracking at t0 and 0.5 s before", table["has_tracking"].to_numpy()))
    keep = np.ones(len(table), dtype=bool)
    rows = []
    for label, mask in steps:
        keep &= mask
        rows.append(
            {
                "Criterion": label,
                "Engagements": int(keep.sum()),
                "Possessions": int(table.loc[keep, "cluster"].nunique()),
            }
        )
    return table[keep].reset_index(drop=True), pd.DataFrame(rows)


def attach_pose(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the pose frame, the flip flags and the camera distance. Returns the poses too."""
    test = test.copy()
    test["pose_frame"] = -1
    test["camera_distance_m"] = np.nan
    for player in ("carrier", "presser"):
        for segment in SEGMENTS:
            test[f"{player}_{segment}_flip"] = False
    poses = []
    for match_id in pose_match_ids():
        rows = test["match_id"] == match_id
        part = test[rows]
        archive = download_pose_archive(match_id)
        flips = pd.read_parquet(
            orientation_table_path(match_id),
            columns=[
                "frame",
                "player_id",
                "head_deg",
                "torso_deg",
                *(f"{segment}_flip" for segment in SEGMENTS),
            ],
        )
        frames = select_pose_frames(part, flips)
        if (frames < 0).any():
            raise RuntimeError(f"{match_id}: included engagements without a pose frame")
        test.loc[rows, "pose_frame"] = frames
        keys = set(zip(frames.tolist(), part["presser_id"].tolist(), strict=True)) | set(
            zip(frames.tolist(), part["carrier_id"].tolist(), strict=True)
        )
        match_poses = cached_engagement_poses(archive, keys, match_id)
        poses.append(match_poses)

        flips = flips.set_index(["frame", "player_id"])
        for player in ("carrier", "presser"):
            index = pd.MultiIndex.from_arrays([frames, part[f"{player}_id"].to_numpy()])
            flagged = flips.reindex(index)
            for segment in SEGMENTS:
                test.loc[rows, f"{player}_{segment}_flip"] = (
                    flagged[f"{segment}_flip"].fillna(False).to_numpy(dtype=bool)
                )
            if player == "carrier":
                for segment in ("head", "torso"):
                    test.loc[rows, f"table_{segment}_deg"] = flagged[f"{segment}_deg"].to_numpy()
        carrier_y = (
            match_poses.set_index(["frame", "player_id"])
            .reindex(pd.MultiIndex.from_arrays([frames, part["carrier_id"].to_numpy()]))["y"]
            .to_numpy()
        )
        width = float(load_match_metadata(match_id)["pitch_width"])
        side = cached_camera_estimate(archive, match_id).side
        test.loc[rows, "camera_distance_m"] = touchline_distance(carrier_y, width, side)
    return test, pd.concat(poses, ignore_index=True)


def boosted_logit(train: pd.DataFrame, test: pd.DataFrame, outcome: str) -> np.ndarray:
    """Baseline logit from a gradient-boosted model with default settings (robustness check 7)."""
    model = Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("numeric", "passthrough", list(NUMERIC_FEATURES)),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            list(CATEGORICAL_FEATURES),
                        ),
                    ]
                ),
            ),
            ("model", HistGradientBoostingClassifier(random_state=0)),
        ]
    )
    model.fit(design_frame(train), train[outcome].to_numpy(dtype=int))
    return logit(model.predict_proba(design_frame(test))[:, 1])


def logistic_logit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str,
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES,
) -> np.ndarray:
    """Baseline logit of the tuned logistic baseline, refitted on the training matches."""
    keep = train[outcome].notna()
    model, _ = tune_and_fit(train[keep], outcome, categorical=categorical)
    return logit(predict_probability(model, test, categorical=categorical))


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


def comparison_row(
    label: str,
    variant: str,
    bound: str,
    y: np.ndarray,
    oof: OutOfFold,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, object]:
    """One line of a results table: sample, both log losses and the paired difference."""
    keep = np.isfinite(oof.model1) if mask is None else mask
    gain = log_loss_gain(y, oof, weights, keep)
    return {
        "check": label,
        "variant": variant,
        "bound": bound,
        "n": int(keep.sum()),
        "events": int(y[keep].sum()),
        "loss0": float(log_loss_terms(y[keep], oof.model0[keep]).mean()),
        "loss1": float(log_loss_terms(y[keep], oof.model1[keep]).mean()),
        "gain": gain[0],
        "low": gain[1],
        "high": gain[2],
        "median_c": float(np.median(oof.chosen_c)) if len(oof.chosen_c) else np.nan,
    }


def run_subset(
    z: np.ndarray,
    features: pd.DataFrame,
    y: np.ndarray,
    test: pd.DataFrame,
    weights: np.ndarray,
    subset: np.ndarray | None = None,
    covariates: np.ndarray | None = None,
) -> tuple[OutOfFold, np.ndarray]:
    """Out-of-fold comparison, refitted on a subset if given. Returns predictions over all rows."""
    subset = np.ones(len(y), dtype=bool) if subset is None else subset
    part = out_of_fold(
        z[subset],
        features.loc[subset, list(POSE_FEATURES)].to_numpy(),
        y[subset],
        test.loc[subset, "cluster"].to_numpy(),
        test.loc[subset, "match_id"].to_numpy(),
        None if covariates is None else covariates[subset],
    )
    p0, p1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    p0[subset], p1[subset] = part.model0, part.model1
    return OutOfFold(p0, p1, part.chosen_c), subset


def bin_contrast(
    p: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, int, int]:
    """Mean of ``values`` in the blind bin minus the in-view bin, with a bootstrap interval."""
    index = bin_index(p)
    blind, view = index == 2, index == 0
    if blind.sum() < MIN_BIN_SIZE or view.sum() < MIN_BIN_SIZE:
        estimate = (
            values[blind].mean() - values[view].mean() if blind.any() and view.any() else np.nan
        )
        return float(estimate), np.nan, np.nan, int(blind.sum()), int(view.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = (weights[:, blind] @ values[blind]) / weights[:, blind].sum(axis=1) - (
            weights[:, view] @ values[view]
        ) / weights[:, view].sum(axis=1)
    low, high = np.nanpercentile(resampled, [2.5, 97.5])
    estimate = values[blind].mean() - values[view].mean()
    return float(estimate), float(low), float(high), int(blind.sum()), int(view.sum())


def proxy_agreement(
    test: pd.DataFrame, features: dict[str, pd.DataFrame], weights: np.ndarray
) -> pd.DataFrame:
    """Exploratory: how well the XY proxy reproduces the pose-based blind-side label.

    The proxy calls an engagement blind if the angle between the carrier's direction of
    travel and the direction to the presser exceeds the field of view half-angle. The pose
    label is ``p_blind_side`` above :data:`LABEL_THRESHOLD`. A second row keeps the
    pre-registered bins and drops the uncertain one. Reads ``approach_angle_heading_deg``
    and ``carrier_speed`` from ``test`` and nothing else: no outcome column is used.
    """
    heading = test["approach_angle_heading_deg"].to_numpy(dtype=float)
    speed = test["carrier_speed"].to_numpy(dtype=float)
    proxy = heading > FOV_HALF_ANGLE_DEG
    band = np.digitize(speed, SPEED_EDGES[1:-1])
    bands = {label: band == k for k, label in enumerate(speed_band_labels(SPEED_EDGES))}
    tables = []
    for feature in BLIND_FEATURES:
        for bound in BOUNDS:
            p = features[bound][feature].to_numpy(dtype=float)
            defined = np.isfinite(p) & np.isfinite(heading) & np.isfinite(speed)
            primary = agreement_table(
                p > LABEL_THRESHOLD, proxy, weights, bands, keep=defined, min_size=MIN_BIN_SIZE
            )
            index = bin_index(p)
            confident = agreement_table(
                index == 2, proxy, weights, keep=defined & (index != 1), min_size=MIN_BIN_SIZE
            ).assign(group=CONFIDENT_ROW)
            tables.append(
                pd.concat([primary, confident], ignore_index=True).assign(
                    feature=feature, bound=bound
                )
            )
    return pd.concat(tables, ignore_index=True)


def compute(train: pd.DataFrame, test: pd.DataFrame, poses: pd.DataFrame) -> dict[str, object]:
    """Run every analysis. Returns tables and scalars for :func:`render`."""
    y = test[PRIMARY_OUTCOME].to_numpy(dtype=float)
    clusters, matches = test["cluster"].to_numpy(), test["match_id"].to_numpy()
    weights = cluster_bootstrap_weights(clusters, matches, N_BOOTSTRAP, seed=0)
    out: dict[str, object] = {"n": len(test), "events": int(y.sum())}

    def features_for(
        bound: str, half_angle: float = FOV_HALF_ANGLE_DEG, factor: float = P90_TO_SIGMA
    ):
        return pose_feature_table(
            test, poses, bound=bound, half_angle_deg=half_angle, p90_to_sigma=factor
        )

    features = {bound: features_for(bound) for bound in BOUNDS}
    out["features"] = features
    for segment in ("head", "torso"):
        gap = features["upper"][f"carrier_{segment}_deg"] - test[f"table_{segment}_deg"]
        if np.nanmax(np.abs((gap + 180.0) % 360.0 - 180.0)) > 0.01:
            raise RuntimeError(f"carrier {segment} angle differs from the orientation table")

    print("baselines ...", flush=True)
    z = logistic_logit(train, test, PRIMARY_OUTCOME)
    cached = cache_dir() / "baseline" / "test_predictions.parquet"
    if cached.exists():
        stored = pd.read_parquet(cached, columns=["event_id", "z_xy"])
        if not (
            list(stored["event_id"]) == list(test["event_id"])
            and np.allclose(stored["z_xy"].to_numpy(), z, atol=1e-6)
        ):
            raise RuntimeError("the refitted baseline differs from the Phase 3 baseline")
    out["z"] = z

    # Primary comparison.
    print("primary comparison ...", flush=True)
    primary, primary_rows, metric_rows, coefficient_tables = {}, [], [], {}
    for bound in BOUNDS:
        oof, _ = run_subset(z, features[bound], y, test, weights)
        primary[bound] = oof
        primary_rows.append(comparison_row("Primary", "all included", bound, y, oof, weights))
        metrics = bootstrap_metrics(
            y, {"model0": oof.model0, "model1": oof.model1}, weights, reference="model1"
        )
        for metric in ("log_loss", "auc", "brier"):
            row0 = metrics[(metrics["model"] == "model0") & (metrics["metric"] == metric)].iloc[0]
            row1 = metrics[(metrics["model"] == "model1") & (metrics["metric"] == metric)].iloc[0]
            diff = metrics[
                (metrics["model"] == "model0") & (metrics["metric"] == f"{metric}_vs_model1")
            ].iloc[0]
            metric_rows.append(
                {
                    "bound": bound,
                    "metric": metric,
                    "model0": row0["estimate"],
                    "model1": row1["estimate"],
                    "difference": diff["estimate"],
                    "low": diff["ci_low"],
                    "high": diff["ci_high"],
                }
            )
        metric_rows.append(
            {
                "bound": bound,
                "metric": "calibration_slope",
                "model0": calibration_slope(y, oof.model0),
                "model1": calibration_slope(y, oof.model1),
                "difference": np.nan,
                "low": np.nan,
                "high": np.nan,
            }
        )
        coefficient_tables[bound] = model1_coefficients(
            z,
            features[bound][list(POSE_FEATURES)].to_numpy(),
            y,
            clusters,
            matches,
            weights,
            POSE_FEATURES,
        )
    out["primary"] = primary
    out["primary_rows"] = pd.DataFrame(primary_rows)
    out["metric_rows"] = pd.DataFrame(metric_rows)
    out["coefficients"] = coefficient_tables

    # Descriptive bins (section 11 and addition 1).
    bins = {}
    for bound in BOUNDS:
        for feature in ("p_blind_side_head", "p_blind_side_torso"):
            bins[(feature, bound)] = bin_table(
                features[bound][feature].to_numpy(), y, primary[bound].model0, weights
            )
    out["bins"] = bins

    # Robustness checks.
    checks: list[dict[str, object]] = []

    def add(label: str, variant: str, bound: str, oof: OutOfFold, mask=None) -> None:
        checks.append(comparison_row(label, variant, bound, y, oof, weights, mask))

    print("robustness checks ...", flush=True)
    match_ids = sorted(np.unique(matches).tolist())
    for bound in BOUNDS:
        pose = features[bound][list(POSE_FEATURES)].to_numpy()
        for held_out in match_ids:
            test_rows = matches == held_out
            oof = train_test_predictions(z, pose, y, clusters, ~test_rows, test_rows)
            fitted_on = next(m for m in match_ids if m != held_out)
            add("1 Leave one match out", f"fit {fitted_on}, test {held_out}", bound, oof)

    upper = features["upper"]
    usable = (
        (upper["carrier_head_sd_deg"] < USABLE_SD_DEG)
        & (upper["carrier_torso_sd_deg"] < USABLE_SD_DEG)
        & (upper["presser_hips_sd_deg"] < USABLE_SD_DEG)
    ).to_numpy()
    out["n_usable"] = int(usable.sum())
    for bound in BOUNDS:
        oof, _ = run_subset(z, features[bound], y, test, weights, usable)
        add("2 Usability subset", f"all three SD below {USABLE_SD_DEG:g} deg", bound, oof)

    baselines = {
        "3 Baseline without SkillCorner labels": logistic_logit(
            train, test, PRIMARY_OUTCOME, NO_LABELS
        ),
        "7 Gradient-boosted baseline": boosted_logit(train, test, PRIMARY_OUTCOME),
        "9 Baseline without phase type and labels": logistic_logit(
            train, test, PRIMARY_OUTCOME, NO_LEAK
        ),
    }
    baseline_scores = [
        {
            "baseline": "registered",
            "log_loss": float(log_loss_terms(y, 1 / (1 + np.exp(-z))).mean()),
        }
    ]
    for label, z_alt in baselines.items():
        baseline_scores.append(
            {
                "baseline": BASELINE_NAMES[label],
                "log_loss": float(log_loss_terms(y, 1 / (1 + np.exp(-z_alt))).mean()),
            }
        )
        for bound in BOUNDS:
            oof, _ = run_subset(z_alt, features[bound], y, test, weights)
            add(label, "baseline refitted on 18 matches", bound, oof)
    out["baseline_scores"] = pd.DataFrame(baseline_scores)

    for half_angle in (90.0, 110.0):
        for bound in BOUNDS:
            oof, _ = run_subset(z, features_for(bound, half_angle), y, test, weights)
            add("4 Field of view", f"half-angle {half_angle:g} deg", bound, oof)
    for factor in (2.0, 3.0):
        oof, _ = run_subset(z, features_for("upper", factor=factor), y, test, weights)
        add("5 Sigma factor", f"sigma = p90 / {factor:g}", "upper", oof)

    before = test["starts_before_possession"].to_numpy(dtype=bool)
    slow = test["carrier_speed"].to_numpy(dtype=float) < SLOW_CARRIER_MPS
    for bound in BOUNDS:
        add("6 Engagement timing", "starts before the possession", bound, primary[bound], before)
        add("6 Engagement timing", "starts at or after it", bound, primary[bound], ~before)
    camera = test["camera_distance_m"].to_numpy(dtype=float)
    for bound in BOUNDS:
        oof, _ = run_subset(z, features[bound], y, test, weights, covariates=camera)
        add("8 Camera distance covariate", "added to both models", bound, oof)
    for bound in BOUNDS:
        add("10 Carrier speed", f"below {SLOW_CARRIER_MPS:g} m/s", bound, primary[bound], slow)
        add(
            "10 Carrier speed",
            f"at or above {SLOW_CARRIER_MPS:g} m/s",
            bound,
            primary[bound],
            ~slow,
        )
    order = {label: int(label.split()[0]) for label in {row["check"] for row in checks}}
    out["checks"] = (
        pd.DataFrame(checks)
        .assign(order=lambda frame: frame["check"].map(order))
        .sort_values("order", kind="stable")
        .drop(columns="order")
        .reset_index(drop=True)
    )

    # Secondary outcomes.
    print("secondary outcomes ...", flush=True)
    secondary, secondary_bins = [], {}
    for outcome, label in SECONDARY_BINARY.items():
        y_secondary = test[outcome].to_numpy(dtype=float)
        z_secondary = logistic_logit(train, test, outcome)
        for bound in BOUNDS:
            oof, _ = run_subset(z_secondary, features[bound], y_secondary, test, weights)
            secondary.append(comparison_row(label, outcome, bound, y_secondary, oof, weights))
            secondary_bins[(outcome, bound)] = bin_table(
                features[bound]["p_blind_side_head"].to_numpy(), y_secondary, oof.model0, weights
            )
    epv = test["possession_epv_total"].to_numpy(dtype=float)
    for bound in BOUNDS:
        secondary_bins[("possession_epv_total", bound)] = bin_table(
            features[bound]["p_blind_side_head"].to_numpy(),
            epv,
            np.full(len(epv), np.nan),
            weights,
        )
    out["secondary"] = pd.DataFrame(secondary)
    out["secondary_bins"] = secondary_bins

    # Exploratory.
    contrasts = []
    for bound in BOUNDS:
        for feature in ("p_blind_side_head", "p_blind_side_torso"):
            p = features[bound][feature].to_numpy()
            rate = bin_contrast(p, y, weights)
            excess = bin_contrast(p, y - primary[bound].model0, weights)
            contrasts.append(
                {
                    "feature": feature,
                    "bound": bound,
                    "n_blind": rate[3],
                    "n_view": rate[4],
                    "rate": rate[:3],
                    "excess": excess[:3],
                }
            )
    out["contrasts"] = contrasts
    unpenalised_rows, unpenalised_coefficients = [], {}
    for bound in BOUNDS:
        pose = features[bound][list(POSE_FEATURES)].to_numpy()
        oof = out_of_fold(z, pose, y, clusters, matches, c_grid=(NO_PENALTY_C,))
        unpenalised_rows.append(
            comparison_row("Model 1 without penalty", "", bound, y, oof, weights)
        )
        unpenalised_coefficients[bound], _ = model1_coefficients(
            z, pose, y, clusters, matches, weights, POSE_FEATURES, c_grid=(NO_PENALTY_C,)
        )
    out["unpenalised_rows"] = pd.DataFrame(unpenalised_rows)
    out["unpenalised_coefficients"] = unpenalised_coefficients
    out["proxy_agreement"] = proxy_agreement(test, features, weights)
    out["weights"] = weights
    return out


def decision(rows: pd.DataFrame) -> str:
    """Wording of the pre-registered decision rule for the primary comparison."""
    above = {row["bound"]: row["low"] > 0 for _, row in rows.iterrows()}
    if all(above.values()):
        return "H1 is supported: the 95% interval lies above zero under both uncertainty bounds."
    if any(above.values()):
        bound = next(b for b, value in above.items() if value)
        return (
            f"The result depends on the error model: the 95% interval lies above zero under the "
            f"{bound} bound only."
        )
    return (
        "No detectable increment: the 95% interval of the log loss difference includes zero or "
        "lies below it under both uncertainty bounds."
    )


# --------------------------------------------------------------------------------------
# Reliability map
# --------------------------------------------------------------------------------------


def pose_match_orientation_tables() -> tuple[pd.DataFrame, dict[int, tuple[float, float]], str]:
    """Orientation tables of all pose matches in one frame, ready for the reliability map.

    Returns
    -------
    table
        The cached orientation tables, concatenated, with ``touchline_distance`` added
        (metres from the touchline behind which the main camera stands).
    pitch_dims
        ``{match_id: (pitch_length, pitch_width)}`` from the match metadata.
    camera_side
        ``"-y"`` or ``"+y"``, the side shared by all matches.

    Raises
    ------
    RuntimeError
        If the camera side of a match cannot be inferred reliably, or differs between matches.
    """
    tables, pitch_dims, sides = [], {}, set()
    for match_id in pose_match_ids():
        archive = download_pose_archive(match_id)
        table = cached_orientation_table(archive, match_id, OrientationConfig())
        metadata = load_match_metadata(match_id)
        pitch_dims[match_id] = (float(metadata["pitch_length"]), float(metadata["pitch_width"]))
        camera = cached_camera_estimate(archive, match_id)
        if not camera.reliable:
            raise RuntimeError(f"camera side of match {match_id} could not be inferred reliably")
        sides.add(camera.side)
        table["touchline_distance"] = touchline_distance(
            table["y"].to_numpy(), pitch_dims[match_id][1], camera.side
        ).astype(np.float32)
        tables.append(table)
    if len(sides) != 1:
        raise RuntimeError("the camera side differs between matches, zones cannot be pooled as is")
    return pd.concat(tables, ignore_index=True), pitch_dims, sides.pop()


@dataclass(frozen=True)
class ReliabilityInputs:
    """The summaries behind the reliability figures, see :func:`reliability_inputs`."""

    with_residuals: pd.DataFrame
    zones: pd.DataFrame
    velocity: pd.DataFrame

    def composite(self) -> dict[str, pd.DataFrame]:
        """Data arguments of :func:`blindside.viz.reliability.plot_reliability_composite`."""
        return {
            "speed_bracket": bracket_summary(self.with_residuals, "speed", SPEED_EDGES),
            "zone_summary": self.zones,
            "velocity": self.velocity,
        }


def reliability_inputs(
    table: pd.DataFrame,
    pitch_dims: Mapping[int, tuple[float, float]],
    min_zone_poses: int = MIN_ZONE_POSES,
) -> ReliabilityInputs:
    """Jitter residuals, the pitch zone summary and the velocity check of an orientation table.

    Parameters
    ----------
    table
        One or more orientation tables, see
        :func:`blindside.features.orientation_table.build_orientation_table`.
    pitch_dims
        ``{match_id: (pitch_length, pitch_width)}`` in metres.
    min_zone_poses
        Zones with fewer poses are left empty. Lower it for a short clip.
    """
    with_residuals = add_residuals(table)
    x, y = standard_xy(with_residuals, pitch_dims)
    zones = pitch_zone_summary(with_residuals, x, y, min_poses=min_zone_poses)
    return ReliabilityInputs(with_residuals, zones, velocity_agreement_by_band(table))
