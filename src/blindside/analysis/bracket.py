"""Bracket the angular uncertainty between a jitter lower bound and the Monte Carlo upper bound.

There is no ground truth for body orientation, so the angular error cannot be measured.
It can be bracketed:

* **Upper bound.** The Monte Carlo SD of the orientation table, which treats the errors of
  the two joints of a segment as independent. Positive correlation between them, which is
  likely, makes the true angular error smaller.
* **Lower bound.** The high-frequency noise that is visible in the angle time series itself:
  the residual of each angle from a local quadratic fit over a short window. Whatever part
  of the error is smooth in time (a pose model that filters, a bias that drifts slowly) is
  invisible to this estimate, so the true angular error is at least this large.

Both bounds rest on assumptions (the published ``p90_mae_cm`` is calibrated, the residual
noise is white within the window, real movement is locally quadratic). The true error lies
between them and cannot be identified from this data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from blindside.analysis.consistency import SEGMENTS, SPEED_EDGES, speed_band_labels
from blindside.data.manifest import POSE_FPS

#: Half windows in pose frames. 2 gives 5 frames (0.2 s), 5 gives 11 frames (0.4 s span).
HALF_WINDOWS: dict[str, int] = {"0.2 s": 2, "0.4 s": 5}
#: The window whose result is carried forward as the lower bound.
PRIMARY_WINDOW = "0.2 s"
AUTOCORRELATION_LAGS: tuple[int, ...] = (1, 5, 25)
#: Converts a median absolute deviation into the SD of a Gaussian.
MAD_TO_SD = 1.4826
#: Ground distance to the inferred camera position, band edges in metres.
CAMERA_EDGES: tuple[float, ...] = (0.0, 60.0, 75.0, 90.0, np.inf)


def quadratic_smoother(half_window: int) -> np.ndarray:
    """Weights that give the value of a least-squares quadratic at the window centre.

    For a window of ``2 * half_window + 1`` equally spaced points, ``weights @ values`` is
    the fitted value at the centre (the Savitzky-Golay smoothing filter of order 2).
    """
    if half_window < 2:
        raise ValueError("a quadratic fit with a residual needs half_window >= 2")
    offsets = np.arange(-half_window, half_window + 1, dtype=float)
    design = np.vander(offsets, 3, increasing=True)
    return np.linalg.pinv(design)[0]


def residual_variance_factor(half_window: int) -> float:
    """Variance of the centre residual for unit white noise: ``1 - weight_of_centre``."""
    return float(1.0 - quadratic_smoother(half_window)[half_window])


def white_noise_autocorrelation(half_window: int, lag: int) -> float:
    """Autocorrelation that the residual filter itself induces on white noise at ``lag``.

    Residuals of neighbouring frames share inputs, so they are correlated even when the
    underlying noise is not. Observed autocorrelations are read against this reference.
    """
    kernel = -quadratic_smoother(half_window)
    kernel[half_window] += 1.0
    if lag >= len(kernel):
        return 0.0
    return float(kernel[lag:] @ kernel[: len(kernel) - lag] / (kernel @ kernel))


def sort_for_time_series(table: pd.DataFrame) -> pd.DataFrame:
    """Rows ordered by match, player and frame, index reset."""
    return table.sort_values(["match_id", "player_id", "frame"], kind="stable").reset_index(
        drop=True
    )


def _same_run(ordered: pd.DataFrame, lag: int) -> np.ndarray:
    """Mask of length ``n - lag``: row ``i + lag`` is the same player ``lag`` frames later."""
    frame = ordered["frame"].to_numpy()
    player = ordered["player_id"].to_numpy()
    match = ordered["match_id"].to_numpy()
    return (
        (frame[lag:] - frame[:-lag] == lag)
        & (player[lag:] == player[:-lag])
        & (match[lag:] == match[:-lag])
    )


def local_quadratic_residual(
    ordered: pd.DataFrame, column: str, half_window: int, chunk: int = 500_000
) -> np.ndarray:
    """Residual of an angle from a local quadratic fit, in degrees.

    Parameters
    ----------
    ordered
        Table from :func:`sort_for_time_series`.
    column
        Angle column in degrees.
    half_window
        The fit uses the ``2 * half_window + 1`` consecutive frames centred on each row.
        The angle is unwrapped inside the window, so a crossing of +/-180 degrees is not
        mistaken for a jump.

    Returns
    -------
    Array aligned with ``ordered``. NaN where the window is not a gap-free run of one
    player or holds a NaN angle.
    """
    angle = ordered[column].to_numpy(dtype=float)
    n, width = len(angle), 2 * half_window + 1
    residual = np.full(n, np.nan)
    if n < width:
        return residual
    full_run = _same_run(ordered, 2 * half_window)  # row i is the first row of the window
    weights = quadratic_smoother(half_window)
    windows = np.lib.stride_tricks.sliding_window_view(angle, width)
    for start in range(0, len(windows), chunk):
        block = windows[start : start + chunk]
        steps = np.mod(np.diff(block, axis=1) + 180.0, 360.0) - 180.0
        unwrapped = np.concatenate([np.zeros((len(block), 1)), np.cumsum(steps, axis=1)], axis=1)
        centre = unwrapped[:, half_window] - unwrapped @ weights
        centre[~full_run[start : start + chunk]] = np.nan
        residual[start + half_window : start + half_window + len(block)] = centre
    return residual


def robust_sd(values: np.ndarray) -> float:
    """SD estimate from the median absolute deviation about zero. NaN values are ignored."""
    values = np.abs(values[np.isfinite(values)])
    return float(MAD_TO_SD * np.median(values)) if len(values) else float("nan")


def jitter_sd(residual: np.ndarray, half_window: int) -> float:
    """White-noise SD of the angle that explains the spread of the residuals.

    The residual of white noise with SD ``s`` has SD ``s * sqrt(residual_variance_factor)``.
    The spread is measured robustly, so the occasional sharp real turn does not count as noise.
    """
    return robust_sd(residual) / np.sqrt(residual_variance_factor(half_window))


def bound_ratio(residual: np.ndarray, upper_sd: np.ndarray, half_window: int) -> float:
    """Lower bound divided by upper bound, pose by pose.

    Each residual is divided by the residual SD that its own Monte Carlo SD predicts. The
    robust SD of these standardised residuals is 1 if the Monte Carlo SD were white noise
    from frame to frame, and below 1 by the factor by which the visible jitter is smaller.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        standardised = residual / (upper_sd * np.sqrt(residual_variance_factor(half_window)))
    return robust_sd(standardised)


def reconciling_correlation(ratio: float) -> float:
    """Left/right joint error correlation at which the Monte Carlo SD equals the lower bound.

    For two joints with similar error, a correlation ``rho`` scales the SD of their
    difference, and to first order the angular SD, by ``sqrt(1 - rho)``. Hence
    ``rho = 1 - ratio ** 2``. Clipped to [0, 1].
    """
    return float(np.clip(1.0 - ratio**2, 0.0, 1.0))


def calibrate_bound_correlation(
    table: pd.DataFrame,
    *,
    half_window: int = HALF_WINDOWS[PRIMARY_WINDOW],
    upper_correlation: float = 0.0,
    segments: tuple[str, ...] = SEGMENTS,
) -> dict[str, float]:
    """Calibrate the left/right error correlation of the lower bound on orientation tables.

    This is the calibration behind ``BOUND_CORRELATION["lower"]`` in
    :mod:`blindside.geometry.uncertainty`. Run it on the tables of other matches to
    recalibrate the bracket, then pass the result to the table builder::

        table = pd.concat([cached_orientation_table(path, match_id) for ...])
        rho = calibrate_bound_correlation(table)
        config = OrientationConfig(lower_correlation=tuple(rho[s] for s in SEGMENT_JOINTS))

    Per segment, the residual of the angle from a local quadratic fit is standardised by
    the residual SD that the Monte Carlo SD of the same pose predicts (:func:`bound_ratio`).
    The correlation is the one at which the Monte Carlo SD would shrink to the visible
    jitter (:func:`reconciling_correlation`).

    Parameters
    ----------
    table
        Orientation table of one or more matches, pooled. Needs ``match_id``,
        ``player_id``, ``frame`` and, per segment, ``{segment}_deg`` and
        ``{segment}_sd_deg``. Only the upper bound columns are read, so the result does not
        depend on the ``lower_correlation`` the table was built with.
    half_window
        Half window of the quadratic fit in frames, default the 0.2 s window.
    upper_correlation
        ``OrientationConfig.correlation`` the table was built with, 0 for independence.
        The Monte Carlo SD then already carries a factor ``sqrt(1 - upper_correlation)``,
        which is taken out again.
    segments
        Segments to calibrate.

    Returns
    -------
    Correlation per segment in [0, 1], not rounded. It is a statement about the frame to
    frame jitter only: error that is smooth in time is invisible to it, so the result is
    the correlation of the lower bound, not an estimate of the true correlation.
    """
    if not -1.0 <= upper_correlation < 1.0:
        raise ValueError("upper_correlation must lie in [-1, 1)")
    ordered = sort_for_time_series(table)
    calibrated = {}
    for segment in segments:
        residual = local_quadratic_residual(ordered, f"{segment}_deg", half_window)
        upper = ordered[f"{segment}_sd_deg"].to_numpy(dtype=float)
        ratio = bound_ratio(residual, upper, half_window) * np.sqrt(1.0 - upper_correlation)
        calibrated[segment] = reconciling_correlation(ratio)
    return calibrated


def add_residuals(table: pd.DataFrame) -> pd.DataFrame:
    """Sorted copy of the table with ``{segment}_res_{h}`` residual columns for every window."""
    ordered = sort_for_time_series(table)
    for segment in SEGMENTS:
        for half_window in HALF_WINDOWS.values():
            ordered[f"{segment}_res_{half_window}"] = local_quadratic_residual(
                ordered, f"{segment}_deg", half_window
            ).astype(np.float32)
    return ordered


def _bracket_row(group: pd.DataFrame, segment: str) -> dict[str, float]:
    upper = group[f"{segment}_sd_deg"].to_numpy(dtype=float)
    row: dict[str, float] = {"n": int(np.isfinite(group[f"{segment}_res_2"]).sum())}
    for label, half_window in HALF_WINDOWS.items():
        residual = group[f"{segment}_res_{half_window}"].to_numpy(dtype=float)
        row[f"lower_sd_deg ({label})"] = jitter_sd(residual, half_window)
        if label == PRIMARY_WINDOW:
            ratio = bound_ratio(residual, upper, half_window)
    row["upper_sd_deg"] = float(np.nanmedian(upper)) if len(upper) else float("nan")
    row["ratio"] = ratio
    row["reconciling_correlation"] = reconciling_correlation(ratio)
    return row


def bracket_summary(
    with_residuals: pd.DataFrame, by: str | None = None, edges: tuple[float, ...] | None = None
) -> pd.DataFrame:
    """Lower and upper bound of the angular SD per segment, optionally by bands of a column.

    Parameters
    ----------
    with_residuals
        Output of :func:`add_residuals`.
    by, edges
        Column to band (for example ``speed`` or ``camera_distance``) and the band edges.

    Returns
    -------
    One row per segment and band: ``n``, the jitter lower bound for both windows, the
    median Monte Carlo SD as the upper bound, the pose-by-pose ratio of the two for the
    primary window and the left/right correlation that would reconcile them.
    """
    rows = []
    for segment in SEGMENTS:
        if by is None:
            rows.append({"segment": segment, **_bracket_row(with_residuals, segment)})
            continue
        values = with_residuals[by].to_numpy(dtype=float)
        if by == "speed":
            labels = speed_band_labels(edges or SPEED_EDGES)
        else:
            labels = [
                f"above {low:g} m" if np.isinf(high) else f"{low:g} to {high:g} m"
                for low, high in zip(edges[:-1], edges[1:], strict=True)
            ]
        bounds = edges or SPEED_EDGES
        for label, low, high in zip(labels, bounds[:-1], bounds[1:], strict=True):
            group = with_residuals[(values >= low) & (values < high)]
            rows.append({"segment": segment, "band": label, **_bracket_row(group, segment)})
    return pd.DataFrame(rows)


def residual_autocorrelation(
    with_residuals: pd.DataFrame, lags: tuple[int, ...] = AUTOCORRELATION_LAGS
) -> pd.DataFrame:
    """Autocorrelation of the jitter residuals within a player's gap-free runs.

    Residuals are clipped at five robust SDs before correlating, so that a handful of
    sharp turns do not dominate. The ``white_noise`` column is the value that the fit
    itself induces on uncorrelated noise.
    """
    rows = []
    for segment in SEGMENTS:
        for label, half_window in HALF_WINDOWS.items():
            residual = with_residuals[f"{segment}_res_{half_window}"].to_numpy(dtype=float)
            limit = 5.0 * robust_sd(residual)
            clipped = np.clip(residual, -limit, limit)
            for lag in lags:
                a, b = clipped[:-lag], clipped[lag:]
                valid = _same_run(with_residuals, lag) & np.isfinite(a) & np.isfinite(b)
                rows.append(
                    {
                        "segment": segment,
                        "window": label,
                        "lag_frames": lag,
                        "lag_s": lag / POSE_FPS,
                        "n": int(valid.sum()),
                        "autocorrelation": float(np.corrcoef(a[valid], b[valid])[0, 1]),
                        "white_noise": white_noise_autocorrelation(half_window, lag),
                    }
                )
    return pd.DataFrame(rows)
