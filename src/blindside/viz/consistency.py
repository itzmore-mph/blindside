"""Figure for the velocity consistency check (Phase 1 gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from blindside.analysis.consistency import SEGMENTS, signed_difference_deg
from blindside.viz.style import (
    INK,
    INK_SECONDARY,
    SEGMENT_COLORS,
    SEGMENT_LABELS,
    SEQUENTIAL,
    style_context,
)

MAX_SPEED = 8.0
SPEED_STEP = 0.5
ANGLE_STEP = 5.0
MIN_BIN_COUNT = 500


def _binned(speed: np.ndarray, value: np.ndarray, edges: np.ndarray, quantile: float) -> np.ndarray:
    index = np.digitize(speed, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        selected = value[index == b]
        if len(selected) >= MIN_BIN_COUNT:
            out[b] = np.quantile(selected, quantile)
    return out


def plot_velocity_check(table: pd.DataFrame) -> Figure:
    """Hip facing minus velocity heading as a function of speed.

    Left: distribution of the signed difference for the hips in each speed bin (every
    column sums to one, so slow and fast bins are comparable) with its median and central
    80% band. Right: median absolute difference by speed for torso, hips and head.
    Speed bins with fewer than ``MIN_BIN_COUNT`` poses are not drawn.
    """
    speed = table["speed"].to_numpy(dtype=float)
    edges = np.arange(0.0, MAX_SPEED + SPEED_STEP, SPEED_STEP)
    centres = (edges[:-1] + edges[1:]) / 2.0
    differences = {
        segment: signed_difference_deg(table[f"{segment}_deg"], table["heading_deg"])
        for segment in SEGMENTS
    }

    with style_context():
        figure = Figure(figsize=(10.5, 4.2), layout="constrained")
        left, right = figure.subplots(1, 2, width_ratios=(1.25, 1.0))

        hips = differences["hips"]
        valid = np.isfinite(hips) & np.isfinite(speed) & (speed < MAX_SPEED)
        angle_edges = np.arange(-180.0, 180.0 + ANGLE_STEP, ANGLE_STEP)
        density, _, _ = np.histogram2d(speed[valid], hips[valid], bins=(edges, angle_edges))
        counts = density.sum(axis=1, keepdims=True)
        density = np.where(counts >= MIN_BIN_COUNT, density / np.maximum(counts, 1), np.nan)
        mesh = left.pcolormesh(
            edges, angle_edges, density.T, cmap=SEQUENTIAL, vmin=0, rasterized=True
        )
        left.plot(centres, _binned(speed[valid], hips[valid], edges, 0.5), color=INK, lw=1.5,
                  label="Median")  # fmt: skip
        for i, quantile in enumerate((0.1, 0.9)):
            left.plot(
                centres, _binned(speed[valid], hips[valid], edges, quantile), color=INK, lw=1.0,
                ls=(0, (4, 2)), label="10th and 90th percentile" if i == 0 else None,
            )  # fmt: skip
        left.set(
            title="Hip facing minus direction of travel",
            xlabel="Player speed (m/s)",
            ylabel="Signed difference (degrees)",
            xlim=(0, MAX_SPEED),
            ylim=(-180, 180),
            yticks=np.arange(-180, 181, 45),
        )
        left.grid(False)
        left.legend(loc="lower right", labelcolor=INK)
        bar = figure.colorbar(mesh, ax=left, pad=0.02)
        bar.set_label("Share of poses in the speed bin, per 5 degrees")
        bar.outline.set_visible(False)

        for segment in SEGMENTS:
            difference = np.abs(differences[segment])
            ok = np.isfinite(difference) & np.isfinite(speed) & (speed < MAX_SPEED)
            median = _binned(speed[ok], difference[ok], edges, 0.5)
            right.plot(centres, median, color=SEGMENT_COLORS[segment], marker="o", ms=4,
                       label=SEGMENT_LABELS[segment])  # fmt: skip
        right.set(
            title="Median absolute difference to direction of travel",
            xlabel="Player speed (m/s)",
            ylabel="Degrees",
            xlim=(0, MAX_SPEED),
            ylim=(0, None),
            xticks=np.arange(0, MAX_SPEED + 1, 2),
        )
        right.legend(loc="upper right", labelcolor=INK)
        n_matches = table["match_id"].nunique()
        figure.suptitle(
            f"Velocity consistency check, {len(table):,} poses from {n_matches} matches. "
            "No ground truth: fast players should face where they run.",
            x=0.01, ha="left", fontsize=9, color=INK_SECONDARY,
        )  # fmt: skip
    return figure
