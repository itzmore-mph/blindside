"""Figures of the reliability map (docs/RELIABILITY.md section 2).

Every function takes the tidy tables of :mod:`blindside.analysis.reliability` and
:mod:`blindside.analysis.bracket` and returns a matplotlib ``Figure``. Nothing is read from
disk here, so the same functions draw the full matches and the sample phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from mplsoccer import Pitch

from blindside.analysis.bracket import PRIMARY_WINDOW
from blindside.analysis.consistency import SEGMENTS, USABLE_SD_DEG
from blindside.analysis.reliability import STANDARD_PITCH, zone_grid
from blindside.viz.style import (
    GRID,
    INK,
    INK_SECONDARY,
    SEGMENT_COLORS,
    SEGMENT_LABELS,
    SEQUENTIAL,
    SURFACE,
    style_context,
)

LOWER_COLUMN = f"lower_sd_deg ({PRIMARY_WINDOW})"
#: Distance in metres between the touchline and the camera marker of the pitch maps.
CAMERA_MARKER_OFFSET = 5.0
FACTOR_STYLES: tuple[tuple[int, int] | str, ...] = ((0, (1, 1)), "-", (0, (5, 2)))


def _short_band(label: str) -> str:
    """``0 to 2 m/s`` becomes ``0 to 2``, ``above 6 m/s`` becomes ``above 6``."""
    return label.replace(" m/s", "")


def _draw_bracket(axis: Axes, speed_bracket: pd.DataFrame) -> None:
    """Range bar from the lower to the upper bound per speed band and segment."""
    bands = list(dict.fromkeys(speed_bracket["band"]))
    width = 0.22
    for i, segment in enumerate(SEGMENTS):
        rows = speed_bracket[speed_bracket["segment"] == segment].set_index("band").loc[bands]
        lower, upper = rows[LOWER_COLUMN].to_numpy(), rows["upper_sd_deg"].to_numpy()
        x = np.arange(len(bands)) + (i - 1) * (width + 0.04)
        axis.bar(x, upper - lower, width, bottom=lower, color=SEGMENT_COLORS[segment],
                 label=SEGMENT_LABELS[segment])  # fmt: skip
    axis.axhline(USABLE_SD_DEG, color=INK_SECONDARY, lw=1.0, ls=(0, (4, 2)))
    axis.text(len(bands) - 0.5, USABLE_SD_DEG - 0.4, f"usability threshold, {USABLE_SD_DEG:g} deg",
              ha="right", va="top", fontsize=8, color=INK_SECONDARY)  # fmt: skip
    axis.set(
        title="Angular error: a bracket, not a number",
        xlabel="Player speed (m/s)",
        ylabel="Angular SD (degrees)",
        xticks=np.arange(len(bands)),
        xticklabels=[_short_band(band) for band in bands],
        xlim=(-0.5, len(bands) - 0.5),
        ylim=(0, USABLE_SD_DEG + 7),
    )
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 0.73), labelcolor=INK, fontsize=8,
                handlelength=1.2)  # fmt: skip


def _draw_zone_map(
    axis: Axes,
    grid: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    camera_side: str | None = "-y",
    pitch_size: tuple[float, float] = STANDARD_PITCH,
) -> QuadMesh:
    """Zone values under the pitch markings. Returns the mesh for a colour bar."""
    length, width = pitch_size
    pad = CAMERA_MARKER_OFFSET + 4.0
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=length,
        pitch_width=width,
        pitch_color=SURFACE,
        line_color=INK_SECONDARY,
        linewidth=0.8,
        line_zorder=2,
        pad_bottom=pad if camera_side == "-y" else 2.0,
        pad_top=pad if camera_side == "+y" else 2.0,
        pad_left=2.0,
        pad_right=2.0,
    )
    pitch.draw(ax=axis)
    x_edges = np.linspace(-length / 2.0, length / 2.0, grid.shape[1] + 1)
    y_edges = np.linspace(-width / 2.0, width / 2.0, grid.shape[0] + 1)
    mesh = axis.pcolormesh(
        x_edges, y_edges, grid, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, zorder=1, rasterized=True,
        edgecolors=SURFACE, linewidth=0.3,
    )  # fmt: skip
    if camera_side is not None:
        sign = -1.0 if camera_side == "-y" else 1.0
        y_camera = sign * (width / 2.0 + CAMERA_MARKER_OFFSET)
        axis.plot(0.0, y_camera, marker="^" if sign < 0 else "v", ms=7, color=INK, zorder=3,
                  clip_on=False)  # fmt: skip
        axis.text(3.0, y_camera, "main camera", ha="left", va="center", fontsize=8,
                  color=INK_SECONDARY)  # fmt: skip
    return mesh


def _draw_velocity(axis: Axes, velocity: pd.DataFrame) -> None:
    """Median absolute difference between facing and direction of travel by speed band."""
    pooled = velocity[velocity["match"] == "all"]
    bands = list(dict.fromkeys(pooled["speed_band"]))
    for segment in SEGMENTS:
        rows = pooled[pooled["segment"] == segment].set_index("speed_band").loc[bands]
        axis.plot(np.arange(len(bands)), rows["median_abs_deg"], color=SEGMENT_COLORS[segment],
                  marker="o", ms=5, label=SEGMENT_LABELS[segment])  # fmt: skip
    axis.set(
        title="Check without ground truth",
        xlabel="Player speed (m/s)",
        ylabel="Median |facing - direction of travel| (degrees)",
        xticks=np.arange(len(bands)),
        xticklabels=[_short_band(band) for band in bands],
        xlim=(-0.3, len(bands) - 0.7),
        ylim=(0, None),
    )
    axis.legend(loc="upper right", labelcolor=INK, fontsize=8)


def plot_reliability_composite(
    speed_bracket: pd.DataFrame,
    zone_summary: pd.DataFrame,
    velocity: pd.DataFrame,
    *,
    map_segment: str = "head",
    map_value: str = "lower_sd_deg",
    map_label: str = "Head jitter, lower bound (degrees)",
    camera_side: str | None = "-y",
    caption: str | None = None,
) -> Figure:
    """The composite reliability figure.

    Left: the bracket of the angular SD per segment and speed band, from the jitter lower
    bound to the Monte Carlo upper bound. Centre: one statistic of the zone summary on the
    pitch, by default the lower bound of the head, with the camera side marked. Right: the
    velocity consistency check for all three segments.

    Parameters
    ----------
    speed_bracket
        ``bracket_summary(with_residuals, "speed", SPEED_EDGES)``.
    zone_summary
        Output of :func:`blindside.analysis.reliability.pitch_zone_summary`.
    velocity
        Output of :func:`blindside.analysis.reliability.velocity_agreement_by_band`.
    map_segment, map_value, map_label
        Segment and column of ``zone_summary`` shown on the pitch, and the colour bar label.
    camera_side
        ``"-y"``, ``"+y"`` or None to leave the camera marker out.
    caption
        Optional line above the panels.
    """
    with style_context():
        figure = Figure(figsize=(12.0, 4.0), layout="constrained")
        left, centre, right = figure.subplots(1, 3, width_ratios=(1.0, 1.35, 1.0))
        _draw_bracket(left, speed_bracket)
        left.text(0.02, 0.98, "top of bar: joint errors independent (upper bound)\n"
                  "bottom of bar: frame-to-frame jitter (lower bound)",
                  transform=left.transAxes, ha="left", va="top", fontsize=7.5,
                  color=INK_SECONDARY)  # fmt: skip
        mesh = _draw_zone_map(
            centre, zone_grid(zone_summary, map_segment, map_value), camera_side=camera_side
        )
        centre.set_title("Where the pose is noisier")
        bar = figure.colorbar(mesh, ax=centre, orientation="horizontal", pad=0.02, shrink=0.7,
                              aspect=30)  # fmt: skip
        bar.set_label(map_label)
        bar.outline.set_visible(False)
        _draw_velocity(right, velocity)
        if caption:
            figure.suptitle(caption, x=0.01, ha="left", fontsize=9, color=INK_SECONDARY)
    return figure


def plot_zone_maps(zone_summary: pd.DataFrame, camera_side: str | None = "-y") -> Figure:
    """Lower and upper bound of the angular SD per pitch zone, one row per segment.

    Each panel has its own colour scale, because the bounds differ by a factor of four to
    six and a shared scale would hide the pattern inside each of them.
    """
    columns = (("lower_sd_deg", "Lower bound (jitter)"), ("upper_sd_deg", "Upper bound (MC)"))
    with style_context():
        figure = Figure(figsize=(9.5, 10.5), layout="constrained")
        axes = figure.subplots(len(SEGMENTS), len(columns))
        for row, segment in enumerate(SEGMENTS):
            for column, (value, title) in enumerate(columns):
                axis = axes[row, column]
                mesh = _draw_zone_map(axis, zone_grid(zone_summary, segment, value),
                                      camera_side=camera_side)  # fmt: skip
                axis.set_title(f"{SEGMENT_LABELS[segment]}: {title}")
                bar = figure.colorbar(mesh, ax=axis, pad=0.02, shrink=0.75)
                bar.set_label("Angular SD (degrees)")
                bar.outline.set_visible(False)
    return figure


def plot_usability_sensitivity(sensitivity: pd.DataFrame) -> Figure:
    """Share of usable poses against the SD threshold, for each sigma conversion factor.

    One panel per segment. Upper bound only: the lower bound does not depend on the factor.
    """
    thresholds = sorted(
        float(c.removeprefix("below_")) for c in sensitivity.columns if c.startswith("below_")
    )
    factors = sorted(sensitivity["p90_to_sigma"].unique())
    with style_context():
        figure = Figure(figsize=(10.0, 3.4), layout="constrained")
        axes = figure.subplots(1, len(SEGMENTS), sharey=True)
        for axis, segment in zip(axes, SEGMENTS, strict=True):
            rows = sensitivity[sensitivity["segment"] == segment].set_index("p90_to_sigma")
            for factor, style in zip(factors, FACTOR_STYLES, strict=False):
                share = [100.0 * rows.loc[factor, f"below_{t:g}"] for t in thresholds]
                axis.plot(thresholds, share, color=SEGMENT_COLORS[segment], ls=style, marker="o",
                          ms=4, label=f"sigma = p90 / {factor:g}")  # fmt: skip
            axis.set(title=SEGMENT_LABELS[segment], xlabel="SD threshold (degrees)",
                     xticks=thresholds, ylim=(0, 102))  # fmt: skip
            axis.legend(loc="lower right", labelcolor=INK, fontsize=8)
        axes[0].set_ylabel("Poses below the threshold (%)")
    return figure


def plot_misalignment(table: pd.DataFrame, max_cm: float = 40.0) -> Figure:
    """Distribution of the offset between the XY tracking position and the pose centroid.

    Left: ground distance. Right: the signed component along the direction of travel,
    positive when the pose centroid is ahead of the tracking position.
    """
    dx = table["centroid_dx_m"].to_numpy(dtype=float) * 100.0
    dy = table["centroid_dy_m"].to_numpy(dtype=float) * 100.0
    heading = np.radians(table["heading_deg"].to_numpy(dtype=float))
    distance = np.hypot(dx, dy)
    along = dx * np.cos(heading) + dy * np.sin(heading)
    with style_context():
        figure = Figure(figsize=(9.0, 3.4), layout="constrained")
        left, right = figure.subplots(1, 2)
        colour = SEQUENTIAL(0.6)
        for axis, values, edges, title, label in (
            (left, distance, np.arange(0.0, max_cm + 1.0, 1.0), "Distance between XY and pose",
             "Ground distance (cm)"),
            (right, along, np.arange(-max_cm, max_cm + 1.0, 1.0), "Offset along the run",
             "Pose ahead of XY (cm)"),
        ):  # fmt: skip
            values = values[np.isfinite(values)]
            weights = np.full(len(values), 100.0 / max(len(values), 1))
            axis.hist(values, bins=edges, weights=weights, color=colour, edgecolor=SURFACE,
                      linewidth=0.3)  # fmt: skip
            if len(values):
                median = float(np.median(values))
                axis.axvline(median, color=INK, lw=1.2)
                axis.text(0.98, 0.95, f"median {median:.1f} cm", transform=axis.transAxes,
                          ha="right", va="top", fontsize=8, color=INK)  # fmt: skip
            axis.set(title=title, xlabel=label, xlim=(edges[0], edges[-1]))
        left.set_ylabel("Share of poses per cm (%)")
        right.axvline(0.0, color=GRID, lw=1.0, zorder=0)
    return figure
