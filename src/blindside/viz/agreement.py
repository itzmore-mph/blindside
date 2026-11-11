"""Figures for the pose blind-side label against the XY proxy.

:func:`plot_proxy_agreement` is README figure 2. :func:`plot_engagement_geometry` shows the
quantities behind both labels for single engagements, and is what the fast path draws on
the sample phase, where the agreement statistics cannot be computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

from blindside.analysis.consistency import SPEED_EDGES, speed_band_labels
from blindside.viz.style import (
    GRID,
    INK,
    INK_SECONDARY,
    SEGMENT_COLORS,
    style_context,
)

#: Segments of the two pose labels, in drawing order, with the feature that carries them.
LABEL_FEATURES: dict[str, str] = {"head": "p_blind_side_head", "torso": "p_blind_side_torso"}
POSE_LABEL_NAMES: dict[str, str] = {"head": "Pose, head", "torso": "Pose, torso"}
PROXY_COLOR = INK_SECONDARY
PROXY_NAME = "XY proxy"
ALL_GROUP = "all"


def _short_band(label: str) -> str:
    return label.replace(" m/s", "")


def _rows(agreement: pd.DataFrame, segment: str, bound: str, groups: list[str]) -> pd.DataFrame:
    selected = agreement[
        (agreement["feature"] == LABEL_FEATURES[segment]) & (agreement["bound"] == bound)
    ].set_index("group")
    return selected.loc[groups]


def _draw_kappa(axis: Axes, agreement: pd.DataFrame, bound: str, bands: list[str]) -> None:
    """Cohen's kappa per carrier speed band for both pose labels, with intervals."""
    x = np.arange(len(bands))
    for offset, segment in zip((-0.07, 0.07), LABEL_FEATURES, strict=True):
        rows = _rows(agreement, segment, bound, bands)
        kappa = rows["kappa"].to_numpy(dtype=float)
        low, high = rows["kappa_low"].to_numpy(dtype=float), rows["kappa_high"].to_numpy(float)
        color = SEGMENT_COLORS[segment]
        axis.plot(x + offset, kappa, color=color, marker="o", ms=5, label=POSE_LABEL_NAMES[segment])
        has_interval = np.isfinite(low) & np.isfinite(high)
        axis.vlines((x + offset)[has_interval], low[has_interval], high[has_interval],
                    color=color, lw=1.4)  # fmt: skip
        # Head labels sit below right of the marker, torso labels above left: clear of both
        # lines. In the first band the two values almost coincide, so the labels go outwards.
        inner = ((6, -8), "left", "top") if offset < 0 else ((-6, 7), "right", "bottom")
        first = ((-7, 0), "right", "center") if offset < 0 else ((7, -6), "left", "top")
        for i, (xi, value) in enumerate(zip(x + offset, kappa, strict=True)):
            shift, align, valign = first if i == 0 else inner
            axis.annotate(f"{value:.2f}", (xi, value), xytext=shift, textcoords="offset points",
                          ha=align, va=valign, fontsize=8, color=INK)  # fmt: skip
    counts = _rows(agreement, "head", bound, bands)["n"].to_numpy(dtype=int)
    axis.axhline(0.0, color=INK_SECONDARY, lw=1.0, ls=(0, (4, 2)))
    axis.text(len(bands) - 0.55, 0.015, "agreement expected by chance", ha="right", va="bottom",
              fontsize=8, color=INK_SECONDARY)  # fmt: skip
    axis.set(
        title="XY proxy against the pose label: agreement",
        xlabel="Carrier speed (m/s), engagements per band",
        ylabel="Cohen's kappa, 95% interval",
        xticks=x,
        xticklabels=[f"{_short_band(b)}\nn = {n:,}" for b, n in zip(bands, counts, strict=True)],
        xlim=(-0.5, len(bands) - 0.5),
        ylim=(-0.08, 1.0),
    )
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper left", labelcolor=INK, fontsize=8, title="XY proxy against",
                title_fontsize=8, alignment="left")  # fmt: skip


def _draw_shares(axis: Axes, agreement: pd.DataFrame, bound: str, bands: list[str]) -> None:
    """Share of engagements each label calls blind, per speed band and overall."""
    groups = [*bands, ALL_GROUP]
    x = np.append(np.arange(len(bands)), len(bands) + 0.35)
    width = 0.25
    series = [
        (POSE_LABEL_NAMES[segment], SEGMENT_COLORS[segment],
         _rows(agreement, segment, bound, groups)["pose_blind"])
        for segment in LABEL_FEATURES
    ]  # fmt: skip
    series.append((PROXY_NAME, PROXY_COLOR, _rows(agreement, "head", bound, groups)["xy_blind"]))
    for i, (label, color, share) in enumerate(series):
        values = 100.0 * share.to_numpy(dtype=float)
        position = x + (i - 1) * (width + 0.02)
        axis.bar(position, values, width, color=color, label=label)
        for xi, value in zip(position, values, strict=True):
            axis.text(xi, value + 0.8, f"{value:.1f}", ha="center", va="bottom", fontsize=7,
                      color=INK)  # fmt: skip
    axis.axvline(len(bands) - 0.33, color=GRID, lw=1.0)
    axis.set(
        title="Who is called blind",
        xlabel="Carrier speed (m/s)",
        ylabel="Engagements called blind (%)",
        xticks=x,
        xticklabels=[*(_short_band(b) for b in bands), "all"],
        xlim=(-0.5, x[-1] + 0.5),
        ylim=(0, 62),
    )
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper right", labelcolor=INK, fontsize=8, ncols=3, handlelength=1.2,
                columnspacing=1.2)  # fmt: skip


def plot_proxy_agreement(
    agreement: pd.DataFrame, *, bound: str = "lower", caption: str | None = None
) -> Figure:
    """README figure 2: the XY proxy for the blind side against the pose label.

    Left: Cohen's kappa of the XY proxy against the head and the torso label per carrier
    speed band, with the cluster bootstrap intervals. Right: the share of engagements
    called blind by each of the three labels, per band and overall. No outcome enters.

    Kappa measures agreement, not accuracy. Without ground truth it cannot say which
    label is right.

    Parameters
    ----------
    agreement
        Output of :func:`blindside.analysis.pipeline.proxy_agreement`: one row per
        ``feature``, ``bound`` and ``group``, columns of
        :func:`blindside.analysis.agreement.agreement_table`.
    bound
        Uncertainty bound of the pose label that is drawn, ``"lower"`` or ``"upper"``.
    caption
        Optional line above the panels.
    """
    bands = speed_band_labels(SPEED_EDGES)
    with style_context():
        figure = Figure(figsize=(12.0, 4.0), layout="constrained")
        left, right = figure.subplots(1, 2, width_ratios=(1.0, 1.2))
        _draw_kappa(left, agreement, bound, bands)
        _draw_shares(right, agreement, bound, bands)
        if caption:
            figure.suptitle(caption, x=0.01, ha="left", fontsize=9, color=INK_SECONDARY)
    return figure


def _wedge(axis: Axes, centre: tuple[float, float], radius: float, angle_deg: float,
           half_width_deg: float, **kwargs: object) -> None:  # fmt: skip
    axis.add_patch(
        Wedge(centre, radius, angle_deg - half_width_deg, angle_deg + half_width_deg, **kwargs)
    )


def _draw_engagement(axis: Axes, row: pd.Series, half_angle_deg: float) -> None:
    carrier = (float(row["carrier_x"]), float(row["carrier_y"]))
    presser = (float(row["presser_x"]), float(row["presser_y"]))
    distance = float(np.hypot(presser[0] - carrier[0], presser[1] - carrier[1]))
    reach = max(distance + 1.5, 5.0)
    head, torso = SEGMENT_COLORS["head"], SEGMENT_COLORS["torso"]

    _wedge(axis, carrier, 0.95 * reach, row["head_deg"], half_angle_deg, color=head, alpha=0.10,
           lw=0)  # fmt: skip
    _wedge(axis, carrier, 0.62 * reach, row["head_deg"], row["head_sd_upper_deg"], color=head,
           alpha=0.35, lw=0)  # fmt: skip
    _wedge(axis, carrier, 0.62 * reach, row["head_deg"], row["head_sd_lower_deg"], color=head,
           alpha=0.95, lw=0)  # fmt: skip
    _wedge(axis, carrier, 0.40 * reach, row["torso_deg"], row["torso_sd_upper_deg"], color=torso,
           alpha=0.35, lw=0)  # fmt: skip
    _wedge(axis, carrier, 0.40 * reach, row["torso_deg"], row["torso_sd_lower_deg"], color=torso,
           alpha=0.95, lw=0)  # fmt: skip

    step = np.array([row["travel_dx"], row["travel_dy"]], dtype=float)
    length = float(np.hypot(*step))
    if np.isfinite(length) and length > 0:
        tip = np.array(carrier) + 0.8 * reach * step / length
        axis.annotate("", tip, carrier, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.3,
                                                     "ls": (0, (4, 2))})  # fmt: skip
        axis.annotate("direction of travel", tip, xytext=(4, 4), textcoords="offset points",
                      fontsize=7.5, color=INK_SECONDARY)  # fmt: skip
    axis.plot(*carrier, "o", ms=7, color=INK, zorder=5)
    axis.plot(*presser, "o", ms=7, mfc="white", mec=INK, mew=1.6, zorder=5)
    axis.annotate("carrier", carrier, xytext=(7, -11), textcoords="offset points", fontsize=8)
    axis.annotate("presser", presser, xytext=(7, -11), textcoords="offset points", fontsize=8)

    proxy_blind = row["approach_angle_heading_deg"] > half_angle_deg
    axis.set_title(
        f"Engagement at {row['time_s']:.1f} s of the clip, carrier at "
        f"{row['carrier_speed']:.1f} m/s"
    )
    axis.set_xlabel(
        f"p_blind_side under the lower / upper bound: head {row['p_head_lower']:.2f} / "
        f"{row['p_head_upper']:.2f}, torso {row['p_torso_lower']:.2f} / "
        f"{row['p_torso_upper']:.2f}\n"
        f"XY proxy: presser {row['approach_angle_heading_deg']:.0f} degrees off the direction "
        f"of travel, {'blind' if proxy_blind else 'in view'}",
        color=INK,
        fontsize=8.5,
        loc="left",
    )
    axis.set(
        xlim=(carrier[0] - reach, carrier[0] + reach),
        ylim=(carrier[1] - reach, carrier[1] + reach),
        aspect="equal",
        xticks=[],
        yticks=[],
    )
    axis.grid(visible=False)
    axis.plot([carrier[0] - 0.9 * reach, carrier[0] - 0.9 * reach + 2.0],
              [carrier[1] - 0.9 * reach] * 2, color=INK_SECONDARY, lw=2)  # fmt: skip
    axis.text(carrier[0] - 0.9 * reach + 1.0, carrier[1] - 0.86 * reach, "2 m", ha="center",
              fontsize=7.5, color=INK_SECONDARY)  # fmt: skip


def plot_engagement_geometry(
    engagements: pd.DataFrame, *, half_angle_deg: float = 100.0, caption: str | None = None
) -> Figure:
    """Both blind-side labels drawn for single engagements, seen from above.

    Each panel shows the carrier with the head visual field (light sector), the head and
    the torso facing direction as wedges whose dark core is the lower and whose light rim
    is the upper bound of the angular uncertainty, the direction of travel that the XY
    proxy uses, and the presser.

    Parameters
    ----------
    engagements
        One row per engagement with ``carrier_x``, ``carrier_y``, ``presser_x``,
        ``presser_y`` (metres), ``travel_dx``, ``travel_dy`` (carrier displacement before
        the start), ``carrier_speed``, ``approach_angle_heading_deg``, ``time_s``, and per
        segment ``{segment}_deg``, ``{segment}_sd_lower_deg``, ``{segment}_sd_upper_deg``,
        ``p_{segment}_lower`` and ``p_{segment}_upper`` for ``head`` and ``torso``.
    half_angle_deg
        Half-angle of the visual field, also the threshold of the XY proxy.
    caption
        Optional line above the panels.
    """
    n = max(len(engagements), 1)
    with style_context():
        figure = Figure(figsize=(5.2 * n, 5.4), layout="constrained")
        axes = np.atleast_1d(figure.subplots(1, n))
        if engagements.empty:
            axes[0].text(0.5, 0.5, "no on-ball engagement with pose in this clip", ha="center",
                         va="center", transform=axes[0].transAxes, color=INK_SECONDARY)  # fmt: skip
            axes[0].set(xticks=[], yticks=[])
        for axis, (_, row) in zip(axes, engagements.iterrows(), strict=False):
            _draw_engagement(axis, row, half_angle_deg)
        handles = [
            Wedge((0, 0), 1, 0, 1, color=SEGMENT_COLORS["head"], label="Head facing"),
            Wedge((0, 0), 1, 0, 1, color=SEGMENT_COLORS["torso"], label="Torso facing"),
            Wedge((0, 0), 1, 0, 1, color=SEGMENT_COLORS["head"], alpha=0.15,
                  label=f"Visual field of the head, {half_angle_deg:g} degrees to each side"),
        ]  # fmt: skip
        figure.legend(handles=handles, loc="outside lower left", ncols=3, fontsize=8,
                      labelcolor=INK)  # fmt: skip
        if caption:
            figure.suptitle(caption, x=0.01, ha="left", fontsize=9, color=INK_SECONDARY)
    return figure
