"""Single engagements on the pitch, for the video walkthrough and ``docs/``.

:func:`plot_tactical_case_study` draws what each blind-side label sees of an engagement:
the visual field of the carrier's head with the bracket of its angular error at the edges,
the field that the XY proxy assumes around the direction of travel, and the presser.
:func:`plot_scene_cone` draws one engagement with the runs that led to it and a box of
its exact values. The type is sized for a screen share, not for the README.

A panel shows one engagement. It can show that the two labels disagree. It cannot show
which one is right: there is no ground truth for orientation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Wedge
from mplsoccer import Pitch

from blindside.viz.style import GRID, INK, INK_SECONDARY, SEGMENT_COLORS, SURFACE, style_context

#: Half extent of a panel in metres along x and y, centred between carrier and presser.
HALF_WINDOW_M: tuple[float, float] = (12.6, 9.0)
#: Radius in metres of the drawn visual field.
FIELD_RADIUS_M = 8.0
PITCH_LINE = "#b9b7b0"
_FONT = {"title": 21, "verdict": 16, "label": 15, "small": 13, "suptitle": 26}


def _unit(angle_deg: float) -> np.ndarray:
    return np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])


def _verdict(blind: bool) -> str:
    return "BLIND" if blind else "IN VIEW"


def _draw_visual_field(axis: Axes, row: pd.Series, half_angle_deg: float) -> None:
    """Head visual field with the lower and the upper bound of the angular error at its edges."""
    carrier = (float(row["carrier_x"]), float(row["carrier_y"]))
    color = SEGMENT_COLORS["head"]
    facing = float(row["head_deg"])
    axis.add_patch(
        Wedge(carrier, FIELD_RADIUS_M, facing - half_angle_deg, facing + half_angle_deg,
              color=color, alpha=0.16, lw=0, zorder=2)
    )  # fmt: skip
    for edge in (facing - half_angle_deg, facing + half_angle_deg):
        for bound, alpha in (("upper", 0.30), ("lower", 0.75)):
            sd = float(row[f"head_sd_{bound}_deg"])
            axis.add_patch(
                Wedge(carrier, FIELD_RADIUS_M, edge - sd, edge + sd, color=color, alpha=alpha,
                      lw=0, zorder=2)
            )  # fmt: skip
    tip = np.array(carrier) + 0.72 * FIELD_RADIUS_M * _unit(facing)
    axis.annotate("", tip, carrier, zorder=4,
                  arrowprops={"arrowstyle": "-|>", "color": color, "lw": 3.2,
                              "mutation_scale": 22})  # fmt: skip


def _draw_proxy_field(axis: Axes, row: pd.Series, half_angle_deg: float) -> None:
    """Direction of travel and the field the XY proxy assumes around it, as a dashed outline."""
    carrier = np.array([float(row["carrier_x"]), float(row["carrier_y"])])
    step = np.array([row["travel_dx"], row["travel_dy"]], dtype=float)
    length = float(np.hypot(*step))
    if not np.isfinite(length) or length == 0.0:
        return
    travel = float(np.degrees(np.arctan2(step[1], step[0])))
    radius = 0.86 * FIELD_RADIUS_M
    axis.add_patch(
        Wedge(tuple(carrier), radius, travel - half_angle_deg, travel + half_angle_deg, fill=False,
              ec=INK, lw=1.8, ls=(0, (5, 3)), zorder=3)
    )  # fmt: skip
    axis.annotate("", carrier + 0.6 * FIELD_RADIUS_M * _unit(travel), carrier, zorder=4,
                  arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 2.4, "ls": (0, (4, 2)),
                              "mutation_scale": 20})  # fmt: skip


def _draw_players(axis: Axes, row: pd.Series, others: pd.DataFrame | None) -> None:
    carrier = np.array([float(row["carrier_x"]), float(row["carrier_y"])])
    presser = np.array([float(row["presser_x"]), float(row["presser_y"])])
    if others is not None and len(others):
        pressing = others["pressing_team"].to_numpy(dtype=bool)
        axis.scatter(others.loc[~pressing, "x"], others.loc[~pressing, "y"], s=150,
                     color=PITCH_LINE, zorder=3, clip_on=True)  # fmt: skip
        axis.scatter(others.loc[pressing, "x"], others.loc[pressing, "y"], s=150, lw=2.2,
                     facecolor=SURFACE, edgecolor=PITCH_LINE, zorder=3, clip_on=True)  # fmt: skip
    axis.plot(*zip(carrier, presser, strict=True), color=INK_SECONDARY, lw=1.4, ls=":", zorder=3)
    axis.scatter(*carrier, s=420, color=INK, zorder=6)
    axis.scatter(*presser, s=420, facecolor=SURFACE, edgecolor=INK, lw=3.4, zorder=6)
    # Each name sits on the far side of its player, away from the other one.
    away = (carrier - presser) / max(float(np.hypot(*(carrier - presser))), 1e-9)
    for name, position, direction in (("Carrier", carrier, away), ("Presser", presser, -away)):
        ha = "left" if direction[0] > 0.35 else "right" if direction[0] < -0.35 else "center"
        va = "bottom" if direction[1] > 0.35 else "top" if direction[1] < -0.35 else "center"
        axis.annotate(
            name, position, xytext=tuple(20 * direction), textcoords="offset points", ha=ha,
            va=va, fontsize=_FONT["label"], fontweight="bold", color=INK, zorder=7,
            bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.85},
        )  # fmt: skip


def _draw_pitch_window(
    axis: Axes,
    row: pd.Series,
    others: pd.DataFrame | None,
    pitch_dims: tuple[float, float],
    half_angle_deg: float,
) -> None:
    """Pitch, both fields and the players of one engagement, zoomed on carrier and presser."""
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=pitch_dims[0],
        pitch_width=pitch_dims[1],
        pitch_color=SURFACE,
        line_color=PITCH_LINE,
        linewidth=1.6,
        goal_type="box",
    )
    pitch.draw(ax=axis)
    carrier = np.array([float(row["carrier_x"]), float(row["carrier_y"])])
    presser = np.array([float(row["presser_x"]), float(row["presser_y"])])
    _draw_visual_field(axis, row, half_angle_deg)
    _draw_proxy_field(axis, row, half_angle_deg)
    _draw_players(axis, row, others)

    centre = (carrier + presser) / 2.0
    half_x, half_y = HALF_WINDOW_M
    axis.set_xlim(centre[0] - half_x, centre[0] + half_x)
    axis.set_ylim(centre[1] - half_y, centre[1] + half_y)
    for spine in axis.spines.values():
        spine.set(visible=True, color=GRID, lw=1.2)
    left, bottom = centre[0] - half_x + 0.8, centre[1] - half_y + 0.8
    axis.plot([left, left + 5.0], [bottom, bottom], color=INK_SECONDARY, lw=4, zorder=7,
              solid_capstyle="butt")  # fmt: skip
    axis.text(left + 2.5, bottom + 0.35, "5 m", ha="center", va="bottom", fontsize=_FONT["small"],
              color=INK_SECONDARY, zorder=7)  # fmt: skip


def _draw_panel(
    axis: Axes,
    row: pd.Series,
    others: pd.DataFrame | None,
    pitch_dims: tuple[float, float],
    half_angle_deg: float,
) -> None:
    _draw_pitch_window(axis, row, others, pitch_dims, half_angle_deg)
    proxy_blind = bool(row["approach_angle_heading_deg"] > half_angle_deg)
    pose_blind = bool(row["p_head_lower"] > 0.5)
    axis.set_title(str(row["title"]), fontsize=_FONT["title"], fontweight="bold", loc="left",
                   pad=14, color=INK)  # fmt: skip
    axis.text(
        0.0, -0.04,
        f"XY proxy: {_verdict(proxy_blind)}\n"
        f"Head pose: {_verdict(pose_blind)}",
        transform=axis.transAxes, ha="left", va="top", fontsize=_FONT["verdict"],
        fontweight="bold", color=INK, linespacing=1.5,
    )  # fmt: skip
    axis.text(
        0.37, -0.04,
        f"{row['approach_angle_heading_deg']:.0f} degrees off the direction of travel\n"
        f"p_blind_side {row['p_head_lower']:.2f} to {row['p_head_upper']:.2f} across both bounds",
        transform=axis.transAxes, ha="left", va="top", fontsize=_FONT["verdict"],
        color=INK_SECONDARY, linespacing=1.5,
    )  # fmt: skip
    axis.text(0.0, -0.2, str(row["source"]), transform=axis.transAxes, ha="left", va="top",
              fontsize=_FONT["small"], color=INK_SECONDARY)  # fmt: skip


def plot_tactical_case_study(
    engagements: pd.DataFrame,
    others: Sequence[pd.DataFrame | None],
    pitch_dims: Sequence[tuple[float, float]],
    *,
    half_angle_deg: float = 100.0,
    headline: str = "Who is pressed from the blind side? Two engagements from above",
    caption: str | None = None,
) -> Figure:
    """One pitch panel per engagement with both blind-side labels drawn into it.

    Each panel shows the carrier (filled) and the presser (ring), the head facing direction
    (solid arrow) with the visual field around it, the direction of travel (dashed arrow)
    with the field the XY proxy assumes around it (dashed outline), and the other players
    in grey. The edges of the visual field carry two bands: the dark one is the lower bound
    of the angular error (frame-to-frame jitter), the light one the upper bound (Monte
    Carlo under independent joint errors).

    Parameters
    ----------
    engagements
        One row per panel with the columns of
        :func:`blindside.analysis.pipeline.engagement_geometry` (``carrier_x``,
        ``carrier_y``, ``presser_x``, ``presser_y``, ``travel_dx``, ``travel_dy``,
        ``carrier_speed``, ``approach_angle_heading_deg``, ``head_deg``,
        ``head_sd_lower_deg``, ``head_sd_upper_deg``, ``p_head_lower``, ``p_head_upper``)
        plus ``title`` (panel heading) and ``source`` (one line naming match and time).
    others
        Per panel the remaining players at the engagement start: ``x``, ``y`` in metres and
        the boolean ``pressing_team``. None draws no context.
    pitch_dims
        Per panel ``(pitch_length, pitch_width)`` in metres.
    half_angle_deg
        Half-angle of the visual field, also the threshold of the XY proxy.
    headline, caption
        Text above and below the panels.

    Raises
    ------
    ValueError
        If the three inputs do not have one entry per panel, or there is no panel.
    """
    n = len(engagements)
    if n == 0 or len(others) != n or len(pitch_dims) != n:
        raise ValueError("engagements, others and pitch_dims need one entry per panel")
    head = SEGMENT_COLORS["head"]
    with style_context(), mpl.rc_context({"axes.grid": False}):
        figure = Figure(figsize=(8.0 * n, 9.0))
        axes = np.atleast_1d(
            figure.subplots(1, n, gridspec_kw={"left": 0.03, "right": 0.97, "top": 0.845,
                                               "bottom": 0.31, "wspace": 0.06})
        )  # fmt: skip
        for axis, (_, row), context, dims in zip(
            axes, engagements.iterrows(), others, pitch_dims, strict=True
        ):
            _draw_panel(axis, row, context, dims, half_angle_deg)
        figure.suptitle(headline, x=0.03, y=0.965, ha="left", fontsize=_FONT["suptitle"],
                        fontweight="bold", color=INK)  # fmt: skip
        handles = [
            Line2D([], [], color=head, lw=3.2, marker=">", ms=10, label="Head facing (ear axis)"),
            Patch(color=head, alpha=0.22,
                  label=f"Visual field, {half_angle_deg:g} degrees to each side"),
            Patch(color=head, alpha=0.75,
                  label="Edge of the field: error bracket, lower (dark) to upper (light)"),
            Line2D([], [], color=INK, lw=2.4, ls=(0, (4, 2)), marker=">", ms=9,
                   label="Direction of travel and the field the XY proxy assumes"),
            Line2D([], [], color=PITCH_LINE, lw=0, marker="o", ms=11,
                   label="Team in possession (ring: pressing team)"),
        ]  # fmt: skip
        figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.022, 0.07), ncols=3,
                      fontsize=_FONT["small"], labelcolor=INK, handlelength=2.4,
                      columnspacing=1.8)  # fmt: skip
        if caption:
            figure.text(0.03, 0.012, caption, ha="left", va="bottom", fontsize=_FONT["small"],
                        color=INK_SECONDARY)  # fmt: skip
    return figure


def _draw_run(axis: Axes, path: np.ndarray | None, end: np.ndarray, *, lw: float) -> None:
    """The run of a player before the engagement start, ending at the drawn position."""
    if path is None or len(path) < 2:
        return
    points = np.vstack([np.asarray(path, dtype=float), end])
    axis.plot(points[:, 0], points[:, 1], color=INK, lw=lw, alpha=0.8, zorder=5,
              solid_capstyle="round")  # fmt: skip
    # One dot per half second of tracking, so that the length of the run reads as speed.
    axis.scatter(points[:-1:5, 0], points[:-1:5, 1], s=26, color=INK, zorder=5)


def _info_lines(row: pd.Series, half_angle_deg: float) -> list[tuple[str, str]]:
    """Name and value of each entry of the box next to the pitch."""
    proxy_blind = bool(row["approach_angle_heading_deg"] > half_angle_deg)
    lines = [
        ("Carrier speed", f"{row['carrier_speed']:.1f} m/s"),
        (
            "XY label",
            f"{_verdict(proxy_blind)}, presser {row['approach_angle_heading_deg']:.0f} degrees\n"
            "off the direction of travel",
        ),
        (
            "p_blind_side, head",
            f"{row['p_head_lower']:.2f} lower bound\n{row['p_head_upper']:.2f} upper bound",
        ),
    ]
    if "p_torso_lower" in row and np.isfinite(row["p_torso_lower"]):
        lines.append(
            (
                "p_blind_side, torso",
                f"{row['p_torso_lower']:.2f} lower bound\n{row['p_torso_upper']:.2f} upper bound",
            )
        )
    lines.append(
        (
            "Head angle SD",
            f"{row['head_sd_lower_deg']:.1f} degrees lower bound\n"
            f"{row['head_sd_upper_deg']:.1f} degrees upper bound",
        )
    )
    if "presser_off_head_deg" in row and np.isfinite(row["presser_off_head_deg"]):
        lines.append(("Presser", f"{row['presser_off_head_deg']:.0f} degrees off the head facing"))
    return lines


def plot_scene_cone(
    engagement: pd.Series,
    others: pd.DataFrame | None,
    pitch_dims: tuple[float, float],
    paths: Mapping[str, np.ndarray] | None = None,
    *,
    half_angle_deg: float = 100.0,
    headline: str = "One press, two labels",
    caption: str | None = None,
) -> Figure:
    """One engagement on the pitch with both blind-side labels, the runs and its values.

    The pitch shows what :func:`plot_tactical_case_study` shows for one panel: the head
    facing (solid arrow) with the visual field and, at its two edges, the lower (dark) and
    the upper (light) bound of the angular error, the direction of travel (dashed arrow)
    with the field the XY proxy assumes (dashed outline), carrier, presser and the other
    players. On top of that it draws the run of presser and carrier before the engagement
    start and, next to the pitch, a box with the exact values.

    Parameters
    ----------
    engagement
        One row of :func:`blindside.analysis.pipeline.engagement_geometry`. Optional
        entries: ``presser_off_head_deg``, ``carrier_name``, ``presser_name``, ``source``.
    others
        The remaining players at the engagement start: ``x``, ``y`` and the boolean
        ``pressing_team``. None draws no context.
    pitch_dims
        ``(pitch_length, pitch_width)`` in metres.
    paths
        ``{"presser": xy, "carrier": xy}`` with positions ``(n, 2)`` before the engagement
        start, oldest first, one per tracking frame. Missing entries draw no run.
    half_angle_deg
        Half-angle of the visual field, also the threshold of the XY proxy.
    headline, caption
        Text above and below the drawing.
    """
    paths = paths or {}
    head = SEGMENT_COLORS["head"]
    row = engagement
    proxy_blind = bool(row["approach_angle_heading_deg"] > half_angle_deg)
    pose_blind = bool(row["p_head_lower"] > 0.5)
    with style_context(), mpl.rc_context({"axes.grid": False}):
        figure = Figure(figsize=(16.0, 9.0))
        axis = figure.add_axes((0.03, 0.2, 0.6, 0.64))
        _draw_pitch_window(axis, row, others, pitch_dims, half_angle_deg)
        for player in ("carrier", "presser"):
            end = np.array([float(row[f"{player}_x"]), float(row[f"{player}_y"])])
            _draw_run(axis, paths.get(player), end, lw=3.0 if player == "presser" else 2.0)
        figure.suptitle(headline, x=0.03, y=0.955, ha="left", fontsize=_FONT["suptitle"],
                        fontweight="bold", color=INK)  # fmt: skip
        figure.text(
            0.03, 0.875,
            f"XY proxy: {_verdict(proxy_blind)}.  Head pose: {_verdict(pose_blind)}.",
            fontsize=_FONT["title"], fontweight="bold", color=INK, ha="left", va="center",
        )  # fmt: skip

        left, top, step = 0.665, 0.82, 0.104
        figure.add_artist(
            FancyBboxPatch((left - 0.012, 0.2), 0.325, 0.64, transform=figure.transFigure,
                           boxstyle="round,pad=0.0,rounding_size=0.008", fc=SURFACE, ec=GRID,
                           lw=1.4)
        )  # fmt: skip
        for index, (name, value) in enumerate(_info_lines(row, half_angle_deg)):
            y = top - index * step
            figure.text(left, y, name, fontsize=_FONT["small"], color=INK_SECONDARY, ha="left",
                        va="top")  # fmt: skip
            figure.text(left, y - 0.026, value, fontsize=_FONT["label"], color=INK, ha="left",
                        va="top", fontweight="bold", linespacing=1.25)  # fmt: skip

        handles = [
            Line2D([], [], color=head, lw=3.2, marker=">", ms=10, label="Head facing (ear axis)"),
            Patch(color=head, alpha=0.22,
                  label=f"Visual field, {half_angle_deg:g} degrees to each side"),
            Patch(color=head, alpha=0.75,
                  label="Edge of the field: error bracket, lower (dark) to upper (light)"),
            Line2D([], [], color=INK, lw=2.4, ls=(0, (4, 2)), marker=">", ms=9,
                   label="Direction of travel and the field the XY proxy assumes"),
            Line2D([], [], color=INK, lw=3.0, marker="o", ms=5,
                   label="Run before the press (dot every 0.5 s)"),
            Line2D([], [], color=PITCH_LINE, lw=0, marker="o", ms=11,
                   label="Team in possession (ring: pressing team)"),
        ]  # fmt: skip
        figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.022, 0.065), ncols=3,
                      fontsize=_FONT["small"], labelcolor=INK, handlelength=2.4,
                      columnspacing=1.8)  # fmt: skip
        footer = []
        if "carrier_name" in row and "presser_name" in row:
            footer.append(f"Carrier: {row['carrier_name']}. Presser: {row['presser_name']}.")
        if "source" in row:
            footer.append(str(row["source"]))
        if footer:
            figure.text(0.03, 0.18, "  ".join(footer), ha="left", va="top",
                        fontsize=_FONT["small"], color=INK_SECONDARY)  # fmt: skip
        if caption:
            figure.text(0.03, 0.012, caption, ha="left", va="bottom", fontsize=_FONT["small"],
                        color=INK_SECONDARY)  # fmt: skip
    return figure
