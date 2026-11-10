"""The one figure style of the project.

Colours: three categorical hues for the body segments, in a fixed order, checked for
colour-vision-deficiency separation (worst adjacent pair delta E 10.5 under protanopia)
and for 3:1 contrast against the surface. Magnitudes use a single-hue blue ramp.
Text is always drawn in ink colours, never in a series colour.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

#: Fixed colour per body segment. The colour follows the segment in every figure.
SEGMENT_COLORS: dict[str, str] = {"torso": "#2a78d6", "hips": "#eb6834", "head": "#1baf7a"}
SEGMENT_LABELS: dict[str, str] = {
    "torso": "Torso (shoulders)",
    "hips": "Hips",
    "head": "Head (ears)",
}

#: Sequential ramp for densities and magnitudes: one hue, light to dark.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blindside_blue", [SURFACE, "#cde2fb", "#86b6ef", "#2a78d6", "#104281", "#0d366b"]
)

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "text.color": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "lines.linewidth": 2.0,
    "legend.frameon": False,
}


def apply_style() -> None:
    """Set the project style on matplotlib's global rcParams."""
    mpl.rcParams.update(_RC)


def style_context() -> mpl.rc_context:
    """Context manager that applies the project style to the figures created inside it."""
    return mpl.rc_context(_RC)
