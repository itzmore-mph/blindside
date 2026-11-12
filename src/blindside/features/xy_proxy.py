"""The XY proxy for a blind-side press, and when it can stand in for body pose.

On a match without body pose the only blind-side label is the XY proxy: the presser comes
from more than the field of view half-angle off the carrier's direction of travel. On the
two matches with body pose that proxy agrees with the pose label beyond chance only when
the carrier moves (docs/INCREMENTAL.md section 1.10, exploratory, no outcome used):

==============  ===============  ==================  ===================
Carrier speed   Engagements      Kappa, head label   Kappa, torso label
==============  ===============  ==================  ===================
below 2 m/s     665              0.09                0.10
2 to 4 m/s      727              0.31                0.41
4 to 6 m/s      370              0.54                0.66
above 6 m/s     113              0.58                0.77
==============  ===============  ==================  ===================

:func:`classify_xy_reliability` turns that table into a filter. "Reliable" means that the
proxy agrees with the pose label, measured on two matches. It does not mean accurate:
there is no ground truth for orientation, so agreement cannot say which label is right.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import ArrayLike

#: Carrier speed in m/s below which the XY proxy agrees with the pose label barely beyond chance.
XY_UNRELIABLE_BELOW_MPS = 2.0
#: Carrier speed in m/s from which the XY proxy is a workable stand-in for the pose label.
XY_RELIABLE_FROM_MPS = 4.0
UNRELIABLE = "unreliable"
INTERMEDIATE = "intermediate"
RELIABLE = "reliable"
UNKNOWN = "unknown"
#: Classes of :func:`classify_xy_reliability`, from the slowest carrier to the fastest.
XY_RELIABILITY_CLASSES: tuple[str, ...] = (UNRELIABLE, INTERMEDIATE, RELIABLE)
#: Threshold of the proxy in degrees, equal to the field of view half-angle of the pose label.
XY_PROXY_ANGLE_DEG = 100.0


@overload
def classify_xy_reliability(speed_ms: float) -> str: ...
@overload
def classify_xy_reliability(speed_ms: ArrayLike) -> np.ndarray: ...
def classify_xy_reliability(speed_ms: ArrayLike) -> str | np.ndarray:
    """Whether an XY-only blind-side label can be trusted at this carrier speed.

    Parameters
    ----------
    speed_ms
        Speed of the ball carrier at the engagement start in m/s, a number or an array.
        The thresholds were measured with the mean speed over the 0.5 s before the start
        (``carrier_speed`` of :func:`blindside.features.xy_baseline.tracking_features`).

    Returns
    -------
    ``"unreliable"`` below 2 m/s, ``"intermediate"`` from 2 to below 4 m/s, ``"reliable"``
    from 4 m/s, and ``"unknown"`` for a missing speed. A string for a number, else an
    object array of the same shape.

    Raises
    ------
    ValueError
        If a speed is negative.

    Examples
    --------
    >>> classify_xy_reliability(1.2)
    'unreliable'
    >>> classify_xy_reliability([1.2, 3.0, 4.0, float("nan")]).tolist()
    ['unreliable', 'intermediate', 'reliable', 'unknown']
    """
    speed = np.asarray(speed_ms, dtype=float)
    if (speed < 0.0).any():
        raise ValueError("speed_ms must not be negative")
    labels = np.array([*XY_RELIABILITY_CLASSES, UNKNOWN], dtype=object)
    index = np.digitize(speed, (XY_UNRELIABLE_BELOW_MPS, XY_RELIABLE_FROM_MPS))
    out = labels[np.where(np.isnan(speed), len(XY_RELIABILITY_CLASSES), index)]
    return str(out) if speed.ndim == 0 else out


def xy_blind_side_proxy(
    approach_angle_heading_deg: ArrayLike, threshold_deg: float = XY_PROXY_ANGLE_DEG
) -> np.ndarray:
    """The XY proxy: blind if the presser comes from behind the carrier's direction of travel.

    Parameters
    ----------
    approach_angle_heading_deg
        Angle in degrees between the carrier's direction of travel and the direction to the
        presser, in [0, 180] (``approach_angle_heading_deg`` of
        :func:`blindside.features.xy_baseline.tracking_features`). NaN for a carrier who
        did not move.
    threshold_deg
        The presser counts as unseen above this angle.

    Returns
    -------
    Boolean array of the same shape. False where the angle is NaN. Read it together with
    :func:`classify_xy_reliability`.
    """
    angle = np.asarray(approach_angle_heading_deg, dtype=float)
    with np.errstate(invalid="ignore"):
        return angle > threshold_deg
