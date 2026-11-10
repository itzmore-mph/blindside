"""Circular statistics for angles in radians.

Angles are directions in the pitch plane: 0 points along +x, counter-clockwise is positive.
Nothing here averages angles arithmetically. All functions are vectorised with NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


def wrap_angle(angle: ArrayLike) -> FloatArray:
    """Wrap angles to the half-open interval (-pi, pi]."""
    angle = np.asarray(angle, dtype=float)
    # pi - ((pi - a) mod 2 pi) maps -pi to +pi, which keeps the interval half-open at -pi.
    return np.pi - np.mod(np.pi - angle, 2.0 * np.pi)


def angular_difference(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """Signed smallest rotation from ``b`` to ``a``, in (-pi, pi].

    Positive when ``a`` is counter-clockwise of ``b``.
    """
    return wrap_angle(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def _mean_vector(
    angles: ArrayLike, axis: int | None, weights: ArrayLike | None
) -> tuple[FloatArray, FloatArray]:
    """Weighted mean of the unit vectors ``(cos, sin)`` along ``axis``."""
    angles = np.asarray(angles, dtype=float)
    if weights is None:
        return np.mean(np.cos(angles), axis=axis), np.mean(np.sin(angles), axis=axis)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), angles.shape)
    total = np.sum(weights, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.sum(weights * np.cos(angles), axis=axis) / total
        s = np.sum(weights * np.sin(angles), axis=axis) / total
    return c, s


def resultant_length(
    angles: ArrayLike, axis: int | None = None, weights: ArrayLike | None = None
) -> FloatArray:
    """Mean resultant length R in [0, 1]. 1 means all angles are identical."""
    c, s = _mean_vector(angles, axis, weights)
    return np.clip(np.hypot(c, s), 0.0, 1.0)


def circular_mean(
    angles: ArrayLike, axis: int | None = None, weights: ArrayLike | None = None
) -> FloatArray:
    """Circular mean direction in (-pi, pi].

    The mean is undefined when the unit vectors cancel exactly (R = 0). In that case the
    result is whatever ``arctan2`` returns for a vector of rounding noise, so check
    :func:`resultant_length` or :func:`circular_std` before trusting it. NaN inputs propagate.
    """
    c, s = _mean_vector(angles, axis, weights)
    return wrap_angle(np.arctan2(s, c))


def circular_std(
    angles: ArrayLike, axis: int | None = None, weights: ArrayLike | None = None
) -> FloatArray:
    """Circular standard deviation ``sqrt(-2 ln R)`` in radians.

    Equals the ordinary standard deviation for a tightly concentrated sample and grows
    without bound as the sample approaches a uniform distribution on the circle.
    """
    r = resultant_length(angles, axis, weights)
    with np.errstate(divide="ignore"):
        return np.sqrt(-2.0 * np.log(r))
