"""blindside: blind-side pressing from broadcast body pose, with measurement uncertainty.

The names below are the public API. They are imported on first use, so ``import blindside``
stays light.

Reading data
    :func:`iter_pose_frames`, :func:`sample_pose_file`, :func:`download_pose_archive`,
    :func:`load_dynamic_events`, :func:`load_match_metadata`
Orientation with uncertainty
    :func:`facing_angle`, :func:`facing_uncertainty`, :func:`sample_facing_angles`,
    :func:`bound_correlation`, :func:`circular_mean`, :func:`circular_std`,
    :func:`build_orientation_table`, :func:`cached_orientation_table`,
    :class:`OrientationConfig`
Blind-side pressure
    :func:`blind_side_probability`, :func:`pose_feature_table`
Blind-side labels without body pose
    :func:`xy_blind_side_proxy`, :func:`classify_xy_reliability`
Agreement of two labels
    :func:`cohen_kappa`, :func:`agreement_table`, :func:`cluster_bootstrap_weights`
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_EXPORTS: dict[str, str] = {
    "iter_pose_frames": "blindside.data.pose",
    "sample_pose_file": "blindside.data.opendata",
    "download_pose_archive": "blindside.data.download",
    "load_dynamic_events": "blindside.data.events",
    "load_match_metadata": "blindside.data.events",
    "facing_angle": "blindside.geometry.orientation",
    "facing_uncertainty": "blindside.geometry.uncertainty",
    "sample_facing_angles": "blindside.geometry.uncertainty",
    "bound_correlation": "blindside.geometry.uncertainty",
    "circular_mean": "blindside.geometry.circular",
    "circular_std": "blindside.geometry.circular",
    "OrientationConfig": "blindside.features.orientation_table",
    "build_orientation_table": "blindside.features.orientation_table",
    "cached_orientation_table": "blindside.features.orientation_table",
    "blind_side_probability": "blindside.features.pose_features",
    "pose_feature_table": "blindside.features.pose_features",
    "classify_xy_reliability": "blindside.features.xy_proxy",
    "xy_blind_side_proxy": "blindside.features.xy_proxy",
    "cohen_kappa": "blindside.analysis.agreement",
    "agreement_table": "blindside.analysis.agreement",
    "cluster_bootstrap_weights": "blindside.analysis.baseline",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Import a public name from its module on first access (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'blindside' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
