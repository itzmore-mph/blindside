"""Orientation geometry, circular statistics and error propagation."""

from blindside.geometry.circular import (
    angular_difference,
    circular_mean,
    circular_std,
    resultant_length,
    wrap_angle,
)
from blindside.geometry.orientation import (
    SEGMENT_JOINTS,
    anatomical_forward,
    facing_angle,
    nose_direction,
    resolve_front_back,
    segment_width,
    vector_angle,
)
from blindside.geometry.uncertainty import (
    DEFAULT_N_SAMPLES,
    P90_TO_SIGMA,
    P90_TO_SIGMA_SENSITIVITY,
    AngularEstimate,
    difference_sigma,
    facing_uncertainty,
    sample_facing_angles,
    sample_vector_angles,
    sigma_from_p90,
    small_angle_std,
    summarise_samples,
    vector_uncertainty,
)

__all__ = [
    "DEFAULT_N_SAMPLES",
    "P90_TO_SIGMA",
    "P90_TO_SIGMA_SENSITIVITY",
    "SEGMENT_JOINTS",
    "AngularEstimate",
    "anatomical_forward",
    "angular_difference",
    "circular_mean",
    "circular_std",
    "difference_sigma",
    "facing_angle",
    "facing_uncertainty",
    "nose_direction",
    "resolve_front_back",
    "resultant_length",
    "sample_facing_angles",
    "sample_vector_angles",
    "segment_width",
    "sigma_from_p90",
    "small_angle_std",
    "summarise_samples",
    "vector_angle",
    "vector_uncertainty",
    "wrap_angle",
]
