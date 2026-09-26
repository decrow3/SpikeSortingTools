"""Reusable, estimator-independent motion-field quality control."""

from .field import MotionField, load_field
from .crossprobe import ClockMap, fit_clock_map, map_intervals
from .deployment import compose_two_layer, sliding_windows, stitch_fields
from .reference import canonical_mask, matched_null, shift_test

__all__ = ["MotionField", "load_field", "ClockMap", "fit_clock_map", "map_intervals",
           "canonical_mask", "matched_null", "shift_test", "compose_two_layer",
           "sliding_windows", "stitch_fields"]
__version__ = "0.1.0"
