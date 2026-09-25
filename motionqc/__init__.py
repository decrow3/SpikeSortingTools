"""Reusable, estimator-independent motion-field quality control."""

from .field import MotionField, load_field
from .reference import canonical_mask, matched_null, shift_test

__all__ = ["MotionField", "load_field", "canonical_mask", "matched_null", "shift_test"]
__version__ = "0.1.0"
