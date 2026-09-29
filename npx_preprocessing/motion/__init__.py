"""Motion-correction recording operators."""

from .interpolated_motion_si import InterpolatedMotionRecording, bad_channel_insertion
from .lattice_remap_si import (
    ExactLatticeRemapRecording,
    audit_lattice_mappings,
    audit_spikeinterface_nearest_kernel,
    exact_coordinate_mapping,
    nearest_coordinate_mapping,
    sample_stepwise_shifts,
    write_mapping_audit,
)

__all__ = [
    "InterpolatedMotionRecording",
    "bad_channel_insertion",
    "ExactLatticeRemapRecording",
    "audit_lattice_mappings",
    "audit_spikeinterface_nearest_kernel",
    "exact_coordinate_mapping",
    "nearest_coordinate_mapping",
    "sample_stepwise_shifts",
    "write_mapping_audit",
]
