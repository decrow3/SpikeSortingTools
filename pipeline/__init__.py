"""Minimal SpikeGLX external-reference rescue pipeline.

Historical preprocessing, motion, curation, and QC code lives in
``pipelineold``.  This package intentionally exposes only the tested rescue
graph and its provenance-guarded materialization and sorting helpers.
"""

from .artifacts import threshold_points, write_artifact_sidecar
from .config import PIPELINE_VERSION, RescueConfig, fingerprint
from .preprocess import (
    build_rescue_recording,
    materialize_rescue_recording,
    phase_correct,
    select_bad_channel_ids,
)
from .sorting import (
    build_kilosort4_params,
    rescue_kilosort4_overrides,
    run_kilosort4,
    validate_applied_settings,
)
from .motion_coordinates import (
    MOTION_COORDINATE_SCHEMA,
    MOTION_FIELD_SCHEMA,
    interpolate_motion_at_spikes,
    build_spikeinterface_motion,
    load_qualified_motion_field,
    write_motion_coordinate_sidecar,
    motion_aware_peeler_kwargs,
)
from .bakeoff import (
    BAKEOFF_SCHEMA,
    CANDIDATES,
    accept_ks4_reference,
    build_bakeoff_plan,
    inspect_bakeoff_environment,
    run_dartsort_challenger,
    run_ks4_seeded_peeler_pair,
    run_kiasort_challenger,
    normalize_dartsort_output,
    resolve_kiasort_installation,
    validate_dartsort_output,
)

__all__ = [
    "PIPELINE_VERSION",
    "RescueConfig",
    "build_kilosort4_params",
    "build_bakeoff_plan",
    "build_rescue_recording",
    "build_spikeinterface_motion",
    "fingerprint",
    "materialize_rescue_recording",
    "MOTION_COORDINATE_SCHEMA",
    "MOTION_FIELD_SCHEMA",
    "BAKEOFF_SCHEMA",
    "CANDIDATES",
    "phase_correct",
    "rescue_kilosort4_overrides",
    "run_kilosort4",
    "run_dartsort_challenger",
    "run_ks4_seeded_peeler_pair",
    "run_kiasort_challenger",
    "normalize_dartsort_output",
    "select_bad_channel_ids",
    "threshold_points",
    "interpolate_motion_at_spikes",
    "load_qualified_motion_field",
    "motion_aware_peeler_kwargs",
    "validate_applied_settings",
    "validate_dartsort_output",
    "accept_ks4_reference",
    "inspect_bakeoff_environment",
    "resolve_kiasort_installation",
    "write_artifact_sidecar",
    "write_motion_coordinate_sidecar",
]
