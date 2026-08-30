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

__all__ = [
    "PIPELINE_VERSION",
    "RescueConfig",
    "build_kilosort4_params",
    "build_rescue_recording",
    "fingerprint",
    "materialize_rescue_recording",
    "phase_correct",
    "rescue_kilosort4_overrides",
    "run_kilosort4",
    "select_bad_channel_ids",
    "threshold_points",
    "validate_applied_settings",
    "write_artifact_sidecar",
]
