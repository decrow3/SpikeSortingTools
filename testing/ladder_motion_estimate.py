"""External non-rigid motion estimation + pre-sort field qualification — D2a.

The original D2b-1 numerical tolerance envelope is retracted pending corrected
reruns.  Until an independent calibration exists, field qualification fails
closed when error, support, or reproducibility evidence is missing.

`estimate_full_session_motion` runs SpikeInterface `correct_motion` (dredge /
nonrigid presets) over a whole recording and saves the `Motion` plus its
diagnostics. `qualify_field` applies a conservative evidence-completeness gate
**before** any expensive sorting run (D2 sequence step 5).

Estimation reads the recording (allowed) but every write goes to a local
`out_dir`; nothing is written under /mnt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MOTION_ESTIMATE_SCHEMA = "luke-ladder-motion-estimate-v1"


@dataclass(frozen=True)
class FieldGate:
    """Conservative pre-sort gate; unknown evidence is a hard failure."""

    max_abs_displacement_um: float = 60.0     # beyond this the injected-truth calibration does not cover it
    max_estimated_gain_error_fraction: float = 0.30
    min_support_fraction: float = 0.95
    min_split_half_correlation: float = 0.80

    @property
    def digest(self) -> str:
        from pipeline.config import fingerprint

        return fingerprint({"stage": "field_gate", **asdict(self)})


def estimate_full_session_motion(
    recording_dir: Path | str,
    out_dir: Path | str,
    *,
    preset: str = "dredge",
    n_jobs: int = 8,
    estimate_motion_kwargs: dict | None = None,
) -> dict:
    """Estimate full-session motion and save estimator artifacts + diagnostics.

    This does not materialize a corrected recording; application is a separate
    operation that requires a passing qualification receipt.
    """
    out_dir = Path(out_dir)
    if str(out_dir).startswith("/mnt/"):
        raise ValueError("refusing to write motion-correction outputs under /mnt")
    out_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.preprocess import validate_accepted_recording
    from spikeinterface.core import load
    from spikeinterface.preprocessing import astype, bandpass_filter, correct_motion

    source_manifest = validate_accepted_recording(Path(recording_dir))
    recording = bandpass_filter(
        astype(load(recording_dir), "float32"), freq_min=300.0, freq_max=6000.0
    )
    _, motion_info = correct_motion(
        recording,
        preset=preset,
        folder=str(out_dir / "motion"),
        output_motion_info=True,
        overwrite=True,
        estimate_motion_kwargs=estimate_motion_kwargs or {},
        n_jobs=n_jobs,
        progress_bar=True,
    )
    motion = motion_info["motion"]
    diag = _field_diagnostics(motion)
    manifest = {
        "schema": MOTION_ESTIMATE_SCHEMA,
        "recording_dir": str(recording_dir),
        "source_request_digest": source_manifest["request_digest"],
        "source_content_sha256": source_manifest["recording_content_sha256"],
        "preset": preset,
        # `correct_motion(..., folder=...)` stores estimator artifacts here; it
        # does not materialize the lazy corrected recording.  Application is a
        # separate, post-qualification operation.
        "motion_info_dir": str(out_dir / "motion"),
        "corrected_recording_dir": None,
        "diagnostics": diag,
    }
    (out_dir / "motion_estimate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _field_diagnostics(
    motion,
    *,
    support_fraction: float | None = None,
    split_half_correlation: float | None = None,
    estimated_gain_error_fraction: float | None = None,
) -> dict:
    """Displacement magnitude, temporal spectrum, and non-rigid spread of a Motion."""
    disp = np.asarray(motion.displacement[0], dtype=np.float64)   # (n_time, n_space)
    t = np.asarray(motion.temporal_bins_s[0], dtype=np.float64)
    if disp.ndim == 1:
        disp = disp[:, None]
    # Motion has an arbitrary spatial gauge.  Centre each window before any
    # magnitude/spectral summary; the native mean is not an estimable physical
    # bias relative to ground truth.
    centred = disp - disp.mean(axis=0, keepdims=True)

    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    # power spectrum of the (space-averaged) displacement trace
    trace = centred.mean(axis=1)
    freqs = np.fft.rfftfreq(trace.size, d=dt)
    power = np.abs(np.fft.rfft(trace)) ** 2
    total = power.sum() or 1.0
    cutoff = 0.05  # Hz
    low_frac = float(power[freqs <= cutoff].sum() / total)
    # highest frequency holding >=1% of peak power = effective bandwidth
    sig = np.flatnonzero(power >= 0.01 * power.max())
    bandwidth = float(freqs[sig.max()]) if sig.size else 0.0

    return {
        "n_time_bins": int(disp.shape[0]),
        "n_spatial_windows": int(disp.shape[1]),
        "max_abs_displacement_um": float(np.abs(centred).max()),
        "p2p_displacement_um": float(centred.max() - centred.min()),
        "temporal_bandwidth_hz": bandwidth,
        "low_freq_power_fraction": low_frac,
        "nonrigid_range_um": float(
            np.median(disp.max(axis=1) - disp.min(axis=1)) if disp.shape[1] > 1 else 0.0
        ),
        "support_fraction": support_fraction,
        "split_half_correlation": split_half_correlation,
        "estimated_gain_error_fraction": estimated_gain_error_fraction,
    }


def qualify_field(diagnostics: dict, gate: FieldGate | None = None) -> dict:
    """Pass/fail a field against the D2b-1 provisional pre-sort envelope."""
    gate = gate or FieldGate()
    checks = {
        "displacement_in_calibrated_range":
            diagnostics["max_abs_displacement_um"] <= gate.max_abs_displacement_um,
        "estimation_error_measured_and_tolerable": (
            diagnostics.get("estimated_gain_error_fraction") is not None
            and diagnostics["estimated_gain_error_fraction"]
            <= gate.max_estimated_gain_error_fraction
        ),
        "support_measured_and_sufficient": (
            diagnostics.get("support_fraction") is not None
            and diagnostics["support_fraction"] >= gate.min_support_fraction
        ),
        "split_half_reproducible": (
            diagnostics.get("split_half_correlation") is not None
            and diagnostics["split_half_correlation"] >= gate.min_split_half_correlation
        ),
    }
    return {
        "gate_digest": gate.digest,
        "checks": checks,
        "passes": all(checks.values()),
        "failed": [k for k, v in checks.items() if not v],
        "note": (
            "Field shape alone cannot establish accuracy. Missing support, "
            "split-half, or error evidence fails closed."
        ),
    }


def _motion_digest(motion) -> str:
    digest = hashlib.sha256()
    for arrays in (motion.temporal_bins_s, motion.spatial_bins_um, motion.displacement):
        for value in arrays:
            a = np.ascontiguousarray(value)
            digest.update(str(a.dtype).encode())
            digest.update(str(a.shape).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def materialize_qualified_correction(
    recording_dir: Path | str,
    motion_info_dir: Path | str,
    out_dir: Path | str,
    *,
    qualification: dict,
    n_jobs: int = 8,
) -> dict:
    """Apply a qualified field to the accepted, unfiltered sorter input.

    Motion is estimated on a filtered view, but correction must be applied to
    the original accepted recording.  This function fails closed unless the
    independent-evidence qualification gate passed.
    """
    if not qualification.get("passes"):
        raise ValueError("refusing to materialize an unqualified motion field")
    out_dir = Path(out_dir)
    if str(out_dir).startswith("/mnt/"):
        raise ValueError("refusing to write motion-correction outputs under /mnt")

    from pipeline.config import PIPELINE_VERSION, fingerprint
    from pipeline.preprocess import (
        MANIFEST_NAME,
        RECORDING_MANIFEST_SCHEMA,
        recording_binary_receipt,
        validate_accepted_recording,
    )
    from spikeinterface.core import load
    from spikeinterface.preprocessing import astype
    from spikeinterface.preprocessing.motion import load_motion_info
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

    source_manifest = validate_accepted_recording(Path(recording_dir))
    motion = load_motion_info(Path(motion_info_dir))["motion"]
    corrected = InterpolateMotionRecording(
        astype(load(recording_dir), "float32"),
        motion,
        border_mode="force_extrapolate",
        spatial_interpolation_method="kriging",
        sigma_um=20.0,
    )
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite accepted recording: {out_dir}")
    corrected.save(folder=out_dir, dtype="int16", n_jobs=n_jobs, progress_bar=True)
    positions = np.asarray(corrected.get_channel_locations(), dtype=np.float64)
    np.save(out_dir / "channel_positions.npy", positions)
    receipt = recording_binary_receipt(out_dir)
    request = {
        "pipeline_version": PIPELINE_VERSION,
        "kind": "qualified_motion_corrected_recording",
        "source_request_digest": source_manifest["request_digest"],
        "source_content_sha256": source_manifest["recording_content_sha256"],
        "motion_digest": _motion_digest(motion),
        "qualification_gate_digest": qualification["gate_digest"],
    }
    manifest = {
        "schema_version": RECORDING_MANIFEST_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "kind": request["kind"],
        "complete": True,
        "request_digest": fingerprint(request),
        "num_samples": int(corrected.get_num_samples()),
        "num_channels": int(corrected.get_num_channels()),
        "sampling_frequency_hz": float(corrected.get_sampling_frequency()),
        "dtype": "int16",
        "selected_start_frame": 0,
        "selected_end_frame": int(corrected.get_num_samples()),
        "gain_uv_per_count": float(source_manifest["gain_uv_per_count"]),
        "expected_binary_bytes": receipt["actual_binary_bytes"],
        "recording_content_sha256": receipt["recording_content_sha256"],
        "recording_binary_files": receipt["recording_binary_files"],
        "motion_digest": request["motion_digest"],
        "qualification": qualification,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
