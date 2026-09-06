"""Post-sort motion coordinates without voltage resampling.

The accepted rescue recording and Kilosort output remain unchanged.  This
module consumes an independently qualified motion-field artifact and emits a
provenance-guarded per-spike sidecar for QC and future unit-family stitching.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import PIPELINE_VERSION, fingerprint
from .sorting import SORT_MANIFEST


MOTION_FIELD_SCHEMA = "qualified-motion-field-v1"
MOTION_COORDINATE_SCHEMA = "motion-coordinate-sidecar-v1"
COORDINATE_MANIFEST = "motion_coordinate_manifest.json"


def _scalar(values: np.lib.npyio.NpzFile, name: str) -> Any:
    value = np.asarray(values[name])
    if value.size != 1:
        raise ValueError(f"Motion-field {name} must be scalar")
    return value.reshape(()).item()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_qualified_motion_field(
    path: Path, *, recording_duration_s: float | None = None
) -> dict[str, Any]:
    """Load and strictly validate the estimator-to-pipeline handoff artifact.

    Required NPZ members are ``schema_version``, ``qualification_passed``,
    ``time_reference``, ``displacement_um``, ``time_s``, ``depth_um``,
    ``support``, and ``confidence``.  Time must be relative to the beginning of
    the selected/materialized recording, not the original acquisition.

    ``recording_duration_s`` turns that last sentence from a declaration into a
    check.  A field may *say* ``selected_recording_start`` and carry acquisition
    -clock values: SpikeInterface writes motion time bins in acquisition time,
    so on Luke 2025-08-04 imec0 every estimator's axis runs 3058.7-13530.7 s on
    a 10473.6 s recording, offset by the SpikeGLX ``firstSample`` origin.  The
    string alone cannot catch that, and interpolating against the wrong clock
    misplaces every displacement by the origin.  Supply the duration whenever
    the caller knows it.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as values:
        required = {
            "schema_version",
            "qualification_passed",
            "qualification_digest",
            "time_reference",
            "depth_reference",
            "displacement_convention",
            "estimator",
            "polarity",
            "displacement_um",
            "time_s",
            "depth_um",
            "support",
            "confidence",
        }
        missing = required - set(values.files)
        if missing:
            raise ValueError(f"Motion field is missing members: {sorted(missing)}")
        schema = str(_scalar(values, "schema_version"))
        if np.asarray(values["qualification_passed"]).dtype.kind != "b":
            raise ValueError("Motion-field qualification_passed must be boolean")
        qualified = bool(_scalar(values, "qualification_passed"))
        time_reference = str(_scalar(values, "time_reference"))
        depth_reference = str(_scalar(values, "depth_reference"))
        displacement_convention = str(_scalar(values, "displacement_convention"))
        displacement = np.asarray(values["displacement_um"], dtype=np.float64)
        times = np.asarray(values["time_s"], dtype=np.float64).reshape(-1)
        depths = np.asarray(values["depth_um"], dtype=np.float64).reshape(-1)
        support = np.asarray(values["support"], dtype=np.float64)
        confidence = np.asarray(values["confidence"], dtype=np.float64)
        metadata = {
            name: str(_scalar(values, name))
            for name in ("estimator", "polarity", "qualification_digest")
        }
    if schema != MOTION_FIELD_SCHEMA:
        raise ValueError(f"Unsupported motion-field schema {schema!r}")
    if not qualified:
        raise ValueError("Motion field has not passed independent qualification")
    if time_reference != "selected_recording_start":
        raise ValueError(
            "Motion-field time_reference must be 'selected_recording_start'"
        )
    if depth_reference != "probe_y_um":
        raise ValueError("Motion-field depth_reference must be 'probe_y_um'")
    if displacement_convention != "observed_depth_offset_um":
        raise ValueError(
            "Motion-field displacement_convention must be 'observed_depth_offset_um'"
        )
    if any(not value for value in metadata.values()):
        raise ValueError(
            "Estimator, polarity, and qualification digest must be nonempty"
        )
    shape = (times.size, depths.size)
    if times.size < 2 or not depths.size or displacement.shape != shape:
        raise ValueError(f"Motion field must have time-by-depth shape {shape}")
    if support.shape != shape or confidence.shape != shape:
        raise ValueError("Support and confidence must match displacement shape")
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("Motion-field times must be finite and strictly increasing")
    if not np.all(np.isfinite(depths)) or not np.all(np.diff(depths) > 0):
        raise ValueError("Motion-field depths must be finite and strictly increasing")
    if recording_duration_s is not None:
        tolerance = max(1.0, 0.001 * float(recording_duration_s))
        if times[0] < -tolerance or times[-1] > float(recording_duration_s) + tolerance:
            raise ValueError(
                f"Motion field declares time_reference 'selected_recording_start' but its times "
                f"span [{times[0]:.3f}, {times[-1]:.3f}] s, outside the recording "
                f"[0, {float(recording_duration_s):.3f}] s. A field written on the acquisition "
                "clock looks exactly like this; map it with the recording's t_start before use."
            )
    if np.any(~np.isfinite(support)) or np.any(support < 0):
        raise ValueError("Motion-field support must be finite and nonnegative")
    if np.any(~np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Motion-field confidence must lie in [0, 1]")
    if np.any((support > 0) & ~np.isfinite(displacement)):
        raise ValueError("Supported motion bins must have finite displacement")
    return {
        "displacement_um": displacement,
        "time_s": times,
        "depth_um": depths,
        "support": support,
        "confidence": confidence,
        "metadata": metadata,
    }


def _bracket(
    grid: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if grid.size == 1:
        zeros = np.zeros(query.size, dtype=np.int64)
        return zeros, zeros, np.zeros(query.size), np.ones(query.size, dtype=bool)
    high = np.searchsorted(grid, query, side="right")
    high = np.clip(high, 1, grid.size - 1)
    low = high - 1
    fraction = (query - grid[low]) / (grid[high] - grid[low])
    in_domain = (query >= grid[0]) & (query <= grid[-1])
    return low, high, fraction, in_domain


def interpolate_motion_at_spikes(
    field: dict[str, Any],
    spike_time_s: np.ndarray,
    raw_depth_um: np.ndarray,
    *,
    min_support: float = 1.0,
    min_confidence: float = 0.5,
) -> dict[str, np.ndarray]:
    """Conservatively bilinear-sample a field at spike time/depth coordinates.

    A spike is supported only when every interpolation corner with nonzero
    weight passes both gates.  No time or depth extrapolation is performed.
    A one-bin depth axis is interpreted as a rigid field.
    """
    times = np.asarray(spike_time_s, dtype=np.float64).reshape(-1)
    depths = np.asarray(raw_depth_um, dtype=np.float64).reshape(-1)
    if times.shape != depths.shape:
        raise ValueError("Spike times and depths must have equal shape")
    if min_support < 0 or not 0 <= min_confidence <= 1:
        raise ValueError("Invalid support/confidence threshold")
    ti0, ti1, tf, time_domain = _bracket(field["time_s"], times)
    zi0, zi1, zf, depth_domain = _bracket(field["depth_um"], depths)
    weights = (
        (1 - tf) * (1 - zf),
        tf * (1 - zf),
        (1 - tf) * zf,
        tf * zf,
    )
    corners = ((ti0, zi0), (ti1, zi0), (ti0, zi1), (ti1, zi1))
    displacement = np.zeros(times.size, dtype=np.float64)
    interpolated_support = np.zeros(times.size, dtype=np.float64)
    interpolated_confidence = np.zeros(times.size, dtype=np.float64)
    supported = np.isfinite(times) & np.isfinite(depths) & time_domain & depth_domain
    for weight, indices in zip(weights, corners):
        d = field["displacement_um"][indices]
        s = field["support"][indices]
        c = field["confidence"][indices]
        active = weight > np.finfo(np.float64).eps
        supported &= ~active | (
            np.isfinite(d) & (s >= min_support) & (c >= min_confidence)
        )
        displacement += weight * np.nan_to_num(d, nan=0.0)
        interpolated_support += weight * s
        interpolated_confidence += weight * c
    displacement[~supported] = np.nan
    interpolated_support[~supported] = np.nan
    interpolated_confidence[~supported] = np.nan
    return {
        "displacement_um": displacement,
        "support": interpolated_support,
        "confidence": interpolated_confidence,
        "supported": supported,
    }


def build_spikeinterface_motion(
    motion_field_path: Path,
    *,
    gain: float = 1.0,
    min_support: float = 1.0,
    min_confidence: float = 0.5,
):
    """Convert a fully supported qualified field to a SpikeInterface Motion.

    Template matching needs a displacement at every requested grid point.  This
    adapter therefore refuses partially supported fields rather than filling or
    extrapolating them silently.
    """
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("Motion gain must be finite and nonnegative")
    field = load_qualified_motion_field(motion_field_path)
    qualified = (
        np.isfinite(field["displacement_um"])
        & (field["support"] >= min_support)
        & (field["confidence"] >= min_confidence)
    )
    if not np.all(qualified):
        failed = int(qualified.size - np.sum(qualified))
        raise RuntimeError(
            f"Motion-aware matching requires a fully supported field; {failed} bins fail gates"
        )
    from spikeinterface.core.motion import Motion

    return Motion(
        displacement=gain * field["displacement_um"],
        temporal_bins_s=field["time_s"],
        spatial_bins_um=field["depth_um"],
        direction="y",
        interpolation_method="linear",
    )


def motion_aware_peeler_kwargs(
    motion_field_path: Path,
    *,
    gain: float = 1.0,
    min_support: float = 1.0,
    min_confidence: float = 0.5,
    interpolation_time_bin_size_s: float = 1.0,
    motion_step_um: float = 1.0,
) -> dict[str, Any]:
    """Build version-checked kwargs for the newer SI TDC motion-aware peeler."""
    import inspect

    from spikeinterface.sortingcomponents.matching.tdc_peeler import TridesclousPeeler

    required = {
        "motion_aware",
        "motion",
        "interpolation_time_bin_size_s",
        "motion_step_um",
    }
    available = inspect.signature(TridesclousPeeler.__init__).parameters.keys()
    missing = required - available
    if missing:
        raise RuntimeError(
            "Installed SpikeInterface lacks the required motion-aware peeler API: "
            f"{sorted(missing)}"
        )
    motion = build_spikeinterface_motion(
        motion_field_path,
        gain=gain,
        min_support=min_support,
        min_confidence=min_confidence,
    )
    return {
        "motion_aware": True,
        "motion": motion,
        "interpolation_time_bin_size_s": float(interpolation_time_bin_size_s),
        "motion_step_um": float(motion_step_um),
    }


def write_motion_coordinate_sidecar(
    sort_dir: Path,
    motion_field_path: Path,
    output_dir: Path,
    *,
    gain: float = 1.0,
    min_support: float = 1.0,
    min_confidence: float = 0.5,
    chunk_spikes: int = 1_000_000,
) -> dict[str, Any]:
    """Atomically write raw and coordinate-corrected depths for a KS4 sort."""
    sort_dir = Path(sort_dir)
    motion_field_path = Path(motion_field_path)
    output_dir = Path(output_dir)
    partial = output_dir.with_name(output_dir.name + ".partial")
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("Motion gain must be finite and nonnegative")
    if chunk_spikes < 1:
        raise ValueError("chunk_spikes must be positive")
    sort_manifest_path = sort_dir / SORT_MANIFEST
    if not sort_manifest_path.exists():
        raise FileNotFoundError(f"Missing accepted sort manifest: {sort_manifest_path}")
    sort_manifest = json.loads(sort_manifest_path.read_text())
    if not sort_manifest.get("complete"):
        raise RuntimeError("Sort manifest is not marked complete")
    field_sha256 = _sha256_file(motion_field_path)
    request = {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": MOTION_COORDINATE_SCHEMA,
        "sort_request_digest": sort_manifest["request_digest"],
        "motion_field_sha256": field_sha256,
        "gain": float(gain),
        "min_support": float(min_support),
        "min_confidence": float(min_confidence),
    }
    request_digest = fingerprint(request)
    manifest_path = output_dir / COORDINATE_MANIFEST
    if partial.exists():
        raise RuntimeError(
            f"Incomplete motion-coordinate sidecar requires inspection: {partial}"
        )
    if output_dir.exists():
        if not manifest_path.exists():
            raise RuntimeError(f"Existing sidecar lacks {COORDINATE_MANIFEST}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request_digest") != request_digest:
            raise RuntimeError(
                "Existing motion-coordinate sidecar belongs to another request"
            )
        return manifest

    sorter_output = sort_dir / "sorter_output"
    spike_times = np.load(sorter_output / "spike_times.npy", mmap_mode="r").reshape(-1)
    spike_clusters = np.load(
        sorter_output / "spike_clusters.npy", mmap_mode="r"
    ).reshape(-1)
    positions = np.load(sorter_output / "spike_positions.npy", mmap_mode="r")
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("spike_positions.npy must contain x/y columns")
    if not (spike_times.size == spike_clusters.size == positions.shape[0]):
        raise RuntimeError("Kilosort per-spike arrays have inconsistent lengths")
    recording_manifest = json.loads(
        (sort_dir.parent / "recording" / "rescue_recording_manifest.json").read_text()
    )
    if not recording_manifest.get("complete"):
        raise RuntimeError("Recording manifest is not marked complete")
    if sort_manifest.get("recording_request_digest") != recording_manifest.get(
        "request_digest"
    ):
        raise RuntimeError("Accepted sort and recording manifests do not match")
    sampling_frequency_hz = float(recording_manifest["sampling_frequency_hz"])
    field = load_qualified_motion_field(motion_field_path)
    partial.mkdir(parents=True)
    count = int(spike_times.size)
    outputs = {
        "spike_time_s": np.lib.format.open_memmap(
            partial / "spike_time_s.npy", mode="w+", dtype=np.float64, shape=(count,)
        ),
        "spike_cluster": np.lib.format.open_memmap(
            partial / "spike_cluster.npy",
            mode="w+",
            dtype=spike_clusters.dtype,
            shape=(count,),
        ),
        "raw_depth_um": np.lib.format.open_memmap(
            partial / "raw_depth_um.npy", mode="w+", dtype=np.float32, shape=(count,)
        ),
        "motion_corrected_depth_um": np.lib.format.open_memmap(
            partial / "motion_corrected_depth_um.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count,),
        ),
        "displacement_um": np.lib.format.open_memmap(
            partial / "displacement_um.npy", mode="w+", dtype=np.float32, shape=(count,)
        ),
        "support": np.lib.format.open_memmap(
            partial / "support.npy", mode="w+", dtype=np.float32, shape=(count,)
        ),
        "confidence": np.lib.format.open_memmap(
            partial / "confidence.npy", mode="w+", dtype=np.float32, shape=(count,)
        ),
        "supported": np.lib.format.open_memmap(
            partial / "supported.npy", mode="w+", dtype=np.bool_, shape=(count,)
        ),
    }
    supported_spike_count = 0
    for start in range(0, count, chunk_spikes):
        stop = min(count, start + chunk_spikes)
        spike_time_s = (
            np.asarray(spike_times[start:stop], dtype=np.float64)
            / sampling_frequency_hz
        )
        raw_depth_um = np.asarray(positions[start:stop, 1], dtype=np.float64)
        sampled = interpolate_motion_at_spikes(
            field,
            spike_time_s,
            raw_depth_um,
            min_support=min_support,
            min_confidence=min_confidence,
        )
        outputs["spike_time_s"][start:stop] = spike_time_s
        outputs["spike_cluster"][start:stop] = spike_clusters[start:stop]
        outputs["raw_depth_um"][start:stop] = raw_depth_um
        outputs["motion_corrected_depth_um"][start:stop] = (
            raw_depth_um - gain * sampled["displacement_um"]
        )
        for name in ("displacement_um", "support", "confidence", "supported"):
            outputs[name][start:stop] = sampled[name]
        supported_spike_count += int(np.sum(sampled["supported"]))
    for values in outputs.values():
        values.flush()
    del outputs
    manifest = {
        **request,
        "request_digest": request_digest,
        "motion_field": str(motion_field_path.resolve()),
        "motion_field_metadata": field["metadata"],
        "sampling_frequency_hz": sampling_frequency_hz,
        "spike_count": count,
        "supported_spike_count": supported_spike_count,
        "supported_spike_fraction": supported_spike_count / count if count else 0.0,
        "chunk_spikes": int(chunk_spikes),
        "coordinate_formula": "motion_corrected_depth_um = raw_depth_um - gain * displacement_um",
        "voltage_modified": False,
        "complete": True,
    }
    (partial / COORDINATE_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output_dir)
    return manifest
