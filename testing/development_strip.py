"""Content-bound, halo-supported depth strips for longitudinal development."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pipeline.config import PIPELINE_VERSION, fingerprint
from pipeline.preprocess import (
    MANIFEST_NAME,
    RECORDING_MANIFEST_SCHEMA,
    _validate_materialized_recording,
    recording_binary_receipt,
    recording_geometry_receipt,
    validate_accepted_recording,
)
from pipeline.runtime import production_environment_receipt


STRIP_SCHEMA = "longitudinal-development-strip-v1"


def repository_receipt(root: Path | None = None) -> dict[str, Any]:
    """Record the exact tracked source and dirty tracked diff identity."""
    root = Path(root or Path(__file__).resolve().parents[1])
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "HEAD", "--binary", "--no-ext-diff"],
            check=True, capture_output=True,
        ).stdout.encode()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot establish repository identity") from error
    return {
        "git_commit": commit,
        "tracked_dirty": bool(diff),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def select_depth_channels(
    recording,
    *,
    processing_depth_um: tuple[float, float] | list[float],
    scoring_depth_um: tuple[float, float] | list[float],
) -> dict[str, Any]:
    """Select physical channel IDs without assuming probe row pitch or ordering."""
    locations = np.asarray(recording.get_channel_locations(), dtype=float)
    channel_ids = np.asarray(recording.get_channel_ids())
    if locations.ndim != 2 or locations.shape[0] != len(channel_ids) or locations.shape[1] < 2:
        raise ValueError("recording must expose one finite x/y location per channel")
    if not np.isfinite(locations).all():
        raise ValueError("channel geometry contains non-finite coordinates")
    p0, p1 = map(float, processing_depth_um)
    s0, s1 = map(float, scoring_depth_um)
    if not p0 < s0 < s1 < p1:
        raise ValueError("scoring depth must be strictly inside processing depth")
    depth = locations[:, 1]
    processing_mask = (depth >= p0) & (depth <= p1)
    scoring_mask = (depth >= s0) & (depth <= s1)
    if not np.any(processing_mask):
        raise ValueError("processing depth selects no channels")
    if not np.any(scoring_mask):
        raise ValueError("scoring depth selects no channels")
    if np.any(scoring_mask & ~processing_mask):
        raise ValueError("scoring channels are not a subset of processing channels")
    selected_locations = locations[processing_mask]
    return {
        "processing_channel_ids": channel_ids[processing_mask].tolist(),
        "interior_channel_ids": channel_ids[scoring_mask].tolist(),
        "halo_channel_ids": channel_ids[processing_mask & ~scoring_mask].tolist(),
        "processing_channel_locations_um": selected_locations.tolist(),
        "available_depth_range_um": [float(depth.min()), float(depth.max())],
    }


def classify_unit_depths(
    depths_um: np.ndarray,
    *,
    processing_depth_um: tuple[float, float] | list[float],
    scoring_depth_um: tuple[float, float] | list[float],
    minimum_edge_exclusion_um: float,
) -> np.ndarray:
    """Return stable `interior`, `halo`, `edge`, or `outside` unit labels."""
    depths = np.asarray(depths_um, dtype=float)
    if not np.isfinite(depths).all():
        raise ValueError("unit depths must be finite")
    p0, p1 = map(float, processing_depth_um)
    s0, s1 = map(float, scoring_depth_um)
    labels = np.full(depths.shape, "outside", dtype="<U8")
    in_processing = (depths >= p0) & (depths <= p1)
    labels[in_processing] = "halo"
    edge = in_processing & (
        (depths - p0 <= minimum_edge_exclusion_um)
        | (p1 - depths <= minimum_edge_exclusion_um)
    )
    labels[edge] = "edge"
    labels[(depths >= s0) & (depths <= s1)] = "interior"
    return labels


def _frame_range(recording, recording_spec: Mapping[str, Any]) -> tuple[int, int]:
    fs = float(recording.get_sampling_frequency())
    source_frames = int(recording.get_num_samples())
    start = int(round(float(recording_spec.get("start_s", 0.0)) * fs))
    count = int(round(float(recording_spec["duration_s"]) * fs))
    end = start + count
    if start < 0 or end > source_frames or start >= end:
        raise ValueError("requested duration lies outside the accepted recording")
    return start, end


def materialize_development_strip(
    accepted_recording_dir: Path | str,
    output_dir: Path | str,
    *,
    recording_spec: Mapping[str, Any],
    spatial_spec: Mapping[str, Any],
    n_jobs: int = 1,
    chunk_duration: str = "10s",
):
    """Materialize a validated channel/frame slice, or reuse its exact identity."""
    from spikeinterface.core import load

    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    source_dir = Path(accepted_recording_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == Path("/mnt") or output_dir.is_relative_to(Path("/mnt")):
        raise ValueError("refusing to write a development strip under /mnt")
    if output_dir == source_dir or output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir):
        raise ValueError("development output must be disjoint from the accepted source")
    source_manifest = validate_accepted_recording(source_dir)
    expected = {
        "request_digest": recording_spec["recording_digest"],
        "recording_content_sha256": recording_spec["recording_content_sha256"],
        "probe_geometry_hash": recording_spec["probe_geometry_hash"],
    }
    for name, value in expected.items():
        if source_manifest.get(name) != value:
            raise RuntimeError(f"accepted recording {name} differs from the experiment specification")

    source = load(source_dir)
    geometry = recording_geometry_receipt(source)
    if geometry["probe_geometry_hash"] != recording_spec["probe_geometry_hash"]:
        raise RuntimeError("loaded recording geometry differs from the experiment specification")
    selection = select_depth_channels(
        source,
        processing_depth_um=spatial_spec["processing_depth_um"],
        scoring_depth_um=spatial_spec["scoring_depth_um"],
    )
    start_frame, end_frame = _frame_range(source, recording_spec)
    selected = source.channel_slice(selection["processing_channel_ids"])
    selected = selected.frame_slice(start_frame=start_frame, end_frame=end_frame)
    selected_geometry = recording_geometry_receipt(selected)
    request = {
        "schema_version": STRIP_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "source_path": str(source_dir),
        "source_recording_request_digest": source_manifest["request_digest"],
        "source_recording_content_sha256": source_manifest["recording_content_sha256"],
        "source_probe_geometry_hash": geometry["probe_geometry_hash"],
        "selected_start_frame": start_frame,
        "selected_end_frame": end_frame,
        "spatial_contract": dict(spatial_spec),
        "selection": selection,
        "selected_geometry": selected_geometry,
    }
    request_digest = fingerprint(request)
    manifest_path = output_dir / MANIFEST_NAME
    if output_dir.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"existing strip lacks {MANIFEST_NAME}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request_digest") != request_digest:
            raise RuntimeError("existing strip belongs to another request")
        validate_accepted_recording(output_dir, manifest)
        return load(output_dir), manifest

    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"partial strip requires inspection: {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    selected.save(
        folder=partial,
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        progress_bar=True,
    )
    integrity = _validate_materialized_recording(partial, selected)
    binary_receipt = recording_binary_receipt(partial)
    manifest = {
        "schema_version": RECORDING_MANIFEST_SCHEMA,
        **request,
        "strip_schema_version": STRIP_SCHEMA,
        "request_digest": request_digest,
        "num_samples": int(selected.get_num_samples()),
        "num_channels": int(selected.get_num_channels()),
        "sampling_frequency_hz": float(selected.get_sampling_frequency()),
        "dtype": str(selected.dtype),
        "expected_binary_bytes": int(
            selected.get_num_samples()
            * selected.get_num_channels()
            * np.dtype(selected.dtype).itemsize
        ),
        "repository": repository_receipt(),
        "environment": production_environment_receipt(),
        **selected_geometry,
        **binary_receipt,
        "integrity": integrity,
        "complete": True,
    }
    (partial / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, output_dir)
    return load(output_dir), manifest
