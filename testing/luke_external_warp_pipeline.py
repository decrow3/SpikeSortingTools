"""Execute the frozen, development-only Option A comparison.

The runner consumes one digest-identified motion field.  It does not estimate
motion, calibrate gain, qualify a field, or change the production gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.config import PIPELINE_VERSION, fingerprint
from pipeline.downstream import (
    build_sort_identity,
    pin_sort_identity,
    run_curation_stage,
    run_matlab_export_stage,
    run_qc_stage,
)
from pipeline.preprocess import (
    MANIFEST_NAME,
    RECORDING_MANIFEST_SCHEMA,
    recording_binary_receipt,
    validate_accepted_recording,
)
from pipeline.runtime import validate_production_environment
from pipeline.sorting import run_kilosort4
from testing.luke_candidate_completeness_qc import CompletenessConfig, train_completeness
from testing.luke_rescue_unique_units_audit import exclusive_event_pairs


SCHEMA = "luke-option-a-external-warp-run-v1"
NO_GAIN_ERROR_KEY = "error_fraction"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_no_gain_error(value: Any, path: str = "contract") -> None:
    if isinstance(value, dict):
        if NO_GAIN_ERROR_KEY in value:
            raise ValueError(
                f"{path} contains {NO_GAIN_ERROR_KEY!r}; estimator disagreement is not gain error"
            )
        for key, child in value.items():
            _assert_no_gain_error(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_gain_error(child, f"{path}[{index}]")


def load_contract(path: Path) -> dict[str, Any]:
    """Load the frozen contract and reject reintroduced gain-error language."""
    contract = json.loads(Path(path).read_text())
    _assert_no_gain_error(contract)
    if contract.get("schema") != "luke-option-a-development-comparison-v1":
        raise ValueError("unsupported Option A contract schema")
    if contract.get("status") != "authored":
        raise ValueError("Option A contract is not in its frozen authored state")
    return contract


def _array_digest(path: Path) -> str:
    return _sha256_file(path)


def _field_from_contract(contract: dict[str, Any], field_dir: Path) -> dict[str, Any]:
    field_spec = contract["field"]
    names = {"motion": "motion_npy", "time": "time_bins_npy", "depth": "depth_bins_npy"}
    arrays: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name, key in names.items():
        path = field_dir / Path(field_spec[key]).name
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen field array: {path}")
        observed = _array_digest(path)
        if observed != field_spec[f"{key.removesuffix('_npy')}_sha256"]:
            raise ValueError(f"frozen field digest mismatch for {name}: {path}")
        arrays[name] = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        digests[name] = observed
    displacement = arrays["motion"]
    times = arrays["time"].reshape(-1)
    depths = arrays["depth"].reshape(-1)
    if displacement.shape != (times.size, depths.size):
        raise ValueError("motion array is not time-by-depth")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("frozen field contains non-finite displacement")
    origin = float(field_spec["acquisition_time_origin_s"])
    recording_times = times - origin
    if not np.all(np.diff(times) > 0) or not np.all(np.diff(depths) > 0):
        raise ValueError("frozen field axes must be strictly increasing")
    if recording_times[0] < 0 or recording_times[-1] > contract["recording"]["duration_s"]:
        raise ValueError("mapped field times fall outside the frozen recording")
    return {
        "displacement_um": displacement,
        "acquisition_time_s": times,
        "recording_time_s": recording_times,
        "depth_um": depths,
        "origin_s": origin,
        "digests": digests,
        "field_digest": fingerprint(digests),
    }


def _make_motion(field: dict[str, Any]):
    from spikeinterface.core.motion import Motion

    return Motion(
        displacement=[field["displacement_um"]],
        temporal_bins_s=[field["recording_time_s"]],
        spatial_bins_um=field["depth_um"],
        direction="y",
        interpolation_method="linear",
    )


def load_endpoint_amendment(path: Path) -> dict[str, Any]:
    amendment = json.loads(Path(path).read_text())
    if amendment.get("schema") != "luke-option-a-endpoint-amendment-v1":
        raise ValueError("unsupported Option A endpoint amendment")
    return amendment


def _support_grid(field: dict[str, Any], amendment: dict[str, Any], fs_hz: float) -> dict[str, Any]:
    artifact = amendment["support_artifact"]
    peaks_path = Path(artifact["peaks_npy"])
    locations_path = Path(artifact["peak_locations_npy"])
    if not peaks_path.is_file() or not locations_path.is_file():
        raise FileNotFoundError("declared measured support artifact is unavailable")
    peaks = np.load(peaks_path, mmap_mode="r")
    locations = np.load(locations_path, mmap_mode="r")
    if peaks.shape != locations.shape or "sample_index" not in peaks.dtype.names or "y" not in locations.dtype.names:
        raise ValueError("support artifact peak arrays do not have the declared schema")
    seconds = np.asarray(peaks["sample_index"], dtype=np.float64) / float(fs_hz)
    depths = np.asarray(locations["y"], dtype=np.float64)
    time_bin_s = float(artifact["time_bin_s"])
    depth_bin_s = float(artifact["depth_bin_s"])
    time_centres = np.arange(
        float(field["recording_time_s"][0]),
        float(field["recording_time_s"][-1]) + time_bin_s,
        time_bin_s,
    )
    time_edges = np.arange(
        float(field["recording_time_s"][0]) - time_bin_s / 2,
        float(field["recording_time_s"][-1]) + time_bin_s,
        time_bin_s,
    )
    depth_centres = np.arange(
        float(artifact["depth_grid_first_center_um"]),
        float(np.max(locations["y"])) + depth_bin_s / 2,
        depth_bin_s,
    )
    if depth_centres.size < 2:
        raise ValueError("measured support grid needs at least two depth bins")
    depth_edges = _edges(depth_centres)
    counts, _, _ = np.histogram2d(seconds, depths, bins=[time_edges, depth_edges])
    return {
        "counts": counts,
        "time_bins_s": time_centres,
        "depth_bins_um": depth_centres,
        "minimum_peaks_per_cell": int(artifact["minimum_peaks_per_cell"]),
        "peaks_sha256": _sha256_file(peaks_path),
        "peak_locations_sha256": _sha256_file(locations_path),
        "n_peaks": int(peaks.size),
    }


def _edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=np.float64)
    if centres.size < 2:
        raise ValueError("support grid needs at least two bins")
    steps = np.diff(centres)
    return np.concatenate(([centres[0] - steps[0] / 2], centres[:-1] + steps / 2, [centres[-1] + steps[-1] / 2]))


def _assert_support_policy(
    recording: Any, field: dict[str, Any], contract: dict[str, Any], amendment: dict[str, Any]
) -> dict[str, Any]:
    domain = contract["domain"]
    start_s, stop_s = map(float, domain["interval_s"])
    lo, hi = map(float, domain["depth_band_um"])
    sigma = float(contract["application_policy"]["sigma_um"])
    support = _support_grid(field, amendment, float(recording.get_sampling_frequency()))
    max_displacement = float(np.max(np.abs(field["displacement_um"])))
    margin = max_displacement + 3.0 * sigma
    if field["recording_time_s"][0] > start_s or field["recording_time_s"][-1] < stop_s:
        raise ValueError("field does not cover the complete frozen interval")
    source_y = np.asarray(recording.get_channel_locations(), dtype=float)[:, 1]
    retained = (source_y >= lo) & (source_y <= hi)
    if int(np.count_nonzero(retained)) != int(domain["n_channels_in_band"]):
        raise ValueError("source geometry does not contain the frozen channel band")
    field_times = np.asarray(field["recording_time_s"])
    selected_times = np.flatnonzero((field_times >= start_s) & (field_times < stop_s))
    selected_channels = np.flatnonzero((source_y >= lo) & (source_y <= hi))
    counts = support["counts"]
    required = int(support["minimum_peaks_per_cell"])
    failures = []
    channel_support_fractions = []
    selected_support_times = np.flatnonzero(
        (support["time_bins_s"] >= start_s) & (support["time_bins_s"] < stop_s)
    )
    if selected_support_times.size == 0:
        raise ValueError("measured support artifact has no cells in the frozen interval")
    channel_neighborhoods: dict[int, set[int]] = {}
    for ti in selected_times:
        displacement = np.interp(source_y[selected_channels], field["depth_um"], field["displacement_um"][ti])
        source_depths = source_y[selected_channels] + displacement
        for channel_id, source_depth in zip(selected_channels, source_depths):
            nearby = np.flatnonzero(np.abs(support["depth_bins_um"] - source_depth) <= 3.0 * sigma)
            if nearby.size == 0:
                failures.append((int(ti), int(channel_id)))
                continue
            channel_neighborhoods.setdefault(int(channel_id), set()).update(int(value) for value in nearby)
    for nearby in channel_neighborhoods.values():
        channel_support_fractions.append(
            float(np.mean(counts[np.ix_(selected_support_times, sorted(nearby))].mean(axis=1) >= required))
        )
    support_gate = float(amendment["support_artifact"]["support_fraction_gate"])
    unsupported_channels = int(np.count_nonzero(np.asarray(channel_support_fractions) < support_gate))
    if failures or unsupported_channels:
        raise ValueError(
            f"measured support policy refused {len(failures)} missing neighborhoods and "
            f"{unsupported_channels} channels below the {support_gate:.3f} support fraction; "
            "no extrapolation or domain narrowing is permitted"
        )
    return {
        "policy": "measured_support_grid_and_remove_channels",
        "interval_s": [start_s, stop_s],
        "depth_band_um": [lo, hi],
        "max_abs_displacement_um": max_displacement,
        "interpolation_margin_um": margin,
        "field_support_checked": True,
        "support_artifact": {
            "peaks_sha256": support["peaks_sha256"],
            "peak_locations_sha256": support["peak_locations_sha256"],
            "n_peaks": support["n_peaks"],
            "grid": list(counts.shape),
            "minimum_peaks_per_cell": required,
            "checked_time_bins": int(selected_times.size),
            "checked_channels": int(selected_channels.size),
            "failed_neighborhoods": len(failures),
            "minimum_channel_neighborhood_support_fraction": float(min(channel_support_fractions)),
            "support_fraction_gate": support_gate,
        },
    }


def _accepted_manifest(folder: Path, source_manifest: dict[str, Any], *, request: dict[str, Any]) -> dict[str, Any]:
    from spikeinterface.core import load

    loaded = load(folder)
    receipt = recording_binary_receipt(folder)
    manifest = {
        "schema_version": RECORDING_MANIFEST_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "complete": True,
        **request,
        "request_digest": fingerprint(request),
        "num_samples": int(loaded.get_num_samples()),
        "num_channels": int(loaded.get_num_channels()),
        "sampling_frequency_hz": float(source_manifest["sampling_frequency_hz"]),
        "dtype": "int16",
        "selected_start_frame": 0,
        "selected_end_frame": int(loaded.get_num_samples()),
        "gain_uv_per_count": float(source_manifest["gain_uv_per_count"]),
        "expected_binary_bytes": receipt["actual_binary_bytes"],
        **receipt,
    }
    manifest["channel_ids"] = [str(value) for value in loaded.get_channel_ids()]
    manifest["channel_locations_um"] = np.asarray(
        loaded.get_channel_locations(), dtype=float
    ).tolist()
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _materialize_arm(
    recording: Any,
    output_dir: Path,
    *,
    source_manifest: dict[str, Any],
    request: dict[str, Any],
    n_jobs: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        manifest_path = output_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise RuntimeError(f"ambiguous partial arm recording: {output_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request_digest") != fingerprint(request):
            raise RuntimeError(f"arm recording belongs to another request: {output_dir}")
        validate_accepted_recording(output_dir, manifest)
        return manifest
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"ambiguous interrupted arm recording: {partial}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    recording.save(folder=partial, dtype="int16", n_jobs=n_jobs, progress_bar=True)
    manifest = _accepted_manifest(partial, source_manifest, request=request)
    os.replace(partial, output_dir)
    return manifest


def _apply_external_warp(recording: Any, field: dict[str, Any], contract: dict[str, Any]):
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

    policy = contract["application_policy"]
    return InterpolateMotionRecording(
        astype(recording, "float32"),
        _make_motion(field),
        border_mode=policy["border_mode"],
        spatial_interpolation_method=policy["spatial_interpolation_method"],
        sigma_um=float(policy["sigma_um"]),
    )


def _prepare_recordings(source_dir: Path, root: Path, contract: dict[str, Any], amendment: dict[str, Any], field: dict[str, Any]) -> dict[str, Any]:
    from spikeinterface.core import load

    source_manifest = validate_accepted_recording(source_dir)
    source = load(source_dir)
    if contract["application_policy"]["polarity_convention"] != "output_depth = source_depth - displacement":
        raise ValueError("field convention is not the verified operator convention")
    support = _assert_support_policy(source, field, contract, amendment)
    domain = contract["domain"]
    fs = float(source.get_sampling_frequency())
    start = int(round(float(domain["interval_s"][0]) * fs))
    stop = int(round(float(domain["interval_s"][1]) * fs))
    corrected_full = _apply_external_warp(source, field, contract)
    corrected_channels = np.asarray(corrected_full.get_channel_ids())
    source_channels = np.asarray(source.get_channel_ids())
    band = tuple(map(float, domain["depth_band_um"]))
    source_positions = np.asarray(source.get_channel_locations(), dtype=float)
    expected_ids = source_channels[
        (source_positions[:, 1] >= band[0]) & (source_positions[:, 1] <= band[1])
    ]
    if not np.all(np.isin(expected_ids, corrected_channels)):
        raise RuntimeError("corrected recording removed a channel inside the frozen band")
    control = source.frame_slice(start_frame=start, end_frame=stop).channel_slice(channel_ids=expected_ids)
    corrected = corrected_full.frame_slice(start_frame=start, end_frame=stop).channel_slice(channel_ids=expected_ids)
    if not np.array_equal(np.asarray(control.get_channel_ids()), np.asarray(corrected.get_channel_ids())):
        raise RuntimeError("control and corrected arms have different final channel IDs")
    if not np.array_equal(np.asarray(control.get_channel_locations()), np.asarray(corrected.get_channel_locations())):
        raise RuntimeError("control and corrected arms have different channel ordering/geometry")
    request_base = {
        "schema": SCHEMA,
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "source_content_sha256": source_manifest["recording_content_sha256"],
        "source_request_digest": source_manifest["request_digest"],
        "motion_arrays": field["digests"],
        "field_digest": field["field_digest"],
        "interval_s": domain["interval_s"],
        "depth_band_um": domain["depth_band_um"],
        "channel_ids": [str(value) for value in expected_ids],
        "channel_ordering_hash": fingerprint([str(value) for value in expected_ids]),
        "time_origin_s": field["origin_s"],
        "crop_order": "external_correction_full_geometry_then_time_and_depth_crop",
        "application_policy": contract["application_policy"],
        "sorter_policy": contract["arms"]["control"],
        "evaluation_mask": {"interval_s": domain["interval_s"], "depth_band_um": domain["depth_band_um"], "channel_ids": [str(value) for value in expected_ids]},
    }
    control_request = {**request_base, "arm": "uncorrected", "external_correction": False}
    corrected_request = {**request_base, "arm": "externally_corrected_ks_motion", "external_correction": True}
    control_manifest = _materialize_arm(control, root / "recordings/control", source_manifest=source_manifest, request=control_request, n_jobs=4)
    corrected_manifest = _materialize_arm(corrected, root / "recordings/corrected", source_manifest=source_manifest, request=corrected_request, n_jobs=4)
    for manifest in (control_manifest, corrected_manifest):
        if manifest["num_channels"] != len(expected_ids):
            raise RuntimeError("materialized arm changed the frozen channel count")
    return {"control": control_manifest, "corrected": corrected_manifest, "support": support}


def _run_arm(recording_dir: Path, root: Path) -> dict[str, Any]:
    sort_dir = root / "sort"
    sort_manifest = run_kilosort4(recording_dir, sort_dir)
    identity = pin_sort_identity(sort_dir, root / "sort_identity.json")
    curated = run_curation_stage(sort_dir / "sorter_output", root / "curation", identity)
    qc = run_qc_stage(recording_dir, root / "curation" / "cur_output", root / "qc", identity)
    export = run_matlab_export_stage(root / "curation" / "cur_output", root / "qc", identity)
    return {"sort": sort_manifest, "identity": identity, "curation": curated, "qc": qc, "export": export}


def _waveform_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else float("nan")


def _reciprocal_matches(baseline: dict[str, Any], candidate: dict[str, Any], amendment: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    criteria = amendment["identity_correspondence"]
    candidates_by_baseline: dict[int, list[dict[str, Any]]] = {}
    candidates_by_candidate: dict[int, list[dict[str, Any]]] = {}
    for baseline_id in baseline["good_ids"]:
        for candidate_id in candidate["good_ids"]:
            a_st = baseline["times_by_cluster"].get(int(baseline_id), np.array([], dtype=np.int64))
            b_st = candidate["times_by_cluster"].get(int(candidate_id), np.array([], dtype=np.int64))
            a_hit, b_hit = exclusive_event_pairs(a_st, b_st, int(criteria["event_tolerance_samples"]))
            denominator = max(1, min(a_st.size, b_st.size))
            coincidence = float(a_hit.size / denominator)
            depth_distance = abs(float(baseline["depth_by_cluster"].get(int(baseline_id), np.nan)) - float(candidate["depth_by_cluster"].get(int(candidate_id), np.nan)))
            waveform_cosine = _waveform_cosine(baseline["waveforms"].get(int(baseline_id), np.array([])), candidate["waveforms"].get(int(candidate_id), np.array([])))
            row = {"baseline_cluster": int(baseline_id), "candidate_cluster": int(candidate_id), "exclusive_coincidence_fraction": coincidence, "depth_distance_um": depth_distance, "waveform_cosine": waveform_cosine}
            if coincidence >= float(criteria["minimum_exclusive_coincidence_fraction"]) and depth_distance <= float(criteria["maximum_depth_distance_um"]) and waveform_cosine >= float(criteria["minimum_waveform_cosine"]):
                candidates_by_baseline.setdefault(int(baseline_id), []).append(row)
                candidates_by_candidate.setdefault(int(candidate_id), []).append(row)
    ambiguous = {"baseline": sorted(k for k, values in candidates_by_baseline.items() if len(values) > 1), "candidate": sorted(k for k, values in candidates_by_candidate.items() if len(values) > 1)}
    pairs = [values[0] for key, values in candidates_by_baseline.items() if len(values) == 1 and len(candidates_by_candidate[values[0]["candidate_cluster"]]) == 1]
    return pairs, {"n_pairs": len(pairs), "n_ambiguous_baseline": len(ambiguous["baseline"]), "n_ambiguous_candidate": len(ambiguous["candidate"]), "ambiguous": ambiguous}


def _arm_observations(root: Path, interval_s: tuple[float, float], fs_hz: float) -> dict[str, Any]:
    curated = Path(root) / "curation" / "cur_output"
    qc = Path(root) / "qc"
    recording_manifest = json.loads(
        (Path(root).parent / "recordings" / Path(root).name / MANIFEST_NAME).read_text()
    )
    materialized_duration_s = float(recording_manifest["num_samples"]) / fs_hz
    interval_duration_s = float(interval_s[1]) - float(interval_s[0])
    if abs(materialized_duration_s - interval_duration_s) <= 1.0 / fs_hz:
        local_interval_s = (0.0, materialized_duration_s)
    else:
        local_interval_s = (
            float(interval_s[0]) - float(recording_manifest["selected_start_frame"]) / fs_hz,
            float(interval_s[1]) - float(recording_manifest["selected_start_frame"]) / fs_hz,
        )
    spike_times = np.load(curated / "spike_times.npy", mmap_mode="r").reshape(-1).astype(np.int64)
    spike_clusters = np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1).astype(np.int64)
    full_st = np.load(curated / "full_st.npy", mmap_mode="r")
    kept = np.load(curated / "kept_spikes.npy", mmap_mode="r")
    retained = full_st[kept]
    if retained.shape[0] != spike_times.size or not np.array_equal(retained[:, 0].astype(np.int64), spike_times):
        raise RuntimeError("curated amplitude source does not align with spike_times")
    labels = np.genfromtxt(curated / "cluster_KSLabel.tsv", delimiter="\t", names=True, dtype=None, encoding="utf-8")
    label_name = next(name for name in labels.dtype.names if name != "cluster_id")
    good_ids = [int(cid) for cid, label in zip(labels["cluster_id"], labels[label_name]) if str(label).lower() == "good"]
    positions = np.load(curated / "spike_positions.npy", mmap_mode="r")
    wave_data = np.load(qc / "waveforms" / "waveforms.npz", allow_pickle=False)
    waveforms = {int(cid): wave for cid, wave in zip(wave_data["cids"], wave_data["waveforms"])}
    seconds = spike_times / float(fs_hz)
    inside = (seconds >= local_interval_s[0]) & (seconds < local_interval_s[1])
    times_by_cluster = {cid: spike_times[inside & (spike_clusters == cid)] for cid in good_ids}
    depth_by_cluster = {cid: float(np.median(positions[inside & (spike_clusters == cid), 1])) for cid in good_ids if np.any(inside & (spike_clusters == cid))}
    contam_data = np.load(qc / "refractory" / "refractory_qc.npz", allow_pickle=False)
    contamination = {}
    qc_cids = np.unique(spike_clusters)
    for index, cid in enumerate(qc_cids):
        likelihoods = contam_data["rvl_tensor"][index]
        valid = np.min(likelihoods, axis=1)
        valid[valid < 0.05] = np.inf
        contamination[int(cid)] = float(contam_data["contamination_test_proportions"][int(np.argmin(valid))]) if np.isfinite(valid).any() else float("nan")
    return {"good_ids": good_ids, "times_by_cluster": times_by_cluster, "depth_by_cluster": depth_by_cluster, "waveforms": waveforms, "contamination": contamination, "samples_by_cluster": {cid: retained[inside & (spike_clusters == cid), 0].astype(np.int64) for cid in good_ids}, "amplitudes_by_cluster": {cid: retained[inside & (spike_clusters == cid), 2].astype(float) for cid in good_ids}, "fs_hz": fs_hz, "local_interval_s": local_interval_s, "physical_interval_s": interval_s}


def evaluate_endpoint(baseline: dict[str, Any], candidate: dict[str, Any], amendment: dict[str, Any], interval_s: tuple[float, float]) -> dict[str, Any]:
    """Evaluate the frozen paired endpoint without population substitution."""
    pairs, matching = _reciprocal_matches(baseline, candidate, amendment)
    completeness_config = CompletenessConfig(**{key: value for key, value in amendment["completeness"].items() if key in {"spikes_per_window", "max_isi_s", "window_indexing", "min_finite_interior_windows"}})
    rows = []
    for pair in pairs:
        bid, cid = pair["baseline_cluster"], pair["candidate_cluster"]
        baseline_interval = baseline.get("local_interval_s", interval_s)
        candidate_interval = candidate.get("local_interval_s", interval_s)
        bfit = train_completeness(baseline["samples_by_cluster"][bid], baseline["amplitudes_by_cluster"][bid], fs_hz=baseline["fs_hz"], interval_s=baseline_interval, config=completeness_config, label=f"baseline:{bid}")
        cfit = train_completeness(candidate["samples_by_cluster"][cid], candidate["amplitudes_by_cluster"][cid], fs_hz=candidate["fs_hz"], interval_s=candidate_interval, config=completeness_config, label=f"candidate:{cid}")
        rows.append({**pair, "baseline": bfit, "candidate": cfit, "measurable": bfit["status"] == "measured" and cfit["status"] == "measured", "contamination_increase": candidate["contamination"].get(cid, np.nan) - baseline["contamination"].get(bid, np.nan), "peak_retention": float(np.max(np.abs(candidate["waveforms"][cid])) / np.max(np.abs(baseline["waveforms"][bid])))})
    measurable = [row for row in rows if row["measurable"]]
    eligible = len(baseline["good_ids"])
    coverage = len(measurable) / eligible if eligible else 0.0
    changes = [row["baseline"]["missing_pct_median"] - row["candidate"]["missing_pct_median"] for row in measurable]
    contamination = [row["contamination_increase"] for row in measurable if np.isfinite(row["contamination_increase"])]
    cosines = [row["waveform_cosine"] for row in rows if np.isfinite(row["waveform_cosine"])]
    retention = [row["peak_retention"] for row in rows if np.isfinite(row["peak_retention"])]
    guardrail_failure = (contamination and max(contamination) > amendment["guardrails"]["maximum_contamination_increase"]) or (cosines and np.percentile(cosines, 10) < amendment["guardrails"]["minimum_waveform_cosine_p10"]) or (retention and np.percentile(retention, 10) < amendment["guardrails"]["minimum_waveform_peak_retention_p10"])
    status = "inconclusive" if coverage < amendment["completeness"]["minimum_coverage_fraction"] or not measurable else ("fail" if guardrail_failure or np.median(changes) < amendment["completeness"]["minimum_improvement_pp"] else "pass")
    return {"status": status, "eligible_units": eligible, "matched_units": len(pairs), "measurable_units": len(measurable), "coverage_fraction": coverage, "paired_completeness_improvement_pp": float(np.median(changes)) if changes else None, "contamination_increase_max": max(contamination) if contamination else None, "waveform_cosine_p10": float(np.percentile(cosines, 10)) if cosines else None, "waveform_peak_retention_p10": float(np.percentile(retention, 10)) if retention else None, "matching": matching, "pairs": rows, "prerequisite": "coverage below floor or no measurable paired fits" if status == "inconclusive" else None}


def _endpoint_report(root: Path, contract: dict[str, Any], amendment: dict[str, Any], arms: dict[str, Any], *, interval_s: tuple[float, float], fs_hz: float) -> dict[str, Any]:
    baseline = _arm_observations(root / "control", interval_s, fs_hz)
    candidate = _arm_observations(root / "corrected", interval_s, fs_hz)
    comparison = evaluate_endpoint(baseline, candidate, amendment, interval_s)
    report = {
        "schema": "luke-option-a-endpoint-report-v2",
        "primary": {"name": contract["endpoints"]["primary"]["name"], **comparison},
        "arms": {name: {"sort_identity_digest": value["identity"]["identity_digest"]} for name, value in arms.items()},
        "interpretation": "Negative closes this field/configuration/domain only; inconclusive identifies its measured denominator and prerequisite.",
    }
    (root / "endpoint_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_comparison(
    *,
    contract_path: Path,
    source_dir: Path,
    field_dir: Path,
    output_root: Path,
    amendment_path: Path | None = None,
    require_cuda: bool = True,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    if amendment_path is None:
        amendment_path = Path(__file__).resolve().parents[1] / "configs" / "option_a_development_comparison.v1.endpoint_amendment.v1.json"
    amendment = load_endpoint_amendment(amendment_path)
    if amendment["base_contract_digest"] != contract["contract_digest"]:
        raise ValueError("endpoint amendment is bound to a different frozen contract")
    output_root = Path(output_root)
    if str(output_root).startswith("/mnt/"):
        raise ValueError("refusing to write Option A outputs under /mnt")
    if not Path(source_dir).is_dir():
        raise FileNotFoundError(f"frozen source recording is unavailable: {source_dir}")
    field = _field_from_contract(contract, Path(field_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    run_identity = {
        "schema": SCHEMA,
        "contract_id": contract["contract_id"],
        "contract_digest": contract["contract_digest"],
        "endpoint_amendment": amendment["schema"],
        "field_digest": field["field_digest"],
        "source_recording": str(Path(source_dir).resolve()),
        "output_root": str(output_root.resolve()),
    }
    try:
        environment = validate_production_environment(require_cuda=require_cuda)
        prepared = _prepare_recordings(Path(source_dir), output_root, contract, amendment, field)
        arms = {
            "control": _run_arm(output_root / "recordings/control", output_root / "control"),
            "corrected": _run_arm(output_root / "recordings/corrected", output_root / "corrected"),
        }
        endpoint = _endpoint_report(output_root, contract, amendment, arms, interval_s=tuple(contract["domain"]["interval_s"]), fs_hz=float(contract["recording"]["sampling_frequency_hz"]))
        result = {**run_identity, "status": "completed", "environment": environment, "preparation": prepared, "arms": arms, "endpoint_report": endpoint}
    except Exception as error:
        result = {
            **run_identity,
            "status": "failed",
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        (output_root / "run_manifest.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
        raise
    (output_root / "run_manifest.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-recording", type=Path, required=True)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--no-cuda", action="store_true", help="only for a CPU integration fixture")
    args = parser.parse_args()
    result = run_comparison(
        contract_path=args.contract,
        source_dir=args.source_recording,
        field_dir=args.field_dir,
        output_root=args.output_root,
        amendment_path=args.amendment,
        require_cuda=not args.no_cuda,
    )
    print(json.dumps({"status": result["status"], "output_root": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()