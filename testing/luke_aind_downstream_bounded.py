"""Run the frozen bounded rescue-versus-AIND downstream comparison.

Preparation and sorting deliberately use separate pinned environments:

* pinned AIND preprocessing: Python 3.12.4 / SpikeInterface 0.104.7;
* frozen Kilosort sorting: the existing ``spikeinterface`` Conda environment.

The artifact sidecar is never applied to voltage.  Every recording and sort is
written through a partial directory and accepted only after manifest checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONFIG = REPO_ROOT / "testing/configs/luke_aind_downstream_bounded_v1.json"
AIND_PYTHON = REPO_ROOT / ".venv-aind/bin/python"
SORT_PYTHON = Path("/home/huklab/anaconda3/envs/spikeinterface/bin/python")
RESCUE_RECORDINGS = {
    probe: Path(
        "/mnt/NPX/Luke/20250804/"
        f"rescue_pipeline_results_Luke0804_V2V1_g0_{probe}/recording"
    )
    for probe in ("imec0", "imec1")
}


def sha256_file(path: Path, block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    conditions = {item["name"]: item for item in config["conditions"]}
    expected = {
        "rescue_ks_car_on": ("locked_rescue", True),
        "pinned_aind_ks_car_on": ("pinned_aind", True),
        "pinned_aind_ks_car_off": ("pinned_aind", False),
    }
    observed = {
        name: (item["preprocessing"], item["kilosort_do_CAR"])
        for name, item in conditions.items()
    }
    if observed != expected:
        raise ValueError(f"The three frozen conditions changed: {observed}")
    aind = config["aind_provenance"]
    if aind["blanking"] is not None or aind["motion_apply"]:
        raise ValueError("Pinned AIND must have no blanking and no motion application")
    if aind["frozen_removed_channels"] != {
        "imec0": ["imec0.ap#AP191"],
        "imec1": ["imec1.ap#AP191"],
    }:
        raise ValueError("Frozen AIND AP191 removal changed")
    covered = {
        value
        for window in config["windows"]
        for value in window["sealed_windows_covered"]
    }
    if covered != {
        "T1_high_motion",
        "T2_relative_quiet",
        "T2_high_motion",
        "T3_high_motion",
        "T3_relative_quiet",
    }:
        raise ValueError(f"Bounded sealed-window panel changed: {sorted(covered)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare", choices=("locked_rescue", "pinned_aind"))
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--probe", choices=("imec0", "imec1"))
    parser.add_argument("--window")
    parser.add_argument("--condition", choices=(
        "rescue_ks_car_on",
        "pinned_aind_ks_car_on",
        "pinned_aind_ks_car_off",
    ))
    return parser.parse_args()


def selected_windows(config: dict[str, Any], name: str | None) -> list[dict[str, Any]]:
    windows = config["windows"]
    if name is None:
        return windows
    selected = [window for window in windows if window["name"] == name]
    if len(selected) != 1:
        raise ValueError(f"Unknown window: {name}")
    return selected


def selected_probes(name: str | None) -> list[str]:
    return [name] if name else ["imec0", "imec1"]


def recording_dir(
    config: dict[str, Any], probe: str, window: str, preprocessing: str
) -> Path:
    return Path(config["output_root"]) / "recordings" / probe / window / preprocessing


def sort_dir(config: dict[str, Any], probe: str, window: str, condition: str) -> Path:
    return Path(config["output_root"]) / "sorts" / probe / window / condition


def plan_payload(config: dict[str, Any]) -> dict[str, Any]:
    jobs = []
    for probe in ("imec0", "imec1"):
        for window in config["windows"]:
            for condition in config["conditions"]:
                jobs.append(
                    {
                        "probe": probe,
                        "window": window["name"],
                        "condition": condition["name"],
                        "start_s": window["start_s"],
                        "duration_s": window["duration_s"],
                        "recording": str(
                            recording_dir(
                                config,
                                probe,
                                window["name"],
                                condition["preprocessing"],
                            )
                        ),
                        "sort": str(
                            sort_dir(config, probe, window["name"], condition["name"])
                        ),
                        "kilosort_do_CAR": condition["kilosort_do_CAR"],
                    }
                )
    return {
        "experiment": config["experiment"],
        "config_digest": canonical_digest(config),
        "n_prepared_recordings": 12,
        "n_sorts": len(jobs),
        "sealed_events_per_probe": 360,
        "motion_apply": False,
        "artifact_sidecar": "annotation_only",
        "jobs": jobs,
    }


def _resolve_physical_channel(recording, physical: int) -> Any:
    matches = [
        value
        for value in recording.get_channel_ids()
        if str(value).endswith(f"AP{physical}") or str(value) == str(physical)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Physical AP{physical} resolved to {matches}")
    return matches[0]


def _write_binary_and_manifest(
    traces: np.ndarray,
    recording,
    target: Path,
    manifest: dict[str, Any],
) -> None:
    partial = target.with_name(target.name + ".partial")
    if target.exists():
        existing = json.loads((target / "recording_manifest.json").read_text())
        if existing.get("request_digest") != manifest["request_digest"]:
            raise RuntimeError(f"Recording cache belongs to another request: {target}")
        print(f"Reusing {target}", flush=True)
        return
    if partial.exists():
        raise RuntimeError(f"Partial recording requires inspection: {partial}")
    partial.mkdir(parents=True)
    binary = partial / "traces.raw"
    values = np.ascontiguousarray(traces)
    values.tofile(binary)
    values_are_uv = manifest.get("units") == "uV"
    gain_property = recording.get_property("gain_to_uV")
    offset_property = recording.get_property("offset_to_uV")
    spec = {
        "file": "traces.raw",
        "dtype": str(values.dtype),
        "shape": [int(value) for value in values.shape],
        "sampling_frequency_hz": float(recording.get_sampling_frequency()),
        "channel_ids": [str(value) for value in recording.get_channel_ids()],
        "channel_locations_um": np.asarray(
            recording.get_channel_locations(), dtype=float
        ).tolist(),
        "gain_to_uV": (
            np.ones(values.shape[1], dtype=float).tolist()
            if values_are_uv
            else None
            if gain_property is None
            else np.asarray(gain_property, dtype=float).tolist()
        ),
        "offset_to_uV": (
            np.zeros(values.shape[1], dtype=float).tolist()
            if values_are_uv
            else None
            if offset_property is None
            else np.asarray(offset_property, dtype=float).tolist()
        ),
    }
    manifest.update(
        binary_bytes=int(binary.stat().st_size),
        binary_sha256=sha256_file(binary),
        recording_spec=spec,
        complete=True,
    )
    (partial / "recording_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    partial.rename(target)
    print(f"Completed {target}", flush=True)


def prepare_one(
    config: dict[str, Any], probe: str, window: dict[str, Any], preprocessing: str
) -> None:
    import spikeinterface
    import spikeinterface.full as si

    target = recording_dir(config, probe, window["name"], preprocessing)
    request = {
        "experiment": config["experiment"],
        "config_digest": canonical_digest(config),
        "probe": probe,
        "window": window,
        "preprocessing": preprocessing,
    }
    manifest = {**request, "request_digest": canonical_digest(request)}
    if target.exists():
        _write_binary_and_manifest(np.empty((0, 0)), None, target, manifest)
        return
    if preprocessing == "locked_rescue":
        if spikeinterface.__version__ != "0.102.1":
            raise RuntimeError("Locked rescue preparation requires SpikeInterface 0.102.1")
        recording = si.load_extractor(RESCUE_RECORDINGS[probe])
        fs = float(recording.get_sampling_frequency())
        start = int(round(window["start_s"] * fs))
        stop = start + int(round(window["duration_s"] * fs))
        sliced = recording.frame_slice(start_frame=start, end_frame=stop)
        traces = sliced.get_traces()
        manifest.update(
            graph=[
                "reuse accepted full-session rescue binary",
                "phase correction -> bilateral 500uV blanking -> AP191 interpolation",
            ],
            source_recording=str(RESCUE_RECORDINGS[probe]),
            units="ADC counts",
            spikeinterface_version=spikeinterface.__version__,
        )
        _write_binary_and_manifest(traces, sliced, target, manifest)
        return
    if spikeinterface.__version__ != config["aind_provenance"]["spikeinterface_version"]:
        raise RuntimeError("Pinned AIND preparation requires SpikeInterface 0.104.7")
    raw = si.read_spikeglx(
        folder_path=config["source"]["data_dir"],
        stream_id=f"{probe}.ap",
    )
    bad_id = _resolve_physical_channel(raw, 191)
    graph = {
        "phase_shift": {"margin_ms": 100.0},
        "highpass_filter": {"freq_min": 300.0},
        "detect_and_remove_bad_channels": {
            "method": "coherence+psd",
            "dead_channel_threshold": -0.5,
            "noisy_channel_threshold": 1.0,
            "outside_channel_threshold": -0.3,
            "outside_channels_location": "top",
            "n_neighbors": 11,
            "channel_filters": {"noise", "dead", "out"},
            "seed": 0,
            "bad_channel_ids": [bad_id],
        },
        "common_reference": {"reference": "global", "operator": "median"},
    }
    processed = si.apply_preprocessing_pipeline(raw, graph)
    fs = float(raw.get_sampling_frequency())
    start = int(round(window["start_s"] * fs))
    stop = int(round((window["start_s"] + window["duration_s"]) * fs))
    traces = processed.get_traces(
        start_frame=start,
        end_frame=stop,
        return_in_uV=True,
    )
    traces = np.asarray(traces, dtype=np.float32)
    manifest.update(
        graph=config["aind_provenance"]["graph"],
        resolved_bad_channel_ids=[str(bad_id)],
        branch_config_digest=config["aind_provenance"]["branch_config_digest"],
        bad_channel_manifest_digest=config["aind_provenance"][
            "bad_channel_manifest_digest"
        ],
        units="uV",
        spikeinterface_version=spikeinterface.__version__,
    )
    _write_binary_and_manifest(traces, processed, target, manifest)


def prepare_selected(
    config: dict[str, Any], preprocessing: str, probe: str | None, window: str | None
) -> None:
    for selected_probe in selected_probes(probe):
        for selected_window in selected_windows(config, window):
            prepare_one(config, selected_probe, selected_window, preprocessing)


def _load_prepared_recording(path: Path):
    import spikeinterface.core as sc

    manifest = json.loads((path / "recording_manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError(f"Incomplete recording: {path}")
    spec = manifest["recording_spec"]
    binary = path / spec["file"]
    expected = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
    if binary.stat().st_size != expected:
        raise RuntimeError(f"Prepared binary size mismatch: {binary}")
    recording = sc.BinaryRecordingExtractor(
        file_paths=[binary],
        sampling_frequency=spec["sampling_frequency_hz"],
        num_channels=spec["shape"][1],
        dtype=spec["dtype"],
        channel_ids=np.asarray(spec["channel_ids"]),
    )
    recording.set_channel_locations(np.asarray(spec["channel_locations_um"], dtype=float))
    if spec["gain_to_uV"] is not None:
        recording.set_property("gain_to_uV", np.asarray(spec["gain_to_uV"], dtype=float))
    if spec["offset_to_uV"] is not None:
        recording.set_property("offset_to_uV", np.asarray(spec["offset_to_uV"], dtype=float))
    return recording, manifest


def sort_one(
    config: dict[str, Any], probe: str, window: dict[str, Any], condition: dict[str, Any]
) -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-aind-downstream-numba-cache")
    from spikeinterface.sorters import run_sorter
    from pipeline.sorting import build_kilosort4_params

    target = sort_dir(config, probe, window["name"], condition["name"])
    if (target / "sorter_output/spike_times.npy").exists():
        print(f"Reusing {target}", flush=True)
        return
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise RuntimeError(f"Partial or ambiguous sort requires inspection: {target}")
    prepared = recording_dir(
        config, probe, window["name"], condition["preprocessing"]
    )
    recording, recording_manifest = _load_prepared_recording(prepared)
    params = build_kilosort4_params()
    params.update(
        do_CAR=bool(condition["kilosort_do_CAR"]),
        do_correction=False,
        skip_kilosort_preprocessing=False,
        highpass_cutoff=300,
    )
    partial.parent.mkdir(parents=True, exist_ok=True)
    run_sorter(
        "kilosort4",
        recording,
        folder=partial,
        verbose=True,
        remove_existing_folder=False,
        **params,
    )
    sorter = partial / "sorter_output"
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    applied = dict(ops.get("settings", {}))
    applied.update(ops)
    if bool(applied["do_CAR"]) != bool(condition["kilosort_do_CAR"]):
        raise RuntimeError("Saved Kilosort do_CAR differs from the frozen condition")
    if int(applied["nblocks"]) != 0:
        raise RuntimeError("Kilosort motion correction was not disabled")
    if int(applied["highpass_cutoff"]) != 300:
        raise RuntimeError("Kilosort high-pass cutoff changed")
    receipt = {
        "complete": True,
        "experiment": config["experiment"],
        "config_digest": canonical_digest(config),
        "probe": probe,
        "window": window,
        "condition": condition,
        "recording_request_digest": recording_manifest["request_digest"],
        "sorter_params": {
            key: ("Infinity" if isinstance(value, float) and math.isinf(value) else value)
            for key, value in params.items()
        },
        "saved_critical_settings": {
            "do_CAR": bool(applied["do_CAR"]),
            "nblocks": int(applied["nblocks"]),
            "highpass_cutoff": int(applied["highpass_cutoff"]),
        },
    }
    (partial / "bounded_sort_manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    partial.rename(target)
    print(f"Completed {target}", flush=True)


def sort_selected(
    config: dict[str, Any], probe: str | None, window: str | None, condition: str | None
) -> None:
    conditions = [
        item for item in config["conditions"] if condition is None or item["name"] == condition
    ]
    for selected_probe in selected_probes(probe):
        for selected_window in selected_windows(config, window):
            for selected_condition in conditions:
                sort_one(config, selected_probe, selected_window, selected_condition)


def run_all(config_path: Path, config: dict[str, Any]) -> None:
    script = Path(__file__).resolve()
    commands = (
        [str(SORT_PYTHON), str(script), "--config", str(config_path), "--prepare", "locked_rescue"],
        [str(AIND_PYTHON), str(script), "--config", str(config_path), "--prepare", "pinned_aind"],
        [str(SORT_PYTHON), str(script), "--config", str(config_path), "--sort"],
        [str(SORT_PYTHON), str(script), "--config", str(config_path), "--score"],
    )
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _refractory_median(
    times: np.ndarray, clusters: np.ndarray, good_units: set[int], fs: float
) -> float:
    values = []
    limit = int(round(1.5e-3 * fs))
    for unit in good_units:
        unit_times = np.sort(times[clusters == unit])
        if len(unit_times) > 1:
            values.append(float(np.mean(np.diff(unit_times) < limit)))
    return float(np.median(values)) if values else math.nan


def valid_sample_mask(times: np.ndarray, duration_frames: int) -> np.ndarray:
    """Select spikes that fall inside the half-open recording interval."""
    values = np.asarray(times)
    return (values >= 0) & (values < duration_frames)


def presence_fraction(
    times: np.ndarray, duration_frames: int, n_bins: int
) -> float:
    """Return occupancy across equal-duration bins without a rounding sliver."""
    if duration_frames <= 0 or n_bins <= 0:
        raise ValueError("duration_frames and n_bins must be positive")
    values = np.asarray(times, dtype=np.int64)
    if len(values) == 0:
        return 0.0
    bins = np.minimum((values * n_bins) // duration_frames, n_bins - 1)
    return float(len(np.unique(bins)) / n_bins)


def _score_one(
    config: dict[str, Any], probe: str, window: dict[str, Any], condition: dict[str, Any]
) -> dict[str, Any]:
    import pandas as pd

    from testing.luke_full_probe_rescue_diagnostics import (
        near_coincident_fraction_sorted,
        shifted_coincidence_null,
    )
    from testing.luke_trace_reviewed_events import local_match_details

    target = sort_dir(config, probe, window["name"], condition["name"])
    sorter = target / "sorter_output"
    if not (sorter / "spike_times.npy").exists():
        raise FileNotFoundError(f"Missing completed sort: {target}")
    prepared = recording_dir(
        config, probe, window["name"], condition["preprocessing"]
    )
    recording_manifest = json.loads((prepared / "recording_manifest.json").read_text())
    spec = recording_manifest["recording_spec"]
    fs = float(spec["sampling_frequency_hz"])
    duration_frames = int(spec["shape"][0])
    duration_s = duration_frames / fs
    times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1).astype(np.int64)
    clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1).astype(np.int32)
    depths = np.load(sorter / "spike_positions.npy", mmap_mode="r")[:, 1].astype(np.float32)
    final_spike_count_raw = int(len(times))
    valid_final = valid_sample_mask(times, duration_frames)
    times = times[valid_final]
    clusters = clusters[valid_final]
    depths = depths[valid_final]
    templates = np.load(sorter / "templates.npy", mmap_mode="r")
    similarity = np.load(sorter / "similar_templates.npy").astype(float)
    channel_positions = np.load(sorter / "channel_positions.npy").astype(float)
    labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t")
    label_column = next(column for column in labels if column != "cluster_id")
    good_units = set(
        labels.loc[
            labels[label_column].astype(str).str.lower().eq("good"), "cluster_id"
        ].astype(int)
    )
    contamination = pd.read_csv(sorter / "cluster_ContamPct.tsv", sep="\t")
    contam_column = next(column for column in contamination if column != "cluster_id")
    good_contamination = contamination.loc[
        contamination.cluster_id.astype(int).isin(good_units), contam_column
    ].to_numpy(float)

    n_bins = max(1, int(round(duration_s / 30.0)))
    presence = []
    for unit in good_units:
        presence.append(
            presence_fraction(times[clusters == unit], duration_frames, n_bins)
        )

    tolerance = int(round(0.5e-3 * fs))
    order = np.argsort(times, kind="stable")
    coincidence = near_coincident_fraction_sorted(
        times[order], clusters[order], depths[order], tolerance, 75.0
    )
    coincidence_null = shifted_coincidence_null(
        times[order], clusters[order], depths[order], duration_frames, tolerance, 8042026
    )

    holdout_key = pd.read_csv(
        REPO_ROOT / "testing/outputs/luke_prospective_holdout/holdout_candidate_key_v2.csv"
    )
    events = holdout_key[
        holdout_key.probe.eq(probe)
        & holdout_key.time_s.ge(window["start_s"])
        & holdout_key.time_s.lt(window["start_s"] + window["duration_s"])
    ].copy()
    global_start = int(round(window["start_s"] * fs))
    event_samples = events.sample_index.to_numpy(np.int64) - global_start
    matches = local_match_details(
        event_samples,
        events.depth_um.to_numpy(float),
        times,
        depths,
        tolerance,
        100.0,
    ).present
    events["recovered"] = matches

    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    template_depths = channel_positions[peak_channels, 1]
    upper = np.triu(np.ones(similarity.shape, dtype=bool), 1)
    nearby = np.abs(template_depths[:, None] - template_depths[None, :]) <= 100.0
    first, second = np.where(upper & nearby & (similarity >= 0.8))
    pair_rows = [
        {
            "first_unit": int(a),
            "second_unit": int(b),
            "both_good": bool(a in good_units and b in good_units),
            "template_similarity": float(similarity[a, b]),
            "depth_difference_um": float(abs(template_depths[a] - template_depths[b])),
        }
        for a, b in zip(first, second)
    ]
    pairs = pd.DataFrame(pair_rows)
    good_pairs = pairs[pairs.both_good].copy() if len(pairs) else pairs
    if len(good_pairs):
        good_pairs = good_pairs.sort_values("template_similarity", ascending=False)
        good_pairs["strong_duplicate_hypothesis"] = False
        good_pairs.loc[good_pairs.index[:3], "strong_duplicate_hypothesis"] = True

    full_st = np.load(sorter / "full_st.npy", mmap_mode="r")
    full_times = np.asarray(full_st[:, 0] if full_st.ndim > 1 else full_st).reshape(-1)
    full_count_raw = int(len(full_times))
    full_count = int(valid_sample_mask(full_times, duration_frames).sum())
    summary = {
        "probe": probe,
        "window": window["name"],
        "condition": condition["name"],
        "duration_s": duration_s,
        "sealed_event_count": int(len(events)),
        "sealed_event_recovery": float(np.mean(matches)) if len(matches) else math.nan,
        "learned_detection_count": full_count,
        "learned_detection_count_raw": full_count_raw,
        "learned_excess_detection_count_removed": full_count_raw - full_count,
        "final_spike_count": int(len(times)),
        "final_spike_count_raw": final_spike_count_raw,
        "final_excess_spike_count_removed": final_spike_count_raw - len(times),
        "learned_to_final_detection_ratio": full_count / max(len(times), 1),
        "final_spikes_per_s": len(times) / duration_s,
        "unit_count": int(templates.shape[0]),
        "kilosort_good_count": int(len(good_units)),
        "median_good_contamination_pct": (
            float(np.nanmedian(good_contamination)) if len(good_contamination) else math.nan
        ),
        "median_good_refractory_fraction_1p5ms": _refractory_median(
            times, clusters, good_units, fs
        ),
        "median_good_presence_fraction_30s": (
            float(np.median(presence)) if presence else math.nan
        ),
        "good_units_present_ge_90pct": int(np.sum(np.asarray(presence) >= 0.9)),
        "coincidence_fraction": coincidence,
        "coincidence_shift_null": coincidence_null,
        "coincidence_excess": coincidence - coincidence_null,
        "nearby_similar_good_good_pairs": int(len(good_pairs)),
        "residual_pairs_selected": int(min(len(good_pairs), 3)),
    }
    output = Path(config["output_root"]) / "analysis" / probe / window["name"] / condition["name"]
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "sealed_event_recovery.csv", index=False)
    good_pairs.to_csv(output / "similar_good_pairs.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if len(good_pairs):
        from testing.luke_full_strip_pair_residual_audit import run_audit

        run_audit(
            sorter,
            output / "similar_good_pairs.csv",
            output / "residual",
            events_per_pair=16,
        )
        residual = pd.read_csv(output / "residual/pair_residual_summary.csv")
        summary["residual_pairs_supporting_redundancy"] = int(
            residual.residual_supports_redundant_templates.sum()
        )
        summary["median_two_template_relative_residual_improvement"] = float(
            residual.median_two_over_best_single_relative_improvement.median()
        )
    else:
        summary["residual_pairs_supporting_redundancy"] = 0
        summary["median_two_template_relative_residual_improvement"] = math.nan
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def score_all(config: dict[str, Any]) -> None:
    import pandas as pd

    rows = []
    for probe in ("imec0", "imec1"):
        for window in config["windows"]:
            for condition in config["conditions"]:
                rows.append(_score_one(config, probe, window, condition))
    frame = pd.DataFrame(rows)
    analysis = Path(config["output_root"]) / "analysis"
    frame.to_csv(analysis / "bounded_condition_scores.csv", index=False)
    metrics = [
        "sealed_event_recovery",
        "learned_detection_count",
        "final_spike_count",
        "learned_to_final_detection_ratio",
        "final_spikes_per_s",
        "median_good_refractory_fraction_1p5ms",
        "median_good_presence_fraction_30s",
        "coincidence_excess",
        "nearby_similar_good_good_pairs",
        "residual_pairs_supporting_redundancy",
        "median_two_template_relative_residual_improvement",
        "kilosort_good_count",
        "median_good_contamination_pct",
    ]
    paired = frame.pivot(index=["probe", "window"], columns="condition", values=metrics)
    contrast_rows = []
    for probe_window, row in paired.iterrows():
        for challenger in ("pinned_aind_ks_car_on", "pinned_aind_ks_car_off"):
            values = {"probe": probe_window[0], "window": probe_window[1], "challenger": challenger}
            for metric in metrics:
                values[f"delta_{metric}"] = float(
                    row[(metric, challenger)] - row[(metric, "rescue_ks_car_on")]
                )
            contrast_rows.append(values)
    pd.DataFrame(contrast_rows).to_csv(analysis / "paired_contrasts.csv", index=False)
    aggregate = frame.groupby("condition")[metrics].median(numeric_only=True)
    aggregate.to_csv(analysis / "condition_medians.csv")
    decision = {
        "status": "results_complete_decision_requires_endpoint_family_review",
        "config_digest": canonical_digest(config),
        "n_sorts": int(len(frame)),
        "primary_endpoint_families": config["primary_endpoint_families"],
        "secondary_diagnostics": config["secondary_diagnostics"],
        "artifact_sidecar_policy": config["artifact_sidecar_policy"],
    }
    (analysis / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.plan:
        print(json.dumps(plan_payload(config), indent=2))
    if args.prepare:
        prepare_selected(config, args.prepare, args.probe, args.window)
    if args.sort:
        sort_selected(config, args.probe, args.window, args.condition)
    if args.score:
        score_all(config)
    if args.run_all:
        run_all(args.config.resolve(), config)
    if not any((args.plan, args.prepare, args.sort, args.score, args.run_all)):
        raise SystemExit("Choose --plan, --prepare, --sort, --score, or --run-all")


if __name__ == "__main__":
    main()
