"""Cache-safe motion-scale sweep for Luke's pathological 120-second window.

The sweep estimates motion from the exact same localized peaks while varying
temporal resolution, spatial window scale, method, and peak split.  Half-split
runs measure estimator reproducibility rather than visual smoothness.  Every
run has a content-addressed manifest and preserves compact native diagnostics.

Examples
--------
Inspect the grid::

    python testing/luke_motion_scale_sweep.py --plan-only

Run and summarize on a CUDA-capable host::

    python testing/luke_motion_scale_sweep.py --run --summarize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_motion_scale_characterization import (
    LUKE_ROOT,
    PROBES,
    WINDOW,
    Window,
    best_lag_correlation,
    correlation,
    decompose_spatial_field,
    interpolate_field,
    summarize_field,
)

WINDOWS = {
    "registration_outlier": WINDOW,
    "shared_template": Window("shared_template", 7095.0, 240.0),
}


RAW_ROOT = LUKE_ROOT / "Luke0804_V2V1_g0"
DEFAULT_OUTPUT = (
    LUKE_ROOT
    / "dredge_pipeline_results_Luke0804_V2V1_g0_imec1"
    / "motion_scale_sweep"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    method: str
    rigid: bool
    win_step_um: float = 200.0
    win_scale_um: float = 300.0
    method_kwargs: dict[str, Any] = field(default_factory=dict)
    split_halves: bool = False


def _dredge_kwargs(bin_s: float, smooth_s: float, max_disp_um: float | None = 80.0) -> dict:
    result = {
        "bin_um": 1.0,
        "bin_s": bin_s,
        "histogram_time_smooth_s": smooth_s,
        "histogram_depth_smooth_um": 1.0,
        "time_horizon_s": 60.0,
        "mincorr": 0.1,
        "device": "cuda",
    }
    if max_disp_um is not None:
        result["max_disp_um"] = max_disp_um
    return result


CANDIDATES = (
    Candidate("dredge_rigid_t0p25", "temporal", "dredge_ap", True, method_kwargs=_dredge_kwargs(0.25, 0.25)),
    Candidate("dredge_rigid_t0p5", "temporal", "dredge_ap", True, method_kwargs=_dredge_kwargs(0.5, 0.5)),
    Candidate("dredge_rigid_t1", "temporal", "dredge_ap", True, method_kwargs=_dredge_kwargs(1.0, 1.0), split_halves=True),
    Candidate("dredge_rigid_t2", "temporal", "dredge_ap", True, method_kwargs=_dredge_kwargs(2.0, 1.0)),
    Candidate("dredge_rigid_t4", "temporal", "dredge_ap", True, method_kwargs=_dredge_kwargs(4.0, 2.0)),
    Candidate(
        "dredge_nr_current_exact",
        "spatial",
        "dredge_ap",
        False,
        win_step_um=100.0,
        win_scale_um=150.0,
        method_kwargs=_dredge_kwargs(1.0, 1.0, max_disp_um=None),
        split_halves=True,
    ),
    Candidate(
        "dredge_nr_current_max80",
        "spatial",
        "dredge_ap",
        False,
        win_step_um=100.0,
        win_scale_um=150.0,
        method_kwargs=_dredge_kwargs(1.0, 1.0, max_disp_um=80.0),
        split_halves=True,
    ),
    Candidate(
        "dredge_nr_100_300",
        "spatial",
        "dredge_ap",
        False,
        win_step_um=100.0,
        win_scale_um=300.0,
        method_kwargs=_dredge_kwargs(1.0, 1.0),
    ),
    Candidate(
        "dredge_nr_200_300",
        "spatial",
        "dredge_ap",
        False,
        win_step_um=200.0,
        win_scale_um=300.0,
        method_kwargs=_dredge_kwargs(1.0, 1.0),
    ),
    Candidate(
        "dredge_nr_200_300_split",
        "spatial_validation",
        "dredge_ap",
        False,
        win_step_um=200.0,
        win_scale_um=300.0,
        method_kwargs=_dredge_kwargs(1.0, 1.0),
        split_halves=True,
    ),
    Candidate(
        "dredge_nr_200_600",
        "spatial",
        "dredge_ap",
        False,
        win_step_um=200.0,
        win_scale_um=600.0,
        method_kwargs=_dredge_kwargs(1.0, 1.0),
        split_halves=True,
    ),
    Candidate(
        "dredge_nr_400_600",
        "spatial",
        "dredge_ap",
        False,
        win_step_um=400.0,
        win_scale_um=600.0,
        method_kwargs=_dredge_kwargs(1.0, 1.0),
    ),
    Candidate(
        "dredge_nr_400_400_preset_scale",
        "spatial_preset_control",
        "dredge_ap",
        False,
        win_step_um=400.0,
        win_scale_um=400.0,
        method_kwargs=_dredge_kwargs(1.0, 1.0),
    ),
    Candidate(
        "decentralized_rigid",
        "method",
        "decentralized",
        True,
        method_kwargs={
            "bin_s": 1.0,
            "histogram_time_smooth_s": 1.0,
            "max_displacement_um": 80.0,
            "time_horizon_s": 60.0,
            "conv_engine": "torch",
            "torch_device": "cuda",
            "batch_size": 32,
        },
    ),
    Candidate(
        "decentralized_nr_200_300",
        "method",
        "decentralized",
        False,
        win_step_um=200.0,
        win_scale_um=300.0,
        method_kwargs={
            "bin_s": 1.0,
            "histogram_time_smooth_s": 1.0,
            "max_displacement_um": 80.0,
            "time_horizon_s": 60.0,
            "conv_engine": "torch",
            "torch_device": "cuda",
            "batch_size": 32,
        },
        split_halves=True,
    ),
    Candidate(
        "iterative_rigid",
        "method",
        "iterative_template",
        True,
        method_kwargs={"bin_s": 1.0, "num_shifts_block": 5},
    ),
    Candidate(
        "iterative_nr_200_300",
        "method",
        "iterative_template",
        False,
        win_step_um=200.0,
        win_scale_um=300.0,
        method_kwargs={"bin_s": 1.0, "num_shifts_block": 5},
    ),
    Candidate(
        "ks_pipeline_default",
        "method",
        "iterative_template",
        False,
        win_step_um=200.0,
        win_scale_um=300.0,
        method_kwargs={"bin_s": 2.0, "num_shifts_block": 5},
    ),
    Candidate(
        "medicine_pipeline_default",
        "exploratory_native",
        "medicine",
        False,
        method_kwargs={
            "time_bin_size": 1.0,
            "num_depth_bins": 2,
            "time_kernel_width": 50,
            "amplitude_threshold_quantile": 0.2,
            "training_steps": 10000,
            "plot_figures": False,
        },
    ),
    Candidate(
        "medicine_nr_8bin_t20",
        "exploratory_native",
        "medicine",
        False,
        method_kwargs={
            "time_bin_size": 1.0,
            "num_depth_bins": 8,
            "time_kernel_width": 20,
            "amplitude_threshold_quantile": 0.2,
            "training_steps": 10000,
            "plot_figures": False,
        },
    ),
    Candidate(
        "dredge_lfp_rigid_100hz",
        "lfp",
        "dredge_lfp",
        True,
        method_kwargs={
            "chunk_len_s": 10.0,
            "max_disp_um": 80.0,
            "time_horizon_s": 60.0,
            "mincorr": 0.8,
            "device": "cuda",
        },
    ),
    Candidate(
        "dredge_lfp_rigid_max20_100hz",
        "lfp",
        "dredge_lfp",
        True,
        method_kwargs={
            "chunk_len_s": 10.0,
            "max_disp_um": 20.0,
            "time_horizon_s": 60.0,
            "mincorr": 0.8,
            "device": "cuda",
        },
    ),
    Candidate(
        "dredge_lfp_nr_200_300_100hz",
        "lfp",
        "dredge_lfp",
        False,
        win_step_um=200.0,
        win_scale_um=300.0,
        method_kwargs={
            "chunk_len_s": 10.0,
            "max_disp_um": 80.0,
            "time_horizon_s": 60.0,
            "mincorr": 0.8,
            "device": "cuda",
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES))
    parser.add_argument("--candidates", nargs="+", default=[candidate.name for candidate in CANDIDATES])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--seed", type=int, default=20250804)
    parser.add_argument(
        "--window-name",
        choices=tuple(WINDOWS),
        default=WINDOW.name,
        help="Prespecified raw-frame window to estimate.",
    )
    return parser.parse_args()


def candidate_by_name(name: str) -> Candidate:
    matches = [candidate for candidate in CANDIDATES if candidate.name == name]
    if not matches:
        raise KeyError(f"Unknown candidate: {name}")
    return matches[0]


def split_names(candidate: Candidate) -> tuple[str, ...]:
    return ("full", "half_a", "half_b") if candidate.split_halves else ("full",)


def stable_peak_split(n_peaks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n_peaks, dtype=np.int8)


def run_spec(probe: str, candidate: Candidate, split: str, seed: int) -> dict:
    spec = {
        "probe": probe,
        "window": asdict(WINDOW),
        "candidate": asdict(candidate),
        "split": split,
        "seed": seed,
        "stream_id": f"{probe}.ap",
        "source_pipeline": str(pipeline_root(probe)),
    }
    if candidate.method == "dredge_lfp":
        spec["source_stream"] = f"{probe}.lf"
        spec["lfp_preprocessing"] = {
            "margin_s": 2.0,
            "bandpass_hz": [0.5, 100.0],
            "resample_hz": 100.0,
            "duplicate_depth_reduction": "mean",
            "common_reference": "global_median",
        }
    return spec


def spec_hash(spec: dict) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_dir(output_dir: Path, probe: str, candidate: Candidate, split: str, seed: int) -> Path:
    spec = run_spec(probe, candidate, split, seed)
    return output_dir / "runs" / probe / candidate.name / f"{split}_{spec_hash(spec)}"


def pipeline_root(probe: str) -> Path:
    return LUKE_ROOT / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}"


def _window_peaks(probe: str, fs: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    motion_root = pipeline_root(probe) / "motion"
    peaks = np.load(motion_root / "peaks.npy", mmap_mode="r")
    locations = np.load(motion_root / "peak_locations.npy", mmap_mode="r")
    start = int(round(WINDOW.start_s * fs))
    stop = int(round((WINDOW.start_s + WINDOW.duration_s) * fs))
    left = int(np.searchsorted(peaks["sample_index"], start, side="left"))
    right = int(np.searchsorted(peaks["sample_index"], stop, side="left"))
    selected_peaks = np.array(peaks[left:right], copy=True)
    selected_locations = np.array(locations[left:right], copy=True)
    selected_peaks["sample_index"] -= start
    return selected_peaks, selected_locations, start, stop


def select_split(
    peaks: np.ndarray, locations: np.ndarray, split: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if split == "full":
        return peaks, locations
    assignments = stable_peak_split(len(peaks), seed)
    selected = assignments == (0 if split == "half_a" else 1)
    return peaks[selected], locations[selected]


def peak_digest(peaks: np.ndarray, locations: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (peaks, locations):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _flatten_extra(prefix: str, value: Any, arrays: dict[str, np.ndarray], scalars: dict[str, Any]) -> None:
    key = prefix.strip("_")
    if isinstance(value, np.ndarray):
        arrays[key] = value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten_extra(f"{key}_{index}", item, arrays, scalars)
    elif isinstance(value, dict):
        for child, item in value.items():
            _flatten_extra(f"{key}_{child}", item, arrays, scalars)
    elif isinstance(value, (str, bool, int, float, np.integer, np.floating)) or value is None:
        scalars[key] = value.item() if isinstance(value, (np.integer, np.floating)) else value


def extra_quality(extra: dict, mincorr: float = 0.1) -> dict[str, float]:
    correlation_values = []
    weight_values = []
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    _flatten_extra("", extra, arrays, scalars)
    for key, value in arrays.items():
        leaf = key.split("_")[0]
        if leaf == "C" and value.ndim >= 2 and value.shape[-1] == value.shape[-2]:
            matrix = value.reshape((-1, value.shape[-2], value.shape[-1]))
            tri = np.triu_indices(value.shape[-1], 1)
            correlation_values.append(matrix[:, tri[0], tri[1]].ravel())
        if leaf == "U" and value.ndim >= 2:
            weight_values.append(value.ravel())
    correlations = np.concatenate(correlation_values) if correlation_values else np.array([])
    correlations = correlations[np.isfinite(correlations)]
    weights = np.concatenate(weight_values) if weight_values else np.array([])
    weights = weights[np.isfinite(weights)]
    return {
        "n_pair_correlations": int(len(correlations)),
        "median_pair_correlation": float(np.median(correlations)) if len(correlations) else np.nan,
        "pair_correlation_fraction_ge_mincorr": float(np.mean(correlations >= mincorr)) if len(correlations) else np.nan,
        "positive_weight_fraction": float(np.mean(weights > 0)) if len(weights) else np.nan,
    }


def save_extra(extra: dict, target: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    _flatten_extra("", extra, arrays, scalars)
    if arrays:
        np.savez_compressed(target / "extra_arrays.npz", **arrays)
    (target / "extra_scalars.json").write_text(json.dumps(scalars, indent=2) + "\n")


def assert_cuda_for(candidates: list[Candidate]) -> None:
    if not any(candidate.method in ("dredge_ap", "dredge_lfp", "decentralized", "medicine") for candidate in candidates):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the estimator sweep outside the sandbox")


def prepare_lfp_recording(raw_root: Path, probe: str):
    """Preprocess the exact LFP window for DREDGE without duplicate depths."""
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre
    from spikeinterface.core import NumpyRecording

    raw = se.read_spikeglx(folder_path=raw_root, load_sync_channel=False, stream_id=f"{probe}.lf")
    fs = float(raw.get_sampling_frequency())
    margin_s = 2.0
    start = int(round((WINDOW.start_s - margin_s) * fs))
    stop = int(round((WINDOW.start_s + WINDOW.duration_s + margin_s) * fs))
    extended = raw.frame_slice(start_frame=start, end_frame=stop)
    filtered = spre.bandpass_filter(extended, freq_min=0.5, freq_max=100.0, dtype="float32")
    resampled = spre.resample(filtered, 100, dtype="float32")
    margin_frames = int(round(margin_s * float(resampled.get_sampling_frequency())))
    windowed = resampled.frame_slice(
        start_frame=margin_frames,
        end_frame=margin_frames + int(round(WINDOW.duration_s * float(resampled.get_sampling_frequency()))),
    )
    traces = np.asarray(windowed.get_traces(), dtype=np.float32)
    locations = windowed.get_channel_locations()
    unique_depths, inverse = np.unique(locations[:, 1], return_inverse=True)
    depth_traces = np.empty((traces.shape[0], len(unique_depths)), dtype=np.float32)
    depth_locations = np.empty((len(unique_depths), 2), dtype=float)
    for index, depth in enumerate(unique_depths):
        channels = inverse == index
        depth_traces[:, index] = np.mean(traces[:, channels], axis=1)
        depth_locations[index] = [float(np.mean(locations[channels, 0])), float(depth)]
    depth_traces -= np.median(depth_traces, axis=1, keepdims=True)
    t_start = float(raw.get_time_info()["t_start"] + WINDOW.start_s)
    result = NumpyRecording(depth_traces, 100.0, t_starts=[t_start])
    result.set_channel_locations(depth_locations)
    return result, {
        "sampling_frequency_hz": 100.0,
        "start_sample": int(round(WINDOW.start_s * fs)),
        "stop_sample": int(round((WINDOW.start_s + WINDOW.duration_s) * fs)),
        "n_input_channels": int(raw.get_num_channels()),
        "n_unique_depths": int(len(unique_depths)),
    }


def run_grid(output_dir: Path, probes: list[str], candidates: list[Candidate], seed: int) -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-motion-scale-numba-cache")
    import spikeinterface.extractors as se
    from spikeinterface.sortingcomponents.motion import estimate_motion

    assert_cuda_for(candidates)
    for probe in probes:
        raw = se.read_spikeglx(
            folder_path=RAW_ROOT, load_sync_channel=False, stream_id=f"{probe}.ap"
        )
        fs = float(raw.get_sampling_frequency())
        peaks, locations, start, stop = _window_peaks(probe, fs)
        recording = raw.frame_slice(start_frame=start, end_frame=stop)
        lfp_recording = None
        lfp_info = None
        for candidate in candidates:
            for split in split_names(candidate):
                selected_peaks, selected_locations = select_split(peaks, locations, split, seed)
                target = run_dir(output_dir, probe, candidate, split, seed)
                result_path = target / "motion.npy"
                spec = run_spec(probe, candidate, split, seed)
                if candidate.method == "dredge_lfp":
                    if lfp_recording is None:
                        lfp_recording, lfp_info = prepare_lfp_recording(RAW_ROOT, probe)
                    spec.update(**lfp_info, n_peaks=0, peak_digest=None)
                else:
                    spec.update(
                        sampling_frequency_hz=fs,
                        start_sample=start,
                        stop_sample=stop,
                        n_peaks=len(selected_peaks),
                        peak_digest=peak_digest(selected_peaks, selected_locations),
                    )
                if result_path.exists():
                    saved = json.loads((target / "manifest.json").read_text())
                    if saved != spec:
                        raise RuntimeError(f"Cache manifest mismatch: {target}")
                    print(f"Reusing {target}")
                    continue
                if target.exists():
                    raise RuntimeError(f"Ambiguous partial run directory: {target}")
                target.mkdir(parents=True)
                (target / "manifest.json").write_text(json.dumps(spec, indent=2) + "\n")
                print(f"Running {probe} / {candidate.name} / {split} ({len(selected_peaks)} peaks)", flush=True)
                if candidate.method == "medicine":
                    import medicine
                    import torch

                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    peak_times = (
                        selected_peaks["sample_index"] / fs
                        + float(recording.get_time_info()["t_start"])
                    )
                    medicine.run_medicine(
                        peak_times=peak_times,
                        peak_depths=selected_locations["y"],
                        peak_amplitudes=selected_peaks["amplitude"],
                        output_dir=target,
                        **candidate.method_kwargs,
                    )
                    extra = {}
                    quality = extra_quality(extra)
                    (target / "native_quality.json").write_text(json.dumps(quality, indent=2) + "\n")
                    continue
                if candidate.method == "dredge_lfp":
                    motion = estimate_motion(
                        recording=lfp_recording,
                        peaks=None,
                        peak_locations=None,
                        direction="y",
                        rigid=candidate.rigid,
                        win_shape="gaussian",
                        win_step_um=candidate.win_step_um,
                        win_scale_um=candidate.win_scale_um,
                        method=candidate.method,
                        extra_outputs=False,
                        progress_bar=True,
                        verbose=False,
                        **candidate.method_kwargs,
                    )
                    extra = {}
                else:
                    motion, extra = estimate_motion(
                        recording=recording,
                        peaks=selected_peaks,
                        peak_locations=selected_locations,
                        direction="y",
                        rigid=candidate.rigid,
                        win_shape="gaussian",
                        win_step_um=candidate.win_step_um,
                        win_scale_um=candidate.win_scale_um,
                        method=candidate.method,
                        extra_outputs=True,
                        progress_bar=True,
                        verbose=False,
                        **candidate.method_kwargs,
                    )
                np.save(target / "motion.npy", motion.displacement[0])
                np.save(target / "time_bins.npy", motion.temporal_bins_s[0])
                np.save(target / "depth_bins.npy", motion.spatial_bins_um)
                save_extra(extra, target)
                quality = extra_quality(extra, float(candidate.method_kwargs.get("mincorr", 0.1)))
                (target / "native_quality.json").write_text(json.dumps(quality, indent=2) + "\n")


def _load_run_field(target: Path, target_times: np.ndarray, target_depths: np.ndarray) -> np.ndarray:
    displacement = np.load(target / "motion.npy")
    times = np.load(target / "time_bins.npy")
    depths = np.load(target / "depth_bins.npy")
    # SI stores a rigid estimate at one representative spatial bin.  A rigid
    # displacement is depth-invariant, so expand it explicitly before using
    # the shared non-rigid interpolation path.
    if len(depths) == 1:
        time_sampled = np.interp(target_times, times, displacement[:, 0])
        return np.repeat(time_sampled[:, None], len(target_depths), axis=1)
    return interpolate_field(
        displacement,
        times,
        depths,
        target_times,
        target_depths,
    )


def summarize(output_dir: Path, probes: list[str], candidates: list[Candidate], seed: int) -> None:
    common_dt = 2.0
    relative_times = np.arange(WINDOW.start_s + 1.0, WINDOW.start_s + WINDOW.duration_s, common_dt)
    # Intersection of the native non-rigid supports in this sweep.  The
    # 600-um windows are centered from 310 through 3510 um, slightly narrower
    # than the other estimators; stay inside that domain rather than
    # extrapolating ten microns at either edge.
    depths = np.arange(310.0, 3510.1, 200.0)
    fields = {}
    decompositions = {}
    rows = []
    for probe in probes:
        t_start = None
        for candidate in candidates:
            if candidate.family == "exploratory_native":
                continue
            for split in split_names(candidate):
                target = run_dir(output_dir, probe, candidate, split, seed)
                if not (target / "motion.npy").exists():
                    continue
                manifest = json.loads((target / "manifest.json").read_text())
                if t_start is None:
                    times = np.load(target / "time_bins.npy")
                    # A sliced SI recording keeps absolute time; infer the raw
                    # start from the requested frame-relative window.
                    native_dt = float(np.median(np.diff(times)))
                    t_start = float(times[0] - WINDOW.start_s - native_dt / 2)
                target_times = relative_times + t_start
                field = _load_run_field(target, target_times, depths)
                key = (probe, candidate.name, split)
                fields[key] = field
                decompositions[key] = decompose_spatial_field(field, depths)
                quality = json.loads((target / "native_quality.json").read_text())
                rows.append(
                    {
                        "probe": probe,
                        "candidate": candidate.name,
                        "family": candidate.family,
                        "method": candidate.method,
                        "split": split,
                        "n_peaks": manifest["n_peaks"],
                        **summarize_field(field, depths, common_dt),
                        **quality,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "motion_scale_sweep_summary.csv", index=False)

    agreement = []
    for probe in probes:
        full_candidates = [candidate for candidate in candidates if (probe, candidate.name, "full") in fields]
        for left_index, left in enumerate(full_candidates):
            for right in full_candidates[left_index + 1 :]:
                first = decompositions[(probe, left.name, "full")]
                second = decompositions[(probe, right.name, "full")]
                agreement.append(
                    {
                        "scope": "cross_candidate",
                        "probe": probe,
                        "left_candidate": left.name,
                        "right_candidate": right.name,
                        "rigid_correlation": correlation(first["rigid"], second["rigid"]),
                        "nonrigid_correlation": correlation(first["residual"], second["residual"]),
                    }
                )
        for candidate in full_candidates:
            a = (probe, candidate.name, "half_a")
            b = (probe, candidate.name, "half_b")
            if a in fields and b in fields:
                first, second = decompositions[a], decompositions[b]
                agreement.append(
                    {
                        "scope": "split_half",
                        "probe": probe,
                        "left_candidate": candidate.name,
                        "right_candidate": candidate.name,
                        "rigid_correlation": correlation(first["rigid"], second["rigid"]),
                        "nonrigid_correlation": correlation(first["residual"], second["residual"]),
                    }
                )
    if set(probes) == set(PROBES):
        for candidate in candidates:
            left = ("imec0", candidate.name, "full")
            right = ("imec1", candidate.name, "full")
            if left not in fields or right not in fields:
                continue
            first, second = decompositions[left], decompositions[right]
            lag, best = best_lag_correlation(first["rigid"], second["rigid"], 5)
            agreement.append(
                {
                    "scope": "cross_probe",
                    "probe": "imec0_vs_imec1",
                    "left_candidate": candidate.name,
                    "right_candidate": candidate.name,
                    "rigid_correlation": correlation(first["rigid"], second["rigid"]),
                    "nonrigid_correlation": correlation(first["residual"], second["residual"]),
                    "best_lag_s": lag * common_dt,
                    "best_lag_rigid_correlation": best,
                }
            )
    pd.DataFrame(agreement).to_csv(output_dir / "motion_scale_sweep_agreement.csv", index=False)


def plan(probes: list[str], candidates: list[Candidate], output_dir: Path, seed: int) -> dict:
    runs = []
    for probe in probes:
        for candidate in candidates:
            for split in split_names(candidate):
                runs.append(
                    {
                        "probe": probe,
                        "candidate": candidate.name,
                        "split": split,
                        "target": str(run_dir(output_dir, probe, candidate, split, seed)),
                    }
                )
    return {
        "window": asdict(WINDOW),
        "probes": probes,
        "candidates": [asdict(candidate) for candidate in candidates],
        "n_runs": len(runs),
        "runs": runs,
        "selection_principle": (
            "Infer supported temporal/spatial scales from split-half, adjacent-parameter, "
            "cross-method, and cross-probe stability before downstream sorting."
        ),
    }


def main() -> None:
    global WINDOW
    args = parse_args()
    WINDOW = WINDOWS[args.window_name]
    candidates = [candidate_by_name(name) for name in args.candidates]
    planned = plan(args.probes, candidates, args.output_dir, args.seed)
    print(json.dumps(planned, indent=2))
    if args.plan_only:
        return
    if not (args.run or args.summarize):
        raise SystemExit("Choose --plan-only, --run, or --summarize")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sweep_plan.json").write_text(json.dumps(planned, indent=2) + "\n")
    if args.run:
        run_grid(args.output_dir, args.probes, candidates, args.seed)
    if args.summarize:
        summarize(args.output_dir, args.probes, candidates, args.seed)


if __name__ == "__main__":
    main()
