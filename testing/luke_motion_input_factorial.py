"""Controlled estimation-only input factorial for selected Luke windows.

The factorial reuses the historical threshold-5 localized peaks and changes
only which observations are supplied to an estimator.  It never modifies
voltage and never reads sorter labels.  Consequently, the amplitude subset is
a post-detection threshold proxy, not a replacement for a true detection-
threshold rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_motion_scale_characterization import (
    decompose_spatial_field,
    interpolate_field,
    summarize_field,
)


LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
RAW_ROOT = LUKE_ROOT / "Luke0804_V2V1_g0"
WINDOWS_CSV = Path("testing/outputs/luke_motion_regime_windows/selected_windows.csv")
OUTPUT = Path("testing/outputs/luke_motion_input_factorial")
FS = 30_000.0
SEED = 20250804
SCRIPT_VERSION = 1


@dataclass(frozen=True)
class Estimator:
    name: str
    method: str
    kwargs: dict[str, Any]


ESTIMATORS = {
    value.name: value
    for value in (
        Estimator(
            "dredge_300_200_cpu",
            "dredge_ap",
            {
                "bin_um": 1.0,
                "bin_s": 1.0,
                "histogram_time_smooth_s": 1.0,
                "histogram_depth_smooth_um": 1.0,
                "time_horizon_s": 60.0,
                "mincorr": 0.1,
                "device": "cpu",
                "max_disp_um": 80.0,
            },
        ),
        Estimator(
            "decentralized_300_200_numpy",
            "decentralized",
            {
                "bin_s": 1.0,
                "histogram_time_smooth_s": 1.0,
                "max_displacement_um": 80.0,
                "time_horizon_s": 60.0,
                "conv_engine": "numpy",
                "temporal_prior": True,
            },
        ),
        Estimator(
            "iterative_300_200",
            "iterative_template",
            {"bin_s": 2.0, "num_shifts_block": 5},
        ),
    )
}

CONDITIONS = (
    "full",
    "random_half",
    "random_quarter",
    "high_amplitude_half",
    "exclude_synchronous",
    "exclude_bursty_seconds",
    "exclude_channel191",
    "exclude_dominant_channel",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--windows-csv", type=Path, default=WINDOWS_CSV)
    parser.add_argument("--regimes", nargs="+", default=["quiet", "rapid_motion", "sustained_noise", "support_dropout", "noise_plus_motion"])
    parser.add_argument("--probes", nargs="+", choices=("imec0", "imec1"), default=["imec1"])
    parser.add_argument("--estimators", nargs="+", choices=tuple(ESTIMATORS), default=["dredge_300_200_cpu", "decentralized_300_200_numpy"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def motion_root(probe: str) -> Path:
    return LUKE_ROOT / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}" / "motion"


def stable_fraction(peaks: np.ndarray) -> np.ndarray:
    sample = np.asarray(peaks["sample_index"], dtype=np.uint64)
    channel = np.asarray(peaks["channel_index"], dtype=np.uint64)
    mixed = sample * np.uint64(11400714819323198485) + channel * np.uint64(0x9E3779B97F4A7C15)
    return (mixed >> np.uint64(32)).astype(np.float64) / float(2**32)


def select_condition(peaks: np.ndarray, locations: np.ndarray, condition: str) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(peaks)
    keep = np.ones(n, dtype=bool)
    details: dict[str, Any] = {}
    if condition == "random_half":
        keep = stable_fraction(peaks) < 0.5
    elif condition == "random_quarter":
        keep = stable_fraction(peaks) < 0.25
    elif condition == "high_amplitude_half":
        cutoff = float(np.median(np.abs(peaks["amplitude"])))
        keep = np.abs(peaks["amplitude"]) >= cutoff
        details["absolute_amplitude_cutoff"] = cutoff
        details["guardrail"] = "post-detection amplitude subset; not a true detect-threshold rerun"
    elif condition == "exclude_synchronous":
        _, inverse, counts = np.unique(peaks["sample_index"], return_inverse=True, return_counts=True)
        keep = counts[inverse] < 8
        details["excluded_sample_multiplicity_ge"] = 8
    elif condition == "exclude_bursty_seconds":
        seconds = np.floor(peaks["sample_index"] / FS).astype(np.int64)
        second_counts = np.bincount(seconds, minlength=int(seconds.max()) + 1)
        median = float(np.median(second_counts))
        mad = float(np.median(np.abs(second_counts - median)))
        cutoff = median + 3.0 * 1.4826 * mad
        burst = second_counts > cutoff
        keep = ~burst[seconds]
        details.update(burst_second_count=int(np.sum(burst)), burst_count_cutoff=float(cutoff))
    elif condition == "exclude_channel191":
        keep = peaks["channel_index"] != 191
        details["excluded_detection_channel"] = 191
        details["guardrail"] = "peak exclusion only; does not undo historical bad-channel interpolation"
    elif condition == "exclude_dominant_channel":
        channel_counts = np.bincount(np.asarray(peaks["channel_index"], dtype=np.int64))
        dominant = int(np.argmax(channel_counts))
        keep = peaks["channel_index"] != dominant
        details["excluded_detection_channel"] = dominant
        details["dominant_channel_input_fraction"] = float(channel_counts[dominant] / n)
        details["guardrail"] = "data-driven peak exclusion; does not alter or reconstruct voltage"
    elif condition != "full":
        raise KeyError(condition)
    selected_peaks = np.array(peaks[keep], copy=True)
    selected_locations = np.array(locations[keep], copy=True)
    details.update(
        input_peaks=int(n),
        selected_peaks=int(len(selected_peaks)),
        retained_fraction=float(np.mean(keep)),
    )
    return selected_peaks, selected_locations, details


def peak_digest(peaks: np.ndarray, locations: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in (peaks, locations):
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def load_window_peaks(probe: str, start_s: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    root = motion_root(probe)
    all_peaks = np.load(root / "peaks.npy", mmap_mode="r")
    all_locations = np.load(root / "peak_locations.npy", mmap_mode="r")
    start = int(round(start_s * FS))
    stop = int(round((start_s + duration_s) * FS))
    left = int(np.searchsorted(all_peaks["sample_index"], start, side="left"))
    right = int(np.searchsorted(all_peaks["sample_index"], stop, side="left"))
    peaks = np.array(all_peaks[left:right], copy=True)
    locations = np.array(all_locations[left:right], copy=True)
    peaks["sample_index"] -= start
    return peaks, locations


def run_path(output: Path, regime: str, probe: str, estimator: str, condition: str) -> Path:
    return output / "runs" / regime / probe / estimator / condition


def build_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    windows = pd.read_csv(args.windows_csv)
    windows = windows[windows.regime.isin(args.regimes)].copy()
    missing = sorted(set(args.regimes) - set(windows.regime))
    if missing:
        raise ValueError(f"Regimes absent from {args.windows_csv}: {missing}")
    runs = []
    for row in windows.itertuples(index=False):
        for probe in args.probes:
            for estimator in args.estimators:
                for condition in args.conditions:
                    runs.append(
                        {
                            "regime": row.regime,
                            "start_s": float(row.start_s),
                            "duration_s": float(row.duration_s),
                            "probe": probe,
                            "estimator": estimator,
                            "condition": condition,
                            "target": str(run_path(args.output_dir, row.regime, probe, estimator, condition)),
                        }
                    )
    plan = {
        "script_version": SCRIPT_VERSION,
        "source_windows": str(args.windows_csv),
        "source_peaks": "historical threshold-5 localized peaks",
        "estimators": [asdict(ESTIMATORS[name]) for name in args.estimators],
        "conditions": args.conditions,
        "runs": runs,
        "n_runs": len(runs),
        "motion_applied": False,
        "sorter_labels_accessed": False,
    }
    return windows, plan


def run_factorial(args: argparse.Namespace, windows: pd.DataFrame) -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-motion-input-numba")
    import spikeinterface.extractors as se
    from spikeinterface.sortingcomponents.motion import estimate_motion

    for row in windows.itertuples(index=False):
        for probe in args.probes:
            raw = se.read_spikeglx(RAW_ROOT, load_sync_channel=False, stream_id=f"{probe}.ap")
            start_frame = int(round(row.start_s * raw.get_sampling_frequency()))
            end_frame = int(round((row.start_s + row.duration_s) * raw.get_sampling_frequency()))
            recording = raw.frame_slice(start_frame=start_frame, end_frame=end_frame)
            peaks, locations = load_window_peaks(probe, row.start_s, row.duration_s)
            for condition in args.conditions:
                selected_peaks, selected_locations, selection = select_condition(peaks, locations, condition)
                for estimator_name in args.estimators:
                    estimator = ESTIMATORS[estimator_name]
                    target = run_path(args.output_dir, row.regime, probe, estimator_name, condition)
                    manifest = {
                        "script_version": SCRIPT_VERSION,
                        "regime": row.regime,
                        "start_s": float(row.start_s),
                        "duration_s": float(row.duration_s),
                        "probe": probe,
                        "condition": condition,
                        "estimator": asdict(estimator),
                        "selection": selection,
                        "peak_digest": peak_digest(selected_peaks, selected_locations),
                        "motion_applied": False,
                    }
                    if (target / "motion.npy").exists():
                        if json.loads((target / "manifest.json").read_text()) != manifest:
                            raise RuntimeError(f"Cache manifest mismatch: {target}")
                        print(f"Reusing {target}", flush=True)
                        continue
                    if target.exists():
                        raise RuntimeError(f"Ambiguous partial run: {target}")
                    target.mkdir(parents=True)
                    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    print(
                        f"Running {row.regime} {probe} {estimator_name} {condition}: "
                        f"{len(selected_peaks)} peaks",
                        flush=True,
                    )
                    motion = estimate_motion(
                        recording=recording,
                        peaks=selected_peaks,
                        peak_locations=selected_locations,
                        direction="y",
                        rigid=False,
                        win_shape="gaussian",
                        win_step_um=200.0,
                        win_scale_um=300.0,
                        method=estimator.method,
                        extra_outputs=False,
                        progress_bar=False,
                        verbose=False,
                        **estimator.kwargs,
                    )
                    time_origin = float(recording.get_time_info().get("t_start") or 0.0)
                    np.save(target / "motion.npy", motion.displacement[0])
                    np.save(target / "time_bins_relative_s.npy", motion.temporal_bins_s[0] - time_origin)
                    np.save(target / "depth_bins_um.npy", motion.spatial_bins_um)


def load_common_field(target: Path, duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(1.0, duration_s, 2.0)
    depths = np.arange(310.0, 3510.1, 200.0)
    field = interpolate_field(
        np.load(target / "motion.npy"),
        np.load(target / "time_bins_relative_s.npy"),
        np.load(target / "depth_bins_um.npy"),
        times,
        depths,
    )
    return field, times, depths


def corr(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float).ravel()
    y = np.asarray(second, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def summarize(args: argparse.Namespace, plan: dict) -> None:
    fields: dict[tuple[str, str, str, str], np.ndarray] = {}
    decomposed = {}
    rows = []
    completed = sorted((args.output_dir / "runs").rglob("manifest.json"))
    for manifest_path in completed:
        target = manifest_path.parent
        if not (target / "motion.npy").exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        spec = {
            "regime": manifest["regime"],
            "start_s": manifest["start_s"],
            "duration_s": manifest["duration_s"],
            "probe": manifest["probe"],
            "estimator": manifest["estimator"]["name"],
            "condition": manifest["condition"],
        }
        field, _, depths = load_common_field(target, float(spec["duration_s"]))
        key = (spec["regime"], spec["probe"], spec["estimator"], spec["condition"])
        fields[key] = field
        decomposed[key] = decompose_spatial_field(field, depths)
        rows.append(
            {
                **{name: spec[name] for name in ("regime", "start_s", "duration_s", "probe", "estimator", "condition")},
                **manifest["selection"],
                **summarize_field(field, depths, 2.0),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "factorial_field_summary.csv", index=False)

    agreements = []
    estimator_names = sorted({key[2] for key in fields})
    for key, field in fields.items():
        regime, probe, estimator, condition = key
        full_key = (regime, probe, estimator, "full")
        if full_key in fields:
            first = decomposed[key]
            base = decomposed[full_key]
            agreements.append(
                {
                    "scope": "condition_vs_full",
                    "regime": regime,
                    "probe": probe,
                    "condition": condition,
                    "left_estimator": estimator,
                    "right_estimator": estimator,
                    "rigid_correlation": corr(first["rigid"], base["rigid"]),
                    "nonrigid_correlation": corr(first["residual"], base["residual"]),
                }
            )
        for other in estimator_names:
            if other <= estimator:
                continue
            other_key = (regime, probe, other, condition)
            if other_key not in fields:
                continue
            first = decomposed[key]
            second = decomposed[other_key]
            agreements.append(
                {
                    "scope": "cross_estimator",
                    "regime": regime,
                    "probe": probe,
                    "condition": condition,
                    "left_estimator": estimator,
                    "right_estimator": other,
                    "rigid_correlation": corr(first["rigid"], second["rigid"]),
                    "nonrigid_correlation": corr(first["residual"], second["residual"]),
                }
            )
        other_probe = "imec1" if probe == "imec0" else "imec0"
        other_probe_key = (regime, other_probe, estimator, condition)
        if probe == "imec0" and other_probe_key in fields:
            first = decomposed[key]
            second = decomposed[other_probe_key]
            agreements.append(
                {
                    "scope": "cross_probe",
                    "regime": regime,
                    "probe": "imec0_vs_imec1",
                    "condition": condition,
                    "left_estimator": estimator,
                    "right_estimator": estimator,
                    "rigid_correlation": corr(first["rigid"], second["rigid"]),
                    "nonrigid_correlation": corr(first["residual"], second["residual"]),
                }
            )
    pd.DataFrame(agreements).to_csv(args.output_dir / "factorial_agreement.csv", index=False)
    completed_manifest = {
        "script_version": SCRIPT_VERSION,
        "completed_runs": int(len(fields)),
        "completed_by_probe": summary.groupby("probe").size().astype(int).to_dict(),
        "completed_by_estimator": summary.groupby("estimator").size().astype(int).to_dict(),
        "motion_applied": False,
        "sorter_labels_accessed": False,
        "source_peaks": "historical threshold-5 localized peaks",
    }
    (args.output_dir / "completed_run_manifest.json").write_text(
        json.dumps(completed_manifest, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    windows, plan = build_plan(args)
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return
    if not (args.run or args.summarize):
        raise SystemExit("Choose --plan-only, --run, or --summarize")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "factorial_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.run:
        run_factorial(args, windows)
    if args.summarize:
        summarize(args, plan)


if __name__ == "__main__":
    main()
