"""Historical-equivalent detection-threshold audit for Luke 2025-08-04.

The original random noise sample was not persisted, so this uses the historical
threshold-5 peaks and localized positions and infers each channel's accepted
amplitude boundary from the full session.  Multiples of 6/5 and 7/5 then form
channel-specific, exactly nested subsets approximating true thresholds 6 and
7 without changing preprocessing or localization.  No motion is applied and
no sorter output or unit label is accessed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_motion_input_factorial import ESTIMATORS
from testing.luke_motion_scale_characterization import (
    decompose_spatial_field,
    interpolate_field,
    summarize_field,
)


LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
RAW_ROOT = LUKE_ROOT / "Luke0804_V2V1_g0"
HISTORICAL_ROOT = LUKE_ROOT / "dredge_pipeline_results_Luke0804_V2V1_g0_imec1" / "motion"
WINDOWS_CSV = Path("testing/outputs/luke_motion_regime_windows/selected_windows.csv")
OUTPUT = Path("testing/outputs/luke_motion_detection_threshold_factorial")
STREAM_ID = "imec1.ap"
THRESHOLDS = (5.0, 6.0, 7.0)
ESTIMATOR_NAMES = ("dredge_300_200_cpu", "decentralized_300_200_numpy")
SCRIPT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--windows-csv", type=Path, default=WINDOWS_CSV)
    parser.add_argument("--regimes", nargs="+", default=[
        "quiet", "rapid_motion", "sustained_noise", "support_dropout", "noise_plus_motion"
    ])
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    parser.add_argument("--estimators", nargs="+", choices=ESTIMATOR_NAMES, default=list(ESTIMATOR_NAMES))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def corr(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=float).ravel()
    y = np.asarray(second, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def load_windows(args: argparse.Namespace) -> pd.DataFrame:
    windows = pd.read_csv(args.windows_csv)
    windows = windows[windows.regime.isin(args.regimes)].copy()
    missing = sorted(set(args.regimes) - set(windows.regime))
    if missing:
        raise ValueError(f"Regimes absent from {args.windows_csv}: {missing}")
    return windows


def threshold_label(value: float) -> str:
    return f"threshold_{value:g}".replace(".", "p")


def estimator_target(output: Path, regime: str, threshold: float, estimator: str) -> Path:
    return output / "runs" / regime / threshold_label(threshold) / estimator


def peak_target(output: Path, regime: str) -> Path:
    return output / "peaks" / regime


def historical_window(start_s: float, duration_s: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    peaks = np.load(HISTORICAL_ROOT / "peaks.npy", mmap_mode="r")
    locations = np.load(HISTORICAL_ROOT / "peak_locations.npy", mmap_mode="r")
    start = int(round(start_s * fs))
    stop = int(round((start_s + duration_s) * fs))
    left = int(np.searchsorted(peaks["sample_index"], start, side="left"))
    right = int(np.searchsorted(peaks["sample_index"], stop, side="left"))
    result = np.array(peaks[left:right], copy=True)
    result_locations = np.array(locations[left:right], copy=True)
    result["sample_index"] -= start
    return result, result_locations


def infer_historical_detection_boundaries() -> np.ndarray:
    """Infer per-channel threshold-5 amplitude floors from all accepted peaks."""
    peaks = np.load(HISTORICAL_ROOT / "peaks.npy", mmap_mode="r")
    n_channels = int(np.max(peaks["channel_index"])) + 1
    boundaries = np.full(n_channels, np.inf)
    np.minimum.at(
        boundaries,
        np.asarray(peaks["channel_index"], dtype=np.int64),
        -np.asarray(peaks["amplitude"], dtype=float),
    )
    if not np.all(np.isfinite(boundaries)):
        raise ValueError("Cannot infer a threshold boundary for every channel")
    return boundaries


def overlap_fraction(left: np.ndarray, right: np.ndarray) -> float:
    left_pairs = np.rec.fromarrays([left["sample_index"], left["channel_index"]])
    right_pairs = np.rec.fromarrays([right["sample_index"], right["channel_index"]])
    if len(left_pairs) == 0:
        return np.nan
    return float(np.isin(left_pairs, right_pairs).mean())


def save_motion(target: Path, motion, time_origin: float, manifest: dict) -> None:
    target.mkdir(parents=True, exist_ok=False)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.save(target / "motion.npy", motion.displacement[0])
    np.save(target / "time_bins_relative_s.npy", motion.temporal_bins_s[0] - time_origin)
    np.save(target / "depth_bins_um.npy", motion.spatial_bins_um)


def run(args: argparse.Namespace, windows: pd.DataFrame) -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-motion-threshold-numba")
    import spikeinterface.extractors as se
    from spikeinterface.sortingcomponents.motion import estimate_motion

    raw = se.read_spikeglx(RAW_ROOT, load_sync_channel=False, stream_id=STREAM_ID)
    boundaries = infer_historical_detection_boundaries()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "inferred_historical_threshold5_boundaries.npy", boundaries)

    for row in windows.itertuples(index=False):
        target = peak_target(args.output_dir, row.regime)
        start_frame = int(round(float(row.start_s) * raw.get_sampling_frequency()))
        stop_frame = int(round((float(row.start_s) + float(row.duration_s)) * raw.get_sampling_frequency()))
        recording = raw.frame_slice(start_frame=start_frame, end_frame=stop_frame)
        base_peaks, base_locations = historical_window(
            float(row.start_s), float(row.duration_s), raw.get_sampling_frequency()
        )
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(json.dumps({
            "script_version": SCRIPT_VERSION,
            "regime": row.regime,
            "start_s": float(row.start_s),
            "duration_s": float(row.duration_s),
            "stream_id": STREAM_ID,
            "base_threshold": 5.0,
            "base_threshold_peak_count": int(len(base_peaks)),
            "boundary_inference": "minimum accepted negative amplitude per channel over full historical session",
            "motion_applied": False,
            "sorter_labels_accessed": False,
        }, indent=2) + "\n")

        for threshold in args.thresholds:
            if threshold < 5.0:
                raise ValueError("Historical threshold-5 cache cannot reconstruct thresholds below 5")
            keep = -np.asarray(base_peaks["amplitude"], dtype=float) >= (
                threshold / 5.0
            ) * boundaries[base_peaks["channel_index"]]
            peaks = base_peaks[keep]
            locations = base_locations[keep]
            for estimator_name in args.estimators:
                estimator = ESTIMATORS[estimator_name]
                out = estimator_target(args.output_dir, row.regime, threshold, estimator_name)
                manifest = {
                    "script_version": SCRIPT_VERSION,
                    "regime": row.regime,
                    "start_s": float(row.start_s),
                    "duration_s": float(row.duration_s),
                    "stream_id": STREAM_ID,
                    "threshold": float(threshold),
                    "peak_count": int(len(peaks)),
                    "estimator": {"name": estimator.name, "method": estimator.method, "kwargs": estimator.kwargs},
                    "motion_applied": False,
                    "sorter_labels_accessed": False,
                }
                if (out / "motion.npy").exists():
                    if json.loads((out / "manifest.json").read_text()) != manifest:
                        raise RuntimeError(f"Cache manifest mismatch: {out}")
                    continue
                if out.exists():
                    raise RuntimeError(f"Ambiguous partial estimator cache: {out}")
                print(f"Estimating {row.regime} threshold {threshold:g} {estimator_name}: {len(peaks):,} peaks", flush=True)
                motion = estimate_motion(
                    recording=recording,
                    peaks=peaks,
                    peak_locations=locations,
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
                save_motion(out, motion, time_origin, manifest)


def summarize(args: argparse.Namespace) -> None:
    rows = []
    fields = {}
    parts = []
    for manifest_path in sorted((args.output_dir / "runs").rglob("manifest.json")):
        target = manifest_path.parent
        if not (target / "motion.npy").exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        duration = float(manifest["duration_s"])
        times = np.arange(1.0, duration, 2.0)
        depths = np.arange(310.0, 3510.1, 200.0)
        field = interpolate_field(
            np.load(target / "motion.npy"),
            np.load(target / "time_bins_relative_s.npy"),
            np.load(target / "depth_bins_um.npy"),
            times,
            depths,
        )
        key = (manifest["regime"], manifest["estimator"]["name"], float(manifest["threshold"]))
        fields[key] = decompose_spatial_field(field, depths)
        rows.append({
            "regime": manifest["regime"],
            "estimator": manifest["estimator"]["name"],
            "threshold": float(manifest["threshold"]),
            "peak_count": int(manifest["peak_count"]),
            **summarize_field(field, depths, 2.0),
        })
    for key, value in fields.items():
        regime, estimator, threshold = key
        base = fields.get((regime, estimator, 5.0))
        if base is None:
            continue
        parts.append({
            "regime": regime,
            "estimator": estimator,
            "threshold": threshold,
            "rigid_correlation_vs_threshold5": corr(value["rigid"], base["rigid"]),
            "nonrigid_correlation_vs_threshold5": corr(value["residual"], base["residual"]),
        })
    cross_estimator = []
    for regime in sorted({key[0] for key in fields}):
        for threshold in sorted({key[2] for key in fields if key[0] == regime}):
            dredge = fields.get((regime, "dredge_300_200_cpu", threshold))
            decentralized = fields.get((regime, "decentralized_300_200_numpy", threshold))
            if dredge is None or decentralized is None:
                continue
            cross_estimator.append({
                "regime": regime,
                "threshold": threshold,
                "rigid_dredge_decentralized_correlation": corr(dredge["rigid"], decentralized["rigid"]),
                "nonrigid_dredge_decentralized_correlation": corr(dredge["residual"], decentralized["residual"]),
            })
    summary = pd.DataFrame(rows).sort_values(["regime", "estimator", "threshold"])
    agreements = pd.DataFrame(parts).sort_values(["regime", "estimator", "threshold"])
    summary.to_csv(args.output_dir / "threshold_field_summary.csv", index=False)
    agreements.to_csv(args.output_dir / "threshold_agreement.csv", index=False)
    pd.DataFrame(cross_estimator).to_csv(
        args.output_dir / "threshold_cross_estimator_agreement.csv", index=False
    )
    peak_validation = [json.loads(path.read_text()) for path in sorted((args.output_dir / "peaks").rglob("manifest.json"))]
    pd.DataFrame(peak_validation).to_csv(args.output_dir / "historical_threshold5_inputs.csv", index=False)
    (args.output_dir / "completed_run_manifest.json").write_text(json.dumps({
        "script_version": SCRIPT_VERSION,
        "completed_runs": int(len(summary)),
        "thresholds": sorted(summary.threshold.unique().tolist()),
        "regimes": sorted(summary.regime.unique().tolist()),
        "motion_applied": False,
        "sorter_labels_accessed": False,
    }, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    windows = load_windows(args)
    plan = {
        "script_version": SCRIPT_VERSION,
        "stream_id": STREAM_ID,
        "regimes": windows.regime.tolist(),
        "thresholds": args.thresholds,
        "estimators": args.estimators,
        "conditioning": "historical threshold-5 localized peak cache from the ap_300_3000 branch",
        "nested_detection": "infer the original per-channel accepted-amplitude floor; derive 6/7 using exact 6/5 and 7/5 multiples",
        "motion_applied": False,
        "sorter_labels_accessed": False,
    }
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return
    if not (args.run or args.summarize):
        raise SystemExit("Choose --plan-only, --run, or --summarize")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "threshold_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.run:
        run(args, windows)
    if args.summarize:
        summarize(args)


if __name__ == "__main__":
    main()
