"""Development-only W2 RF evaluator for CP saved final-label arrays.

This adapter keeps the accepted dots evaluator's stimulus, gaze, lag, filtering,
eligibility, and scoring code, but makes the cropped-sort provenance explicit.
It freezes both trial splits on the original 81-trial universe and then selects
only complete W2 trials.  The outer holdout is never scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import RANSACRegressor


DARTSORT_ROOT = Path("/home/huklab/Documents/DARTsort")
sys.path.insert(0, str(DARTSORT_ROOT))
from experiments.luke0804_imec1.dots_rf_common import (  # noqa: E402
    bin_sorting_to_trial_frames,
    crossvalidated_sta_snr,
    frozen_kfold_trial_folds,
)
from experiments.luke0804_imec1.evaluate_dots_rf import (  # noqa: E402
    lag_safe_mask,
    lagged_stimulus_gaze_mask,
    load_gaze,
    validate_gaze_calibration_receipt,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def explicit_imec1_clock(path: Path):
    """Fit only imec1Time -> NidaqTime; never accept an imec0 fallback."""
    mat = loadmat(path, simplify_cells=True)
    matches = [
        (name, value)
        for name, value in mat.items()
        if isinstance(value, dict)
        and "imec1Time" in value
        and "NidaqTime" in value
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one explicit imec1 clock struct, got {len(matches)}")
    struct_name, struct = matches[0]
    imec = np.asarray(struct["imec1Time"], dtype=np.float64).reshape(-1)
    nidaq = np.asarray(struct["NidaqTime"], dtype=np.float64).reshape(-1)
    n = min(imec.size, nidaq.size)
    finite = np.isfinite(imec[:n]) & np.isfinite(nidaq[:n])
    imec, nidaq = imec[:n][finite], nidaq[:n][finite]
    model = RANSACRegressor(random_state=1002, residual_threshold=100.0).fit(
        imec.reshape(-1, 1), nidaq
    )
    slope = float(model.estimator_.coef_[0])
    intercept = float(model.estimator_.intercept_)
    residual = nidaq - (slope * imec + intercept)
    info = {
        "source_struct": struct_name,
        "imec_field": "imec1Time",
        "n_pairs": int(n),
        "n_finite_pairs": int(imec.size),
        "n_inliers": int(model.inlier_mask_.sum()),
        "slope": slope,
        "intercept_samples": intercept,
        "residual_median_samples": float(np.median(residual)),
        "residual_mad_samples": float(
            np.median(np.abs(residual - np.median(residual)))
        ),
        "residual_max_abs_samples": float(np.max(np.abs(residual))),
    }
    return lambda samples: slope * np.asarray(samples) + intercept, info


def parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be NAME=NPZ")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("arm must be NAME=NPZ")
    return name, Path(path)


def atomic_json(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stimulus-cache", type=Path, required=True)
    parser.add_argument("--gaze-csv", type=Path, required=True)
    parser.add_argument("--interval-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing existing output: {output} or {partial}")
    partial.mkdir(parents=True)
    try:
        config = json.loads(args.config.read_text())
        metadata = json.loads(args.interval_metadata.read_text())
        fs = float(metadata["sampling_frequency_hz"])
        source_start, source_end = map(
            int, metadata["exact_source_frame_interval_half_open"]
        )
        n_local = int(metadata["local_frame_count"])
        if source_end - source_start != n_local:
            raise ValueError("Inconsistent explicit W2 frame metadata")
        imec_to_nidaq, clock_info = explicit_imec1_clock(
            Path(config["imec_to_nidaq_timing_mat"])
        )
        nidaq_fs = float(config["nidaq_sampling_frequency_hz"])
        support_left, support_right = (
            float(x) / nidaq_fs
            for x in imec_to_nidaq(np.asarray([source_start, source_end]))
        )

        cache = args.stimulus_cache.resolve()
        cache_manifest = json.loads((cache / "manifest.json").read_text())
        for filename, key in (
            ("stimulus_frames.npz", "stimulus_frames_sha256"),
            ("trial_table.csv", "trial_table_sha256"),
            ("trial_split.json", "trial_split_sha256"),
        ):
            if sha256(cache / filename) != cache_manifest[key]:
                raise ValueError(f"Stimulus-cache hash mismatch: {filename}")
        with np.load(cache / "stimulus_frames.npz", allow_pickle=False) as data:
            positive = np.asarray(data["valid_positive_time"], dtype=bool)
            centers = np.asarray(data["t_bin_centers_s"], dtype=np.float64)[positive]
            left = np.asarray(data["t_bin_left_s"], dtype=np.float64)[positive]
            right = np.asarray(data["t_bin_right_s"], dtype=np.float64)[positive]
            dots = np.asarray(data["dots_pix"])[positive]
            trials = np.asarray(data["trial_ordinals"], dtype=np.int64)[positive]
            ppd = float(data["pix_per_deg"])

        # Freeze outer and inner assignments on the original full trial universe.
        unique_trials = np.unique(trials)
        selection = config["parameter_selection"]
        outer = frozen_kfold_trial_folds(
            unique_trials,
            seed=int(selection["outer_split_seed"]),
            n_folds=int(selection["outer_folds"]),
            n_temporal_blocks=int(selection["outer_temporal_blocks"]),
        )
        holdout_fold = int(selection["final_holdout_fold"])
        development = {
            int(t) for t in unique_trials if outer[int(t)] != holdout_fold
        }
        holdout = {int(t) for t in unique_trials if outer[int(t)] == holdout_fold}
        if (len(development), len(holdout)) != (
            int(selection["expected_development_trials"]),
            int(selection["expected_final_holdout_trials"]),
        ):
            raise ValueError("Frozen outer split count mismatch")
        inner = {
            int(key): int(value)
            for key, value in json.loads((cache / "trial_split.json").read_text())[
                "fold_by_trial_ordinal"
            ].items()
        }
        complete_trials = {
            int(t)
            for t in unique_trials
            if np.all(left[trials == t] >= support_left)
            and np.all(right[trials == t] <= support_right)
        }
        selected_development = complete_trials & development
        selected_holdout = complete_trials & holdout
        if selected_holdout & selected_development:
            raise AssertionError("Outer split overlap")
        gaze_calibration = validate_gaze_calibration_receipt(
            config, args.gaze_csv.resolve(), development, holdout
        )

        # Reuse the accepted evaluator's exact gaze, lag-safety and lagged-gaze logic.
        gaze_args = argparse.Namespace(
            gaze_csv=args.gaze_csv,
            gaze_time_column="t_ephys",
            gaze_x_column="i",
            gaze_y_column="j",
            gaze_valid_column="valid",
        )
        frame_gaze, gaze_valid = load_gaze(gaze_args, centers)
        lags = np.asarray(config["lags"], dtype=np.int64)
        safe = lag_safe_mask(trials, int(lags.max()))
        lagged_gaze = lagged_stimulus_gaze_mask(gaze_valid, trials, lags)
        selected_frame = np.asarray(
            [int(t) in selected_development for t in trials], dtype=bool
        )
        base_valid = safe & lagged_gaze & selected_frame
        frame_fold = np.asarray([inner[int(t)] for t in trials], dtype=np.int8)

        rowley_repo = Path(config["data_rowley_repo"]).resolve()
        sys.path.insert(0, str(rowley_repo))
        import DataRowleyV1V2.dots_calibration.training as training_module
        import DataRowleyV1V2.utils.rf as rf_module

        half = float(config["roi_half_width_deg"])
        roi_pix = np.flipud(np.asarray([[-half, half], [-half, half]]) * ppd)
        dxy_pix = float(config["dxy_deg"]) * ppd
        edges = [np.arange(lo, hi + dxy_pix, dxy_pix) for lo, hi in roi_pix]
        stim = training_module.bin_dots_to_stimulus(dots, frame_gaze, *edges)
        reverse = bool(config.get("sta_reverse_correlate", True))
        arm_summaries = {}
        for arm_name, arm_path in args.arm:
            with np.load(arm_path, allow_pickle=False) as data:
                if "times_samples" not in data or "labels" not in data:
                    raise ValueError(f"{arm_name}: missing final times_samples/labels")
                times = np.asarray(data["times_samples"], dtype=np.int64)
                labels = np.asarray(data["labels"], dtype=np.int64)
                if "sampling_frequency" in data:
                    arm_fs = float(data["sampling_frequency"])
                    if not np.isclose(arm_fs, fs, rtol=0, atol=1e-9):
                        raise ValueError(f"{arm_name}: sampling frequency mismatch")
            if times.shape != labels.shape or times.ndim != 1:
                raise ValueError(f"{arm_name}: unaligned final event arrays")
            if np.any(times < 0) or np.any(times >= n_local):
                raise ValueError(f"{arm_name}: local sample outside explicit W2 support")
            keep = labels >= 0
            labels_kept = labels[keep]
            units = np.unique(labels_kept)
            original_samples = times[keep] + source_start
            spike_times_s = imec_to_nidaq(original_samples) / nidaq_fs
            robs = bin_sorting_to_trial_frames(
                spike_times_s, labels_kept, units, left, right, trials
            )
            fold_spikes = np.stack(
                [robs[base_valid & (frame_fold == fold)].sum(axis=0) for fold in (0, 1)]
            )
            total_spikes = fold_spikes.sum(axis=0)
            eligible = (
                (total_spikes >= int(config["minimum_total_spikes"]))
                & np.all(
                    fold_spikes >= int(config["minimum_spikes_per_fold"]), axis=0
                )
            )
            eligible_units = units[eligible]
            eligible_robs = robs[:, eligible]
            pd.DataFrame(
                {
                    "arm": arm_name,
                    "unit_id": units,
                    "n_spikes": total_spikes,
                    "n_spikes_fold0": fold_spikes[0],
                    "n_spikes_fold1": fold_spikes[1],
                    "eligible": eligible,
                }
            ).to_csv(partial / f"{arm_name}_rf_eligibility.csv", index=False)
            if eligible_units.size == 0:
                arm_summaries[arm_name] = {
                    "status": "unavailable_no_eligible_units",
                    "n_units_before_eligibility": int(units.size),
                }
                continue
            stas = []
            for fold in (0, 1):
                dfs = base_valid & (frame_fold == fold)
                sta = rf_module.calc_sta(
                    stim,
                    eligible_robs,
                    lags,
                    dfs=dfs.astype(np.float32),
                    reverse_correlate=reverse,
                    batch_size=args.batch_size,
                    device=args.device,
                    progress=True,
                ).cpu().numpy()
                if sta.ndim == 5 and sta.shape[2] == 1:
                    sta = sta[:, :, 0]
                stas.append(sta)
            pooled = rf_module.calc_sta(
                stim,
                eligible_robs,
                lags,
                dfs=base_valid.astype(np.float32),
                reverse_correlate=reverse,
                batch_size=args.batch_size,
                device=args.device,
                progress=True,
            ).cpu().numpy()
            if pooled.ndim == 5 and pooled.shape[2] == 1:
                pooled = pooled[:, :, 0]
            cv = crossvalidated_sta_snr(
                stas[0], stas[1], dxy_deg=float(config["dxy_deg"])
            )
            pooled_snr, _, _ = training_module.calculate_rf_snr(
                pooled, dxy_deg=float(config["dxy_deg"])
            )
            rows = []
            for index, unit in enumerate(eligible_units):
                rows.append(
                    {
                        "arm": arm_name,
                        "unit_id": int(unit),
                        "n_spikes": int(total_spikes[eligible][index]),
                        "n_spikes_fold0": int(fold_spikes[0, eligible][index]),
                        "n_spikes_fold1": int(fold_spikes[1, eligible][index]),
                        "cv_snr": float(cv["cv_snr"][index]),
                        "fold_map_cosine": float(cv["fold_map_cosine"][index]),
                        "pooled_snr_descriptive": float(pooled_snr[index]),
                    }
                )
            pd.DataFrame(rows).to_csv(partial / f"{arm_name}_unit_rf_metrics.csv", index=False)
            values = np.asarray([row["cv_snr"] for row in rows])
            arm_summaries[arm_name] = {
                "status": "complete_development_only",
                "n_units_before_eligibility": int(units.size),
                "n_units_eligible": int(eligible_units.size),
                "cv_snr_median": float(np.median(values)),
                "cv_snr_mean": float(np.mean(values)),
            }

        support = {
            "source_frame_interval_half_open": [source_start, source_end],
            "mapped_nidaq_seconds_half_open": [support_left, support_right],
            "complete_trial_ordinals": sorted(complete_trials),
            "development_trial_ordinals": sorted(selected_development),
            "outer_holdout_trial_ordinals_not_evaluated": sorted(selected_holdout),
            "development_inner_fold_counts": {
                str(fold): sum(inner[t] == fold for t in selected_development)
                for fold in (0, 1)
            },
            "development_valid_frames": int(base_valid.sum()),
            "outer_holdout_opened": False,
        }
        summary = {
            "status": "complete" if all(
                row["status"] == "complete_development_only"
                for row in arm_summaries.values()
            ) else "finite_unavailable_endpoint",
            "arms": arm_summaries,
            "support": support,
            "clock": clock_info,
            "gaze_calibration": gaze_calibration,
            "hashes": {
                "config": sha256(args.config),
                "stimulus_manifest": sha256(cache / "manifest.json"),
                "stimulus_frames": sha256(cache / "stimulus_frames.npz"),
                "trial_table": sha256(cache / "trial_table.csv"),
                "trial_split": sha256(cache / "trial_split.json"),
                "gaze_csv": sha256(args.gaze_csv),
                "interval_metadata": sha256(args.interval_metadata),
                **{f"arm:{name}": sha256(path) for name, path in args.arm},
            },
            "elapsed_s": time.perf_counter() - started,
            "scientific_scope": (
                "Common development-only W2 RF screen. Inner CV is conditional on "
                "the accepted shared gaze calibration trained on all 61 development trials."
            ),
        }
        atomic_json(partial / "SUMMARY.json", summary)
        products = []
        for path in sorted(partial.iterdir()):
            products.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
        atomic_json(
            partial / "COMPLETE.json",
            {"status": summary["status"], "products": products, "written_last": True},
        )
        os.replace(partial, output)
    except Exception:
        # Preserve partial evidence for diagnosis; never present it as accepted output.
        raise


if __name__ == "__main__":
    main()
