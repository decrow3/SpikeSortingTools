"""Replicate no correction versus 0.25x rigid DREDGE on Luke's shared window."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import (
    ClaimSetting,
    PIPELINE_ROOT,
    WINDOWS,
    build_sorter_params,
    cross_unit_near_coincident_fraction,
    event_local_samples,
    events_in_window,
    load_reference_settings,
    local_match_mask,
    window_frames,
)
from testing.luke_motion_candidate_sort import field_displacement
from testing.luke_upstream_sorter_ablation import parse_extraction_counts
from testing.luke_upstream_stage_ablation import DEFAULT_REVIEW, RAW_ROOT, STREAM_ID, build_recording_stages


WINDOW = next(window for window in WINDOWS if window.name == "shared_template")
OUTPUT_ROOT = PIPELINE_ROOT / "motion_candidate_replication/shared_template"
MOTION_PARENT = PIPELINE_ROOT / "motion_scale_sweep/runs/imec1/dredge_nr_200_300"
PRESET_SCALE_MOTION_PARENT = (
    PIPELINE_ROOT / "motion_scale_sweep/runs/imec1/dredge_nr_400_400_preset_scale"
)
CLAIM_OFF = ClaimSetting("claim_off", 0.0, 0.0)
CONDITION_SIGMA_UM = {
    "no_external_correction": None,
    "rigid_gain_025": 20.0,
    "rigid_gain_025_sigma10": 10.0,
    "rigid_gain_025_p2": 20.0,
    "single_ks_preprocessing": None,
    "single_ks_preprocessing_rigid_gain_025_p2": 20.0,
    "single_ks_preprocessing_dredge_400_400_p2": 20.0,
}
CONDITIONS = tuple(CONDITION_SIGMA_UM)
P2_CONDITIONS = {
    "rigid_gain_025_p2",
    "single_ks_preprocessing_rigid_gain_025_p2",
    "single_ks_preprocessing_dredge_400_400_p2",
}
SINGLE_PASS_CONDITIONS = {
    "single_ks_preprocessing",
    "single_ks_preprocessing_rigid_gain_025_p2",
    "single_ks_preprocessing_dredge_400_400_p2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--n-jitters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20250804)
    return parser.parse_args()


def _shared_motion_path(parent: Path) -> Path:
    matches = []
    for target in parent.glob("full_*"):
        manifest = target / "manifest.json"
        if not manifest.exists() or not (target / "motion.npy").exists():
            continue
        spec = json.loads(manifest.read_text())
        if spec.get("window", {}).get("name") == WINDOW.name:
            matches.append(target)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one complete shared-window motion run, found {matches}")
    return matches[0]


def shared_motion_path() -> Path:
    return _shared_motion_path(MOTION_PARENT)


def recording_path(condition: str) -> Path:
    return OUTPUT_ROOT / "recordings" / condition


def sort_path(condition: str) -> Path:
    return OUTPUT_ROOT / "sorts" / condition


def prepare_recordings() -> None:
    import spikeinterface.extractors as se
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    raw = se.read_spikeglx(folder_path=RAW_ROOT, load_sync_channel=False, stream_id=STREAM_ID)
    stages, _, _ = build_recording_stages(raw)
    fs = float(raw.get_sampling_frequency())
    start, stop = window_frames(WINDOW, fs)
    baseline = stages["current_conditioned"].frame_slice(start_frame=start, end_frame=stop)

    motion_path = shared_motion_path()
    displacement = field_displacement(np.load(motion_path / "motion.npy"), "rigid_gain_025")
    motion = Motion(
        displacement=displacement,
        temporal_bins_s=np.load(motion_path / "time_bins.npy"),
        spatial_bins_um=np.load(motion_path / "depth_bins.npy"),
    )
    preset_scale_path = _shared_motion_path(PRESET_SCALE_MOTION_PARENT)
    preset_scale_motion = Motion(
        displacement=np.load(preset_scale_path / "motion.npy"),
        temporal_bins_s=np.load(preset_scale_path / "time_bins.npy"),
        spatial_bins_um=np.load(preset_scale_path / "depth_bins.npy"),
    )
    recordings = {
        "no_external_correction": baseline,
        "single_ks_preprocessing": stages["interpolated_unfiltered"].frame_slice(
            start_frame=start, end_frame=stop
        ),
    }
    single_pass_baseline = recordings["single_ks_preprocessing"]
    for condition, sigma_um in CONDITION_SIGMA_UM.items():
        if sigma_um is None:
            continue
        p = 2 if condition in P2_CONDITIONS else 1
        border_mode = "force_extrapolate" if p == 2 else "force_zeros"
        interpolation_source = (
            single_pass_baseline if condition in SINGLE_PASS_CONDITIONS else baseline
        )
        interpolation_motion = (
            preset_scale_motion
            if condition == "single_ks_preprocessing_dredge_400_400_p2"
            else motion
        )
        recordings[condition] = interpolate_motion(
            interpolation_source.astype("float"),
            interpolation_motion,
            border_mode=border_mode,
            spatial_interpolation_method="kriging",
            sigma_um=sigma_um,
            p=p,
        ).astype("int16")
    for condition, recording in recordings.items():
        target = recording_path(condition)
        metadata = target / "window_manifest.json"
        if target.exists():
            if not metadata.exists():
                raise RuntimeError(f"Ambiguous existing recording directory: {target}")
            print(f"Reusing {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        recording.save(folder=target, n_jobs=-1, chunk_duration="1s", progress_bar=True)
        metadata.write_text(
            json.dumps(
                {
                    "condition": condition,
                    "window": WINDOW.name,
                    "start_sample": start,
                    "stop_sample": stop,
                    "source_stage": (
                        "interpolated_unfiltered"
                        if condition in SINGLE_PASS_CONDITIONS
                        else "current_conditioned"
                    ),
                    "motion_run": (
                        str(preset_scale_path)
                        if condition == "single_ks_preprocessing_dredge_400_400_p2"
                        else str(motion_path)
                    )
                    if CONDITION_SIGMA_UM[condition] is not None
                    else None,
                    "rigid_gain": (
                        None
                        if condition == "single_ks_preprocessing_dredge_400_400_p2"
                        else 0.25
                    )
                    if CONDITION_SIGMA_UM[condition] is not None
                    else 0.0,
                    "spatial_interpolation_method": "kriging" if CONDITION_SIGMA_UM[condition] is not None else None,
                    "spatial_interpolation_sigma_um": CONDITION_SIGMA_UM[condition],
                    "spatial_interpolation_p": (
                        2 if condition in P2_CONDITIONS else 1
                    )
                    if CONDITION_SIGMA_UM[condition] is not None
                    else None,
                    "interpolation_border_mode": (
                        "force_extrapolate"
                        if condition in P2_CONDITIONS
                        else "force_zeros"
                    )
                    if CONDITION_SIGMA_UM[condition] is not None
                    else None,
                    "claim_mask": "off for downstream diagnostic sort",
                },
                indent=2,
            )
            + "\n"
        )


def assert_gpu_and_patch() -> None:
    import torch
    from kilosort.parameters import DEFAULT_SETTINGS

    missing = {"cross_peel_claim_ms", "cross_peel_claim_um"} - set(DEFAULT_SETTINGS)
    if missing:
        raise RuntimeError(f"Patched Kilosort settings are missing: {sorted(missing)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this phase outside the sandbox")


def run_sorts() -> None:
    import spikeinterface.core as sc
    from spikeinterface.sorters import run_sorter

    assert_gpu_and_patch()
    params = build_sorter_params(CLAIM_OFF)
    for condition in CONDITIONS:
        target = sort_path(condition)
        if (target / "sorter_output/spike_times.npy").exists():
            print(f"Reusing completed sort {target}")
            continue
        if target.exists():
            raise RuntimeError(f"Partial or ambiguous sort directory: {target}")
        recording = sc.load(recording_path(condition))
        target.parent.mkdir(parents=True, exist_ok=True)
        log_path = OUTPUT_ROOT / "logs" / f"{condition}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            with log_path.open("a") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                run_sorter(
                    "kilosort4",
                    recording,
                    folder=str(target),
                    verbose=True,
                    remove_existing_folder=False,
                    **params,
                )
        finally:
            root_logger.removeHandler(handler)
            handler.close()


def score_sorts(review_path: Path, n_jitters: int, seed: int) -> pd.DataFrame:
    _, fs = load_reference_settings()
    events = events_in_window(pd.read_csv(review_path), WINDOW)
    populations = {
        "visual_neural_unmatched": (events.review_label == "neural") & (events.status == "unmatched"),
        "automatic_neural_like_unmatched": events.automatic_neural_like & (events.status == "unmatched"),
        "all_reviewed": pd.Series(True, index=events.index),
    }
    duration_samples = int(round(WINDOW.duration_s * fs))
    tolerance = int(round(0.5e-3 * fs))
    rows = []
    event_rows = []
    for condition in CONDITIONS:
        result = sort_path(condition) / "sorter_output"
        times = np.load(result / "spike_times.npy").reshape(-1).astype(np.int64)
        clusters = np.load(result / "spike_clusters.npy").reshape(-1)
        positions = np.load(result / "spike_positions.npy")
        valid = (times >= 0) & (times < duration_samples)
        times, clusters, depths = times[valid], clusters[valid], positions[valid, 1]
        full_count = len(np.load(result / "full_st.npy", mmap_mode="r"))
        labels = pd.read_csv(result / "cluster_KSLabel.tsv", sep="\t")
        label_column = next(column for column in labels if column != "cluster_id")
        contamination = pd.read_csv(result / "cluster_ContamPct.tsv", sep="\t")
        contam_column = next(column for column in contamination if column != "cluster_id")
        log_path = OUTPUT_ROOT / "logs" / f"{condition}.log"
        universal_count, learned_log_count = parse_extraction_counts(log_path.read_text())
        if learned_log_count is not None and learned_log_count != full_count:
            raise RuntimeError(f"Learned count mismatch for {condition}")
        all_samples = event_local_samples(events.sample_index.to_numpy(), WINDOW, fs)
        all_matches = local_match_mask(
            all_samples,
            events.peak_depth_um.to_numpy(float),
            times,
            depths,
            tolerance,
            100.0,
        )
        for (_, event), recovered in zip(events.iterrows(), all_matches):
            event_rows.append(
                {
                    "condition": condition,
                    "review_id": event.review_id,
                    "review_label": event.review_label,
                    "status": event.status,
                    "automatic_neural_like": bool(event.automatic_neural_like),
                    "time_seconds": float(event.time_seconds),
                    "peak_depth_um": float(event.peak_depth_um),
                    "recovered": bool(recovered),
                }
            )
        refractory = []
        isi_limit = int(round(1.5e-3 * fs))
        for unit in np.unique(clusters):
            unit_times = np.sort(times[clusters == unit])
            if len(unit_times) > 1:
                refractory.append(float(np.mean(np.diff(unit_times) < isi_limit)))
        for population, mask in populations.items():
            subset = events[mask]
            samples = event_local_samples(subset.sample_index.to_numpy(), WINDOW, fs)
            event_depths = subset.peak_depth_um.to_numpy(float)
            observed = float(local_match_mask(samples, event_depths, times, depths, tolerance, 100.0).mean())
            rng = np.random.default_rng(seed)
            null = []
            for _ in range(n_jitters):
                offsets_ms = rng.uniform(20.0, 500.0, len(subset)) * rng.choice((-1.0, 1.0), len(subset))
                shifted = samples + np.rint(offsets_ms * fs / 1000.0).astype(np.int64)
                null.append(float(local_match_mask(shifted, event_depths, times, depths, tolerance, 100.0).mean()))
            rows.append(
                {
                    "condition": condition,
                    "population": population,
                    "n_events": len(subset),
                    "observed_recovery": observed,
                    "jitter_null_mean": float(np.mean(null)),
                    "recovery_above_null": observed - float(np.mean(null)),
                    "universal_detection_count": universal_count,
                    "learned_detection_count": full_count,
                    "n_final_spikes": len(times),
                    "n_units": len(np.unique(clusters)),
                    "n_ks_good": int(labels[label_column].astype(str).str.lower().eq("good").sum()),
                    "median_contamination_pct": float(np.median(contamination[contam_column].to_numpy(float))),
                    "cross_unit_near_coincident_fraction": cross_unit_near_coincident_fraction(times, clusters, depths, tolerance),
                    "median_unit_refractory_violation_fraction": float(np.median(refractory)),
                }
            )
    frame = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_ROOT / "replication_scores.csv", index=False)
    pd.DataFrame(event_rows).to_csv(OUTPUT_ROOT / "paired_event_recovery.csv", index=False)
    return frame


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-motion-replication-numba")
    if not (args.prepare or args.run or args.score):
        raise SystemExit("Choose at least one of --prepare, --run, or --score")
    if args.prepare:
        prepare_recordings()
    if args.run:
        run_sorts()
    if args.score:
        print(score_sorts(args.review_events, args.n_jitters, args.seed).to_string(index=False))


if __name__ == "__main__":
    main()
