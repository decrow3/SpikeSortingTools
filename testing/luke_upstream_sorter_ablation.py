"""Expose upstream Luke failures with claim-off Kilosort on the worst window.

The existing registration-outlier/claim-off sort is the current-pipeline
baseline.  This runner adds three one-factor controls on the identical 120 s
frame range: no external motion, float-preserving conditioning, and both.
Claim masking is disabled intentionally so the learned-template expansion is a
readout of the upstream input rather than a proposed production setting.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
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
from testing.luke_upstream_stage_ablation import (
    DEFAULT_REVIEW,
    RAW_ROOT,
    STREAM_ID,
    build_recording_stages,
)


OUTPUT_ROOT = PIPELINE_ROOT / "upstream_sorter_ablation"
CURRENT_RECORDING = PIPELINE_ROOT / "claimmask_window_sweep/recordings/registration_outlier"
CURRENT_SORT = PIPELINE_ROOT / "claimmask_window_sweep/sorts/registration_outlier/claim_off"
WINDOW = next(window for window in WINDOWS if window.name == "registration_outlier")
CLAIM_OFF = ClaimSetting("claim_off", 0.0, 0.0)


@dataclass(frozen=True)
class Condition:
    name: str
    stage: str | None
    do_ks_car: bool = True


CONDITIONS = (
    Condition("current_motion", None),
    Condition("current_no_motion", "current_conditioned"),
    Condition("float_no_motion", "float_conditioned_control"),
    Condition("float_motion", "float_motion_corrected_control"),
    Condition("bandpass_no_reference", "interpolated_bandpass"),
    Condition("global_reference", "global_reference_control"),
    Condition("local_reference_no_ks_car", "current_conditioned", False),
    Condition("single_ks_preprocessing", "interpolated_unfiltered"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--n-jitters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20250804)
    return parser.parse_args()


def condition_recording_dir(output_dir: Path, condition: Condition) -> Path:
    if condition.name == "current_motion":
        return CURRENT_RECORDING
    return output_dir / "recordings" / condition.name


def condition_sort_dir(output_dir: Path, condition: Condition) -> Path:
    if condition.name == "current_motion":
        return CURRENT_SORT
    return output_dir / "sorts" / condition.name


def prepare_recordings(output_dir: Path) -> None:
    import spikeinterface.extractors as se

    raw = se.read_spikeglx(
        folder_path=RAW_ROOT, load_sync_channel=False, stream_id=STREAM_ID
    )
    stages, _, _ = build_recording_stages(raw)
    fs = float(raw.get_sampling_frequency())
    start, stop = window_frames(WINDOW, fs)
    for condition in CONDITIONS:
        if condition.stage is None:
            if not CURRENT_RECORDING.exists():
                raise FileNotFoundError(CURRENT_RECORDING)
            continue
        target = condition_recording_dir(output_dir, condition)
        metadata = target / "window_manifest.json"
        if target.exists():
            if not metadata.exists():
                raise RuntimeError(f"Ambiguous existing recording directory: {target}")
            print(f"Reusing {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        sliced = stages[condition.stage].frame_slice(start_frame=start, end_frame=stop)
        sliced.save(folder=target, n_jobs=-1, chunk_duration="1s", progress_bar=True)
        metadata.write_text(
            json.dumps(
                {
                    "condition": condition.name,
                    "source_stage": condition.stage,
                    "window": WINDOW.name,
                    "start_sample": start,
                    "stop_sample": stop,
                    "sampling_frequency_hz": fs,
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


def run_sorts(output_dir: Path) -> None:
    import spikeinterface.core as sc
    from spikeinterface.sorters import run_sorter

    assert_gpu_and_patch()
    for condition in CONDITIONS:
        params = build_sorter_params(CLAIM_OFF)
        params["do_CAR"] = condition.do_ks_car
        sort_dir = condition_sort_dir(output_dir, condition)
        result = sort_dir / "sorter_output/spike_times.npy"
        if result.exists():
            print(f"Reusing completed sort {sort_dir}")
            continue
        if condition.name == "current_motion":
            raise FileNotFoundError(result)
        if sort_dir.exists():
            raise RuntimeError(f"Partial or ambiguous sort directory: {sort_dir}")
        recording_dir = condition_recording_dir(output_dir, condition)
        if not recording_dir.exists():
            raise FileNotFoundError(f"Prepare the recording first: {recording_dir}")
        recording = sc.load(recording_dir)
        sort_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "logs" / f"{condition.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("[%(levelname)s] - %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        print(
            f"Running {condition.name} with claim masking disabled and "
            f"Kilosort CAR={condition.do_ks_car}",
            flush=True,
        )
        try:
            with log_path.open("a") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                run_sorter(
                    "kilosort4",
                    recording,
                    folder=str(sort_dir),
                    verbose=True,
                    remove_existing_folder=False,
                    **params,
                )
        finally:
            root_logger.removeHandler(handler)
            handler.close()
        print(f"Completed {condition.name}; log: {log_path}", flush=True)


def parse_extraction_counts(log_text: str) -> tuple[int | None, int | None]:
    counts = [int(value) for value in re.findall(r"(\d+) spikes extracted", log_text)]
    if len(counts) < 2:
        return None, None
    return counts[-2], counts[-1]


def _population_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "visual_neural_unmatched": (events["review_label"] == "neural")
        & (events["status"] == "unmatched"),
        "automatic_neural_like_unmatched": events["automatic_neural_like"]
        & (events["status"] == "unmatched"),
        "all_reviewed": pd.Series(True, index=events.index),
    }


def score_sorts(output_dir: Path, review_path: Path, n_jitters: int, seed: int) -> pd.DataFrame:
    _, fs = load_reference_settings()
    events = events_in_window(pd.read_csv(review_path), WINDOW)
    rng = np.random.default_rng(seed)
    tolerance = int(round(0.5e-3 * fs))
    duration_samples = int(round(WINDOW.duration_s * fs))
    rows = []
    for condition in CONDITIONS:
        result_dir = condition_sort_dir(output_dir, condition) / "sorter_output"
        if not (result_dir / "spike_times.npy").exists():
            continue
        times = np.load(result_dir / "spike_times.npy").reshape(-1).astype(np.int64)
        clusters = np.load(result_dir / "spike_clusters.npy").reshape(-1)
        positions = np.load(result_dir / "spike_positions.npy")
        valid = (times >= 0) & (times < duration_samples)
        times, clusters, depths = times[valid], clusters[valid], positions[valid, 1]
        full_count = len(np.load(result_dir / "full_st.npy", mmap_mode="r"))
        labels = pd.read_csv(result_dir / "cluster_KSLabel.tsv", sep="\t")
        label_column = next(column for column in labels if column != "cluster_id")
        contamination = pd.read_csv(result_dir / "cluster_ContamPct.tsv", sep="\t")
        contam_column = next(column for column in contamination if column != "cluster_id")
        log_path = output_dir / "logs" / f"{condition.name}.log"
        universal_count = learned_log_count = None
        if log_path.exists():
            universal_count, learned_log_count = parse_extraction_counts(log_path.read_text())
        if learned_log_count is not None and learned_log_count != full_count:
            raise RuntimeError(
                f"Learned count mismatch for {condition.name}: log={learned_log_count}, full_st={full_count}"
            )
        duplicate_fraction = cross_unit_near_coincident_fraction(
            times, clusters, depths, tolerance
        )
        refractory = []
        isi_limit = int(round(1.5e-3 * fs))
        for unit in np.unique(clusters):
            unit_times = np.sort(times[clusters == unit])
            if len(unit_times) > 1:
                refractory.append(float(np.mean(np.diff(unit_times) < isi_limit)))
        for population, mask in _population_masks(events).items():
            subset = events[mask]
            samples = event_local_samples(subset["sample_index"].to_numpy(), WINDOW, fs)
            event_depths = subset["peak_depth_um"].to_numpy(float)
            observed = float(
                local_match_mask(samples, event_depths, times, depths, tolerance, 100.0).mean()
            ) if len(subset) else np.nan
            null = []
            if len(subset):
                for _ in range(n_jitters):
                    offsets_ms = rng.uniform(20.0, 500.0, len(subset))
                    offsets_ms *= rng.choice((-1.0, 1.0), len(subset))
                    shifted = samples + np.rint(offsets_ms * fs / 1000.0).astype(np.int64)
                    null.append(float(local_match_mask(shifted, event_depths, times, depths, tolerance, 100.0).mean()))
            rows.append(
                {
                    "condition": condition.name,
                    "population": population,
                    "n_events": len(subset),
                    "observed_recovery": observed,
                    "jitter_null_mean": float(np.mean(null)) if null else np.nan,
                    "recovery_above_null": observed - float(np.mean(null)) if null else np.nan,
                    "universal_detection_count": universal_count,
                    "learned_detection_count": full_count,
                    "learned_to_universal_expansion": (
                        full_count / universal_count if universal_count else np.nan
                    ),
                    "n_final_spikes": len(times),
                    "n_units": len(np.unique(clusters)),
                    "n_ks_good": int(labels[label_column].astype(str).str.lower().eq("good").sum()),
                    "median_contamination_pct": float(np.median(contamination[contam_column].to_numpy(float))),
                    "cross_unit_near_coincident_fraction": duplicate_fraction,
                    "median_unit_refractory_violation_fraction": float(np.median(refractory)),
                }
            )
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "upstream_sorter_ablation_scores.csv", index=False)
    return result


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-upstream-sorter-numba-cache")
    if not (args.prepare or args.run or args.score):
        raise SystemExit("Choose at least one of --prepare, --run, or --score")
    if args.prepare:
        prepare_recordings(args.output_dir)
    if args.run:
        run_sorts(args.output_dir)
    if args.score:
        score_sorts(args.output_dir, args.review_events, args.n_jitters, args.seed)


if __name__ == "__main__":
    main()
