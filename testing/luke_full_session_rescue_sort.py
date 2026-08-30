"""Materialize, sort, and score Luke imec1's full-session rescue candidate."""

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
    build_sorter_params,
    cross_unit_near_coincident_fraction,
    load_reference_settings,
    local_match_mask,
)
from testing.luke_upstream_sorter_ablation import parse_extraction_counts
from testing.luke_upstream_stage_ablation import (
    DEFAULT_REVIEW,
    PIPELINE_ROOT,
    RAW_ROOT,
    STREAM_ID,
    build_recording_stages,
)


OUTPUT_ROOT = PIPELINE_ROOT / "full_session_rescue/single_ks_preprocessing_claim_off"
RECORDING_PATH = OUTPUT_ROOT / "recording"
SORT_PATH = OUTPUT_ROOT / "sort"
CLAIM_OFF = ClaimSetting("claim_off", 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--n-jitters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20250804)
    return parser.parse_args()


def prepare_recording() -> None:
    import spikeinterface.extractors as se

    metadata = RECORDING_PATH / "full_session_manifest.json"
    if RECORDING_PATH.exists():
        if not metadata.exists():
            raise RuntimeError(f"Ambiguous partial recording: {RECORDING_PATH}")
        print(f"Reusing {RECORDING_PATH}")
        return
    raw = se.read_spikeglx(
        folder_path=RAW_ROOT, load_sync_channel=False, stream_id=STREAM_ID
    )
    stages, bad_ids, gain = build_recording_stages(raw)
    recording = stages["interpolated_unfiltered"]
    expected_bytes = (
        recording.get_num_samples() * recording.get_num_channels() * np.dtype("int16").itemsize
    )
    RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    recording.save(
        folder=RECORDING_PATH,
        n_jobs=20,
        chunk_duration="1s",
        progress_bar=True,
    )
    metadata.write_text(
        json.dumps(
            {
                "condition": "single_ks_preprocessing_claim_off",
                "scope": "full_session",
                "raw_root": str(RAW_ROOT),
                "stream_id": STREAM_ID,
                "source_stage": "phase_shift_saturation_blank_bad_channel_interpolation",
                "external_filter": None,
                "external_reference": None,
                "external_voltage_motion_correction": False,
                "bad_channel_ids": [str(value) for value in bad_ids],
                "gain_uv_per_bit": gain,
                "num_samples": recording.get_num_samples(),
                "num_channels": recording.get_num_channels(),
                "sampling_frequency_hz": recording.get_sampling_frequency(),
                "dtype": str(recording.dtype),
                "expected_binary_bytes": expected_bytes,
                "claim_mask": "off for downstream Kilosort run",
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


def run_sort() -> None:
    import spikeinterface.core as sc
    from spikeinterface.sorters import run_sorter

    assert_gpu_and_patch()
    result = SORT_PATH / "sorter_output/spike_times.npy"
    if result.exists():
        print(f"Reusing completed sort {SORT_PATH}")
        return
    if SORT_PATH.exists():
        raise RuntimeError(f"Partial or ambiguous sort: {SORT_PATH}")
    if not (RECORDING_PATH / "full_session_manifest.json").exists():
        raise FileNotFoundError(f"Prepare the full recording first: {RECORDING_PATH}")
    recording = sc.load(RECORDING_PATH)
    params = build_sorter_params(CLAIM_OFF)
    SORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / "kilosort.log"
    handler = logging.FileHandler(log_path)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        with log_path.open("a") as log_file, contextlib.redirect_stdout(
            log_file
        ), contextlib.redirect_stderr(log_file):
            run_sorter(
                "kilosort4",
                recording,
                folder=str(SORT_PATH),
                verbose=True,
                remove_existing_folder=False,
                **params,
            )
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def score_sort(review_path: Path, n_jitters: int, seed: int) -> pd.DataFrame:
    result = SORT_PATH / "sorter_output"
    if not (result / "spike_times.npy").exists():
        raise FileNotFoundError(result)
    _, fs = load_reference_settings()
    events = pd.read_csv(review_path)
    populations = {
        "visual_neural_unmatched": (events.review_label == "neural")
        & (events.status == "unmatched"),
        "visual_neural_all": events.review_label == "neural",
        "all_reviewed": pd.Series(True, index=events.index),
    }
    times = np.load(result / "spike_times.npy").reshape(-1).astype(np.int64)
    clusters = np.load(result / "spike_clusters.npy").reshape(-1)
    depths = np.load(result / "spike_positions.npy")[:, 1]
    tolerance = int(round(0.5e-3 * fs))
    duration_samples = int(round(10473.553879363088 * fs))
    valid = (times >= 0) & (times < duration_samples)
    times, clusters, depths = times[valid], clusters[valid], depths[valid]
    full_count = len(np.load(result / "full_st.npy", mmap_mode="r"))
    labels = pd.read_csv(result / "cluster_KSLabel.tsv", sep="\t")
    label_column = next(column for column in labels if column != "cluster_id")
    contamination = pd.read_csv(result / "cluster_ContamPct.tsv", sep="\t")
    contam_column = next(column for column in contamination if column != "cluster_id")
    log_path = OUTPUT_ROOT / "kilosort.log"
    universal_count, learned_log_count = parse_extraction_counts(log_path.read_text())
    if learned_log_count is not None and learned_log_count != full_count:
        raise RuntimeError(
            f"Learned count mismatch: log={learned_log_count}, full_st={full_count}"
        )
    all_matches = local_match_mask(
        events.sample_index.to_numpy(np.int64),
        events.peak_depth_um.to_numpy(float),
        times,
        depths,
        tolerance,
        100.0,
    )
    event_recovery = events[
        ["review_id", "window", "status", "review_label", "sample_index", "peak_depth_um"]
    ].copy()
    event_recovery["recovered"] = all_matches
    event_recovery.to_csv(OUTPUT_ROOT / "paired_event_recovery.csv", index=False)

    refractory = []
    isi_limit = int(round(1.5e-3 * fs))
    order = np.argsort(clusters, kind="stable")
    sorted_clusters = clusters[order]
    sorted_times = times[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_clusters)) + 1, len(sorted_clusters)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        unit_times = np.sort(sorted_times[start:stop])
        if len(unit_times) > 1:
            refractory.append(float(np.mean(np.diff(unit_times) < isi_limit)))

    rng = np.random.default_rng(seed)
    rows = []
    for population, mask in populations.items():
        subset = events.loc[mask]
        samples = subset.sample_index.to_numpy(np.int64)
        event_depths = subset.peak_depth_um.to_numpy(float)
        observed = float(
            local_match_mask(samples, event_depths, times, depths, tolerance, 100.0).mean()
        )
        null = []
        for _ in range(n_jitters):
            offsets_ms = rng.uniform(20.0, 500.0, len(subset))
            offsets_ms *= rng.choice((-1.0, 1.0), len(subset))
            shifted = samples + np.rint(offsets_ms * fs / 1000.0).astype(np.int64)
            null.append(
                float(
                    local_match_mask(
                        shifted, event_depths, times, depths, tolerance, 100.0
                    ).mean()
                )
            )
        rows.append(
            {
                "condition": "single_ks_preprocessing_claim_off",
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
                "median_contamination_pct": float(
                    np.median(contamination[contam_column].to_numpy(float))
                ),
                "cross_unit_near_coincident_fraction": cross_unit_near_coincident_fraction(
                    times, clusters, depths, tolerance
                ),
                "median_unit_refractory_violation_fraction": float(np.median(refractory)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_ROOT / "full_session_scores.csv", index=False)
    return frame


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-full-session-rescue-numba")
    if not (args.prepare or args.run or args.score):
        raise SystemExit("Choose at least one of --prepare, --run, or --score")
    if args.prepare:
        prepare_recording()
    if args.run:
        run_sort()
    if args.score:
        print(score_sort(args.review_events, args.n_jitters, args.seed).to_string(index=False))


if __name__ == "__main__":
    main()
