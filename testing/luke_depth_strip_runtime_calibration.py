"""Run a bounded Kilosort runtime calibration on the prepared Luke depth strip.

The default 600 s interval spans 7,800--8,400 s and includes the pathological
window and the large imec1 motion outlier.  It uses the exact no-motion baseline
settings planned for the full-duration strip but does not launch the full sort.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import build_sorter_params
from testing.luke_two_axis_pilot import CLAIM_OFF, OUTPUT_ROOT, assert_gpu_and_patch


RECORDING_DIR = OUTPUT_ROOT / "recordings/core_depth_strip"
OUTPUT_DIR = OUTPUT_ROOT / "runtime_calibration/core_depth_7800_600s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, default=RECORDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-s", type=float, default=7800.0)
    parser.add_argument("--duration-s", type=float, default=600.0)
    return parser.parse_args()


def calibration_frame_range(
    start_s: float, duration_s: float, fs: float, n_frames: int
) -> tuple[int, int]:
    if start_s < 0 or duration_s <= 0:
        raise ValueError("start must be nonnegative and duration must be positive")
    start = int(round(start_s * fs))
    stop = start + int(round(duration_s * fs))
    if start >= n_frames or stop > n_frames:
        raise ValueError("calibration interval exceeds the recording")
    return start, stop


def run_calibration(
    recording_dir: Path, output_dir: Path, start_s: float, duration_s: float
) -> dict:
    import spikeinterface.core as sc
    from spikeinterface.sorters import run_sorter

    assert_gpu_and_patch()
    recording = sc.load(recording_dir)
    fs = float(recording.get_sampling_frequency())
    start, stop = calibration_frame_range(
        start_s, duration_s, fs, recording.get_num_samples()
    )
    selected = recording.frame_slice(start_frame=start, end_frame=stop)
    target = output_dir / "single_ks_preprocessing_claim_off"
    result = target / "sorter_output/spike_times.npy"
    if result.exists():
        raise RuntimeError(f"Calibration already completed: {result}")
    if target.exists():
        raise RuntimeError(f"Ambiguous calibration output exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "calibration.log"
    params = build_sorter_params(CLAIM_OFF, bad_channels=None)
    handler = logging.FileHandler(log_path)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    started = time.perf_counter()
    try:
        with log_path.open("a") as log_file, contextlib.redirect_stdout(
            log_file
        ), contextlib.redirect_stderr(log_file):
            run_sorter(
                "kilosort4",
                selected,
                folder=str(target),
                verbose=True,
                remove_existing_folder=False,
                **params,
            )
    finally:
        root_logger.removeHandler(handler)
        handler.close()
    elapsed_s = time.perf_counter() - started
    spike_times = np.load(result, mmap_mode="r")
    clusters = np.load(
        target / "sorter_output/spike_clusters.npy", mmap_mode="r"
    )
    full_duration_s = recording.get_num_samples() / fs
    linear_full_runtime_s = elapsed_s * full_duration_s / duration_s
    summary = {
        "recording_dir": str(recording_dir),
        "output_dir": str(output_dir),
        "start_s": start / fs,
        "duration_s": (stop - start) / fs,
        "n_channels": recording.get_num_channels(),
        "elapsed_s": elapsed_s,
        "n_spikes": int(spike_times.shape[0]),
        "n_units": int(np.unique(clusters).size),
        "full_duration_s": full_duration_s,
        "linear_full_runtime_estimate_s": linear_full_runtime_s,
        "linear_full_runtime_estimate_hours": linear_full_runtime_s / 3600,
        "caveat": "Kilosort has fixed and data-dependent stages; linear scaling is a planning estimate, not a guarantee.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    run_calibration(
        args.recording_dir, args.output_dir, args.start_s, args.duration_s
    )


if __name__ == "__main__":
    main()
