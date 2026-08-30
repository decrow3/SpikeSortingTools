"""Audit Kilosort's batch-zeroing artifact threshold on hard Luke snippets.

The candidate recording is phase-shifted, otherwise unchanged, and has channel
191 interpolated. For each Kilosort-sized batch, reproduce CAR and FFT
high-pass filtering, then report the maximum absolute value and which reviewed
events would be erased by candidate artifact thresholds. No sorting is run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_two_axis_pilot import DEFAULT_REVIEW, PILOTS


V2_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_conditioning_v2_pilot_imec1"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_kilosort_artifact_threshold_audit")
WINDOWS = ("neutral_template", "pathological")
THRESHOLDS_COUNTS = (100, 150, 200, 250, 300, 400, 500, 750, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=60000)
    parser.add_argument("--padding", type=int, default=61)
    return parser.parse_args()


def padded_traces(recording, start: int, stop: int, padding: int) -> np.ndarray:
    n = recording.get_num_samples()
    read_start = max(0, start - padding)
    read_stop = min(n, stop + padding)
    values = recording.get_traces(start_frame=read_start, end_frame=read_stop)
    before = max(0, padding - start)
    after = max(0, stop + padding - n)
    if before or after:
        values = np.pad(values, ((before, after), (0, 0)), mode="edge")
    return np.asarray(values, dtype=np.float32)


def ks_car_highpass_max_abs(
    values: np.ndarray, fs: float, device: torch.device
) -> float:
    from kilosort.io import fft_highpass
    from kilosort.preprocessing import get_highpass_filter

    x = torch.as_tensor(values.T, dtype=torch.float32, device=device)
    x = x - x.mean(dim=1, keepdim=True)
    x = x - torch.median(x, dim=0, keepdim=True).values
    hp = get_highpass_filter(fs=fs, cutoff=300, device=device)
    fwav = fft_highpass(hp, NT=x.shape[1])
    x = torch.real(torch.fft.ifft(torch.fft.fft(x) * torch.conj(fwav)))
    x = torch.fft.fftshift(x, dim=-1)
    return float(torch.max(torch.abs(x)).item())


def threshold_summary(
    batches: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for window in WINDOWS:
        local_batches = batches[batches.window == window]
        local_events = events[events.window == window]
        for threshold in THRESHOLDS_COUNTS:
            rejected = local_batches.max_abs_post_car_highpass >= threshold
            rejected_ids = set(local_batches.loc[rejected, "batch_index"])
            erased = local_events.batch_index.isin(rejected_ids)
            neural = local_events.review_label.eq("neural")
            rows.append(
                {
                    "window": window,
                    "artifact_threshold_counts": threshold,
                    "rejected_batches": int(rejected.sum()),
                    "rejected_batch_fraction": float(rejected.mean()),
                    "reviewed_events_erased": int(erased.sum()),
                    "reviewed_neural_events_erased": int((erased & neural).sum()),
                    "reviewed_neural_erased_fraction": float(
                        (erased & neural).sum() / max(1, neural.sum())
                    ),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    import spikeinterface.core as sc
    from spikeinterface.preprocessing import interpolate_bad_channels

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed = pd.read_csv(args.review_events)
    batch_rows = []
    event_rows = []
    for window in WINDOWS:
        pilot = PILOTS[window]
        recording = sc.load(V2_ROOT / "recordings" / window)
        recording = interpolate_bad_channels(recording, bad_channel_ids=[191])
        fs = float(recording.get_sampling_frequency())
        global_start = int(round(float(pilot.start_s) * fs))
        n = recording.get_num_samples()
        for batch, start in enumerate(range(0, n, args.batch_size)):
            stop = min(n, start + args.batch_size)
            values = padded_traces(recording, start, stop, args.padding)
            batch_rows.append(
                {
                    "window": window,
                    "batch_index": batch,
                    "start_s_local": start / fs,
                    "stop_s_local": stop / fs,
                    "max_abs_post_car_highpass": ks_car_highpass_max_abs(
                        values, fs, device
                    ),
                }
            )
        selected = reviewed[
            reviewed.sample_index.between(global_start, global_start + n - 1)
        ]
        for _, event in selected.iterrows():
            local_sample = int(event.sample_index) - global_start
            event_rows.append(
                {
                    "window": window,
                    "review_id": event.review_id,
                    "review_label": event.review_label,
                    "status": event.status,
                    "batch_index": local_sample // args.batch_size,
                }
            )
    batches = pd.DataFrame(batch_rows)
    events = pd.DataFrame(event_rows)
    summary = threshold_summary(batches, events)
    batches.to_csv(args.output_dir / "batch_maxima.csv", index=False)
    events.to_csv(args.output_dir / "reviewed_event_batches.csv", index=False)
    summary.to_csv(args.output_dir / "threshold_summary.csv", index=False)
    manifest = {
        "sorting_run": False,
        "motion_correction": False,
        "claim_mask": "off",
        "device": str(device),
        "batch_size": args.batch_size,
        "padding": args.padding,
        "threshold_semantics": (
            "Kilosort zeros the complete padded batch if any post-CAR/high-pass "
            "absolute value reaches the threshold"
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run(parse_args())
