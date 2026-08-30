"""Audit full-probe conditioning versus conditioning after a depth slice.

The full-duration Luke depth strip was built by selecting physical channels
176--271 before phase correction, 500 uV samplewise blanking, and interpolation
of physical channel 191.  This audit compares that path with conditioning all
384 AP channels first and slicing afterward.  Only short, prespecified windows
are evaluated; no recording or sort is materialized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing import luke_two_axis_pilot as pilot


OUTPUT_DIR = Path("testing/outputs/luke_conditioning_order_audit")
CHANNELS = np.arange(176, 272, dtype=int)
BAD_CHANNEL = 191
WINDOWS = {
    "good": 7095.0,
    "neutral": 7215.0,
    "pathological": 8160.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def comparison_metrics(
    full_then_slice: np.ndarray,
    slice_then_condition: np.ndarray,
    channel_ids: np.ndarray,
    window: str,
) -> list[dict]:
    """Return overall and channel-resolved difference metrics."""
    left = np.asarray(full_then_slice)
    right = np.asarray(slice_then_condition)
    ids = np.asarray(channel_ids, dtype=int)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Conditioning branches must be equal-shaped 2D arrays")
    if left.shape[1] != len(ids):
        raise ValueError("channel_ids do not match the trace matrices")
    delta = left.astype(np.float64) - right.astype(np.float64)

    rows: list[dict] = []
    groups = [
        ("all", np.ones(len(ids), dtype=bool)),
        ("excluding_191", ids != BAD_CHANNEL),
        ("channel_191", ids == BAD_CHANNEL),
    ]
    for group, keep in groups:
        a = left[:, keep].ravel().astype(np.float64)
        b = right[:, keep].ravel().astype(np.float64)
        d = delta[:, keep].ravel()
        if a.size == 0:
            continue
        if np.std(a) == 0 or np.std(b) == 0:
            correlation = float(a.size > 0 and np.array_equal(a, b))
        else:
            correlation = float(np.corrcoef(a, b)[0, 1])
        rows.append(
            {
                "window": window,
                "group": group,
                "n_values": int(a.size),
                "fraction_equal": float(np.mean(d == 0)),
                "mean_abs_difference_counts": float(np.mean(np.abs(d))),
                "p99_abs_difference_counts": float(
                    np.quantile(np.abs(d), 0.99)
                ),
                "max_abs_difference_counts": float(np.max(np.abs(d))),
                "rmse_counts": float(np.sqrt(np.mean(d**2))),
                "correlation": correlation,
                "full_min_counts": float(np.min(a)),
                "full_max_counts": float(np.max(a)),
                "slice_min_counts": float(np.min(b)),
                "slice_max_counts": float(np.max(b)),
            }
        )
    return rows


def run_audit(duration_s: float, output_dir: Path) -> dict:
    if duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    full_conditioned = pilot.load_source_recording(None, "legacy")
    full_then_slice = full_conditioned.channel_slice(channel_ids=CHANNELS)
    slice_then_condition = pilot.load_source_recording(CHANNELS, "legacy")
    fs = float(full_then_slice.get_sampling_frequency())
    n_frames = int(round(duration_s * fs))
    rows: list[dict] = []
    for name, start_s in WINDOWS.items():
        start = int(round(start_s * fs))
        stop = start + n_frames
        left = full_then_slice.get_traces(
            start_frame=start, end_frame=stop, return_scaled=False
        )
        right = slice_then_condition.get_traces(
            start_frame=start, end_frame=stop, return_scaled=False
        )
        rows.extend(comparison_metrics(left, right, CHANNELS, name))

    metrics = pd.DataFrame(rows)
    outside = metrics[metrics.group == "excluding_191"]
    bad = metrics[metrics.group == "channel_191"]
    # Phase correction and samplewise blanking should be exactly channel-local.
    # Only recomputed interpolation weights for channel 191 are allowed to differ.
    exact_outside_191 = bool((outside.fraction_equal == 1.0).all())
    channel_191_close = bool(
        (bad.correlation >= 0.9999).all()
        and (bad.p99_abs_difference_counts <= 1.0).all()
        and (bad.max_abs_difference_counts <= 3.0).all()
    )
    passed = exact_outside_191 and channel_191_close
    decision = {
        "audit": "full_384_then_slice_vs_slice_then_condition",
        "windows": WINDOWS,
        "duration_s_per_window": duration_s,
        "channel_first": int(CHANNELS[0]),
        "channel_last": int(CHANNELS[-1]),
        "n_channels": int(len(CHANNELS)),
        "physical_bad_channel": BAD_CHANNEL,
        "criteria": {
            "outside_channel_191_fraction_equal": 1.0,
            "channel_191_minimum_correlation": 0.9999,
            "channel_191_maximum_p99_abs_difference_counts": 1.0,
            "channel_191_maximum_absolute_difference_counts": 3.0,
        },
        "exact_outside_channel_191": exact_outside_191,
        "channel_191_close": channel_191_close,
        "passed": passed,
        "interpretation": (
            "Depth-first conditioning is equivalent within the prespecified gate."
            if passed
            else "Depth-first conditioning is not equivalent; rebuild or revise the gate before sorting."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "conditioning_order_metrics.csv", index=False)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    return decision


def main() -> None:
    args = parse_args()
    decision = run_audit(args.duration_s, args.output_dir)
    print(json.dumps(decision, indent=2))
    if not decision["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
