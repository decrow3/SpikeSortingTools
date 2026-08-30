"""Snippet-scale saturation-policy audit for Luke imec1.

This is deliberately motion-free and sorter-free. It compares the current
samplewise pre-filter replacement against interval interpolation and
post-filter blanking, using the same fixed ordinary/saturation-enriched batches
and reviewed event-centered neighborhoods as the conditioning audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing import luke_conditioning_stage_audit as conditioning
from testing.luke_upstream_stage_ablation import event_metrics


DEFAULT_OUTPUT = Path("testing/outputs/luke_saturation_policy_audit")
INTERPOLATION_MARGINS_MS = (0.25, 0.5, 1.0)
POSTFILTER_MARGINS_MS = (0.25, 0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-events", type=Path, default=conditioning.DEFAULT_REVIEW)
    parser.add_argument("--batch-duration-s", type=float, default=1.0)
    parser.add_argument("--batches-per-window", type=int, default=1)
    parser.add_argument("--max-events-per-window", type=int, default=10)
    parser.add_argument(
        "--saturation-index", type=Path, default=conditioning.DEFAULT_SATURATION_INDEX
    )
    parser.add_argument("--saturation-batches", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def interpolate_masked_intervals(
    values: np.ndarray, mask: np.ndarray, radius_samples: int
) -> np.ndarray:
    """Linearly bridge complete masked intervals independently per channel."""
    result = np.asarray(values, dtype=np.float32).copy()
    expanded = conditioning.dilate_time_mask(mask, radius_samples)
    sample_index = np.arange(result.shape[0])
    for channel in range(result.shape[1]):
        bad = expanded[:, channel]
        if not np.any(bad):
            continue
        good = ~bad
        if not np.any(good):
            result[:, channel] = np.median(result[:, channel])
            continue
        result[bad, channel] = np.interp(
            sample_index[bad], sample_index[good], result[good, channel]
        )
    return result


def saturation_stages(
    raw_values: np.ndarray,
    phase_values: np.ndarray,
    fs: float,
    threshold_counts: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    # Phase shifting can move/smear an extreme sample by one sample, so use the
    # union for conservative artifact support rather than trusting either alone.
    saturation_mask = (np.abs(raw_values) >= threshold_counts) | (
        np.abs(phase_values) >= threshold_counts
    )
    baseline = conditioning.ks_highpass(conditioning.ks_center_car(phase_values), fs)
    stages = {"no_saturation_replacement": baseline}
    legacy = conditioning.clip_saturation(phase_values, threshold_counts)
    stages["legacy_point_blank_prefilter"] = conditioning.ks_highpass(
        conditioning.ks_center_car(legacy), fs
    )
    for margin_ms in INTERPOLATION_MARGINS_MS:
        radius = int(round(margin_ms * 1e-3 * fs))
        repaired = interpolate_masked_intervals(phase_values, saturation_mask, radius)
        stages[f"linear_interval_prefilter_{margin_ms:g}ms"] = conditioning.ks_highpass(
            conditioning.ks_center_car(repaired), fs
        )
    for margin_ms in POSTFILTER_MARGINS_MS:
        radius = int(round(margin_ms * 1e-3 * fs))
        stages[f"postfilter_blank_{margin_ms:g}ms"] = conditioning.apply_postfilter_blank(
            baseline, saturation_mask, radius
        )
    return stages, saturation_mask


def selected_batches(args: argparse.Namespace, fs: float):
    duration = min(args.batch_duration_s, 0.25) if args.smoke else args.batch_duration_s
    count = 1 if args.smoke else args.batches_per_window
    for window in conditioning.WINDOWS:
        starts = conditioning.choose_batch_starts(window, fs, duration, count)
        for batch, start in enumerate(starts):
            yield window.name, batch, int(start), duration
    if not args.smoke:
        for batch, (kind, start_s) in enumerate(
            conditioning.saturation_enriched_starts(
                args.saturation_index, args.saturation_batches
            )
        ):
            yield kind, batch, int(round(start_s * fs)), duration


def run(args: argparse.Namespace) -> None:
    raw, _, phase_float, fs, gain, _ = conditioning.load_raw_recordings()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions = raw.get_channel_locations().astype(float)
    threshold_counts = conditioning.SATURATION_UV / gain
    rows: list[dict] = []
    for window, batch, start, duration in selected_batches(args, fs):
        stop = start + int(round(duration * fs))
        raw_values = raw.get_traces(start_frame=start, end_frame=stop)
        phase_values = phase_float.get_traces(start_frame=start, end_frame=stop)
        stages, mask = saturation_stages(raw_values, phase_values, fs, threshold_counts)
        for stage, values in stages.items():
            rows.append(
                {
                    "window": window,
                    "batch": batch,
                    "start_s": start / fs,
                    "stage": stage,
                    "saturation_sample_channel_fraction": float(np.mean(mask)),
                    **conditioning.batch_metrics(values, mask, positions, fs),
                }
            )
    batch_frame = pd.DataFrame(rows)
    batch_frame.to_csv(args.output_dir / "batch_metrics.csv", index=False)
    conditioning.summarize(batch_frame).to_csv(
        args.output_dir / "stage_summary.csv", index=False
    )

    events = pd.read_csv(args.review_events)
    event_rows: list[dict] = []
    half = int(round(10e-3 * fs))
    max_events = 2 if args.smoke else args.max_events_per_window
    depths = positions[:, 1]
    for window in conditioning.WINDOWS:
        lo = int(round(window.start_s * fs))
        hi = int(round((window.start_s + window.duration_s) * fs))
        selected = events[events["sample_index"].between(lo + half, hi - half - 1)]
        if len(selected) > max_events:
            selected = selected.iloc[
                np.unique(np.linspace(0, len(selected) - 1, max_events, dtype=int))
            ]
        for _, event in selected.iterrows():
            center = int(event["sample_index"])
            raw_values = raw.get_traces(
                start_frame=center - half, end_frame=center + half + 1
            )
            phase_values = phase_float.get_traces(
                start_frame=center - half, end_frame=center + half + 1
            )
            stages, mask = saturation_stages(raw_values, phase_values, fs, threshold_counts)
            channels = conditioning.review_channel_indices(event, depths)
            local_mask = mask[:, channels]
            for stage, values in stages.items():
                metrics, _ = event_metrics(values[:, channels], depths[channels], fs)
                local_peak = int(metrics["peak_channel"])
                metrics["peak_channel"] = int(channels[local_peak])
                event_rows.append(
                    {
                        "window": window.name,
                        "review_id": event["review_id"],
                        "review_label": event["review_label"],
                        "status": event["status"],
                        "reference_depth_um": float(event["peak_depth_um"]),
                        "saturation_in_event_neighborhood": bool(np.any(local_mask)),
                        "saturation_fraction_in_event_neighborhood": float(
                            np.mean(local_mask)
                        ),
                        "stage": stage,
                        **metrics,
                    }
                )
    pd.DataFrame(event_rows).to_csv(
        args.output_dir / "reviewed_event_metrics.csv", index=False
    )
    print(conditioning.summarize(batch_frame).to_string(index=False))


if __name__ == "__main__":
    run(parse_args())
