"""Localize Luke imec1's reference-controlled positive and negative events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_yates_raw_voltage_audit import (
    collapse_candidates,
    extrema_candidates,
    load_specs,
    local_median_reference,
    nearest_neighbors,
    read_batch,
    select_batch_starts,
    shank_median_reference,
    spatial_neighbors,
)


DEFAULT_OUTPUT = Path("testing/outputs/luke_imec1_event_localization_audit")
STAGES = (
    "common_bandpass_equal_5_reference",
    "common_bandpass_local_reference",
    "common_bandpass_shank_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-duration-s", type=float, default=2.0)
    parser.add_argument("--padding-s", type=float, default=0.1)
    parser.add_argument("--threshold-uv", type=float, default=75.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Recompute summaries from output-dir/channel_event_batch_metrics.csv",
    )
    return parser.parse_args()


def effective_channel_count(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() <= 0:
        return 0.0
    probabilities = counts[counts > 0] / counts.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def channels_for_fraction(counts: np.ndarray, fraction: float) -> int:
    counts = np.sort(np.asarray(counts, dtype=float))[::-1]
    if counts.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(counts), fraction * counts.sum()) + 1)


def summarize_channel_counts(
    channel_rows: pd.DataFrame, target_channel: int = 216
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    group_keys = ["dataset", "window_kind", "stage", "polarity"]
    channel_summary = (
        channel_rows.groupby(group_keys + ["channel", "y_um"], as_index=False)
        .agg(event_count=("event_count", "sum"), sampled_duration_s=("duration_s", "sum"))
    )
    channel_summary["event_rate_per_s"] = (
        channel_summary["event_count"] / channel_summary["sampled_duration_s"]
    )

    localization_rows: list[dict] = []
    for keys, cohort in channel_summary.groupby(group_keys, sort=True):
        cohort = cohort.sort_values("channel").reset_index(drop=True)
        counts = cohort["event_count"].to_numpy(dtype=float)
        total = float(counts.sum())
        target = cohort.loc[cohort["channel"] == target_channel]
        if len(target) != 1:
            raise ValueError(f"Expected channel {target_channel} once in {keys}")
        target_count = float(target["event_count"].iloc[0])
        target_depth = float(target["y_um"].iloc[0])
        near = np.abs(cohort["y_um"].to_numpy() - target_depth) <= 100.0
        top_n = max(1, int(np.ceil(0.05 * len(cohort))))
        top_fraction = float(np.sort(counts)[-top_n:].sum() / total) if total else 0.0
        localization_rows.append(
            {
                **dict(zip(group_keys, keys)),
                "total_events": int(total),
                "total_event_rate_per_s": total
                / float(cohort["sampled_duration_s"].iloc[0]),
                "target_channel": target_channel,
                "target_event_count": int(target_count),
                "target_event_fraction": target_count / total if total else 0.0,
                "target_count_percentile": float(np.mean(counts <= target_count)),
                "within_100um_target_fraction": float(counts[near].sum() / total)
                if total
                else 0.0,
                "top_5pct_channel_fraction": top_fraction,
                "effective_channel_count": effective_channel_count(counts),
                "channels_for_50pct_events": channels_for_fraction(counts, 0.5),
            }
        )
    localization = pd.DataFrame(localization_rows)

    correlations: list[dict] = []
    positive = channel_summary.loc[channel_summary["polarity"] == "positive"]
    for stage, stage_rows in positive.groupby("stage"):
        profiles = {
            window: group.sort_values("channel")["event_rate_per_s"].to_numpy()
            for window, group in stage_rows.groupby("window_kind")
        }
        for first, second in (
            ("pathological", "shared"),
            ("pathological", "session-wide"),
            ("shared", "session-wide"),
        ):
            if first in profiles and second in profiles:
                correlation = float(spearmanr(profiles[first], profiles[second]).statistic)
                correlations.append(
                    {
                        "stage": stage,
                        "first_window": first,
                        "second_window": second,
                        "spearman_channel_burden": correlation,
                    }
                )
    correlation_table = pd.DataFrame(correlations)

    positive_localization = localization.loc[localization["polarity"] == "positive"]
    row216_plausible = bool(
        positive_localization["target_event_fraction"].max() >= 0.05
        or positive_localization["within_100um_target_fraction"].max() >= 0.20
    )
    concentrated_elsewhere = bool(
        positive_localization["effective_channel_count"].min() <= 50
        or positive_localization["top_5pct_channel_fraction"].max() >= 0.50
    )
    if row216_plausible:
        decision_name = "row216_remains_plausible_driver"
    elif concentrated_elsewhere:
        decision_name = (
            "row216_not_driver_positive_excess_concentrated_in_stable_multichannel_bands"
        )
    else:
        decision_name = "row216_not_dominant_positive_excess_is_distributed"
    decision = {
        "decision": decision_name,
        "positive_event_localization": {
            "row216_event_fraction_range": [
                float(positive_localization["target_event_fraction"].min()),
                float(positive_localization["target_event_fraction"].max()),
            ],
            "within_100um_row216_fraction_range": [
                float(positive_localization["within_100um_target_fraction"].min()),
                float(positive_localization["within_100um_target_fraction"].max()),
            ],
            "row216_count_percentile_range": [
                float(positive_localization["target_count_percentile"].min()),
                float(positive_localization["target_count_percentile"].max()),
            ],
            "top_5pct_channel_fraction_range": [
                float(positive_localization["top_5pct_channel_fraction"].min()),
                float(positive_localization["top_5pct_channel_fraction"].max()),
            ],
            "effective_channel_count_range": [
                float(positive_localization["effective_channel_count"].min()),
                float(positive_localization["effective_channel_count"].max()),
            ],
            "channels_for_half_events_range": [
                int(positive_localization["channels_for_50pct_events"].min()),
                int(positive_localization["channels_for_50pct_events"].max()),
            ],
        },
        "positive_channel_profile_replication": correlations,
        "interpretation": (
            "A row-216-centered burden remains large enough to require isolation."
            if row216_plausible
            else (
                "Row 216 is not the driver, but stable multichannel/depth bands "
                "concentrate the positive burden."
                if concentrated_elsewhere
                else "Row 216 and its immediate depth band do not account for enough of "
                "the positive burden to explain the imec1 polarity imbalance alone."
            )
        ),
    }
    return channel_summary, localization, correlation_table, decision


def run(
    n_batches: int,
    batch_s: float,
    padding_s: float,
    threshold_uv: float,
    output_dir: Path,
) -> dict:
    specs = [spec for spec in load_specs() if spec.name.startswith("Luke imec1")]
    sos = butter(3, (300.0, 6000.0), btype="bandpass", fs=30000.0, output="sos")
    rows: list[dict] = []
    for spec in specs:
        print(f"Analyzing {spec.name}", flush=True)
        collapse_neighbors = spatial_neighbors(spec.locations_um, spec.shanks, 100.0)
        local_neighbors = spatial_neighbors(spec.locations_um, spec.shanks, 100.0)
        equal_neighbors = nearest_neighbors(spec.locations_um, spec.shanks, 5)
        starts = select_batch_starts(spec, n_batches, batch_s, padding_s)
        trim = int(round(padding_s * spec.sampling_rate_hz))
        temporal_radius = int(round(0.0005 * spec.sampling_rate_hz))
        for batch_index, start_s in enumerate(starts):
            raw_uv = read_batch(spec, float(start_s), batch_s, padding_s)
            filtered = sosfiltfilt(sos, raw_uv, axis=0).astype(np.float32)
            if trim:
                filtered = filtered[trim:-trim]
            stages = {
                "common_bandpass_equal_5_reference": local_median_reference(
                    filtered, equal_neighbors
                ),
                "common_bandpass_local_reference": local_median_reference(
                    filtered, local_neighbors
                ),
                "common_bandpass_shank_median": shank_median_reference(
                    filtered, spec.shanks
                ),
            }
            for stage, values in stages.items():
                for polarity, negative in (("negative", True), ("positive", False)):
                    times, channels, amplitudes = extrema_candidates(values, negative)
                    selected = np.flatnonzero(amplitudes >= threshold_uv)
                    kept_local = collapse_candidates(
                        times[selected],
                        channels[selected],
                        amplitudes[selected],
                        collapse_neighbors,
                        len(values),
                        temporal_radius,
                    )
                    kept_channels = channels[selected][kept_local]
                    counts = np.bincount(
                        kept_channels, minlength=spec.neural_channels
                    )
                    for channel, count in enumerate(counts):
                        rows.append(
                            {
                                "dataset": spec.name,
                                "window_kind": spec.window_kind,
                                "stage": stage,
                                "polarity": polarity,
                                "batch_index": batch_index,
                                "batch_start_s": float(start_s),
                                "duration_s": batch_s,
                                "channel": channel,
                                "y_um": float(spec.locations_um[channel, 1]),
                                "event_count": int(count),
                            }
                        )
            print(f"  batch {batch_index + 1}/{n_batches}", flush=True)
    batch_metrics = pd.DataFrame(rows)
    channel_summary, localization, correlations, decision = summarize_channel_counts(
        batch_metrics
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_metrics.to_csv(output_dir / "channel_event_batch_metrics.csv", index=False)
    channel_summary.to_csv(output_dir / "channel_event_summary.csv", index=False)
    localization.to_csv(output_dir / "localization_summary.csv", index=False)
    correlations.to_csv(output_dir / "profile_correlations.csv", index=False)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    args = parse_args()
    if args.summarize_existing:
        batch_metrics = pd.read_csv(args.output_dir / "channel_event_batch_metrics.csv")
        channel_summary, localization, correlations, result = summarize_channel_counts(
            batch_metrics
        )
        channel_summary.to_csv(args.output_dir / "channel_event_summary.csv", index=False)
        localization.to_csv(args.output_dir / "localization_summary.csv", index=False)
        correlations.to_csv(args.output_dir / "profile_correlations.csv", index=False)
        (args.output_dir / "decision.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2))
        return
    result = run(
        args.n_batches,
        args.batch_duration_s,
        args.padding_s,
        args.threshold_uv,
        args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
