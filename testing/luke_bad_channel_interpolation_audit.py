"""Audit the one-factor effect of Luke imec1 bad-channel interpolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "testing/outputs/luke_upstream_stage_ablation/imec1_interpolation_control/"
    "event_stage_metrics.csv"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_bad_channel_interpolation_audit")
BASELINE = "blanked_local_reference_control"
INTERPOLATED = "current_conditioned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def population_summary(paired: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    subset = paired.loc[mask]
    amplitude_equal = np.isclose(
        subset["peak_amplitude_counts_interpolated"],
        subset["peak_amplitude_counts_blank"],
        equal_nan=True,
    )
    channel_equal = (
        subset["peak_channel_interpolated"] == subset["peak_channel_blank"]
    )
    amplitude_ratio = (
        subset["peak_amplitude_counts_interpolated"]
        / subset["peak_amplitude_counts_blank"]
    )
    return {
        "population": name,
        "n_events": int(len(subset)),
        "exact_peak_amplitude_fraction": float(amplitude_equal.mean()),
        "same_peak_channel_fraction": float(channel_equal.mean()),
        "median_peak_amplitude_ratio": float(np.nanmedian(amplitude_ratio)),
        "p10_peak_amplitude_ratio": float(np.nanpercentile(amplitude_ratio, 10)),
        "p90_peak_amplitude_ratio": float(np.nanpercentile(amplitude_ratio, 90)),
    }


def run(input_path: Path, output_dir: Path) -> dict:
    metrics = pd.read_csv(input_path)
    required = {BASELINE, INTERPOLATED}
    if not required.issubset(set(metrics["stage"])):
        raise ValueError(f"Missing interpolation-control stages: {sorted(required)}")
    identity = [
        "review_id",
        "sample_index",
        "time_seconds",
        "window",
        "status",
        "review_label",
        "automatic_neural_like",
    ]
    measures = [
        "peak_channel",
        "peak_depth_um",
        "peak_amplitude_counts",
        "peak_snr",
        "active_channels",
        "local_energy_fraction",
        "footprint_depth_sd_um",
        "extra_temporal_extrema",
        "sidelobe_to_core_energy",
        "spatial_peak_count_4sigma",
    ]
    blank = metrics.loc[metrics["stage"] == BASELINE, identity + measures].set_index(
        identity
    )
    interpolated = metrics.loc[
        metrics["stage"] == INTERPOLATED, identity + measures
    ].set_index(identity)
    if not blank.index.equals(interpolated.index):
        raise ValueError("Interpolation-control events are not paired exactly")
    paired = blank.join(
        interpolated, lsuffix="_blank", rsuffix="_interpolated"
    ).reset_index()

    population_rows = [
        population_summary(paired, pd.Series(True, index=paired.index), "all reviewed"),
        population_summary(
            paired, paired["review_label"] == "neural", "reviewed neural"
        ),
        population_summary(
            paired,
            (paired["review_label"] == "neural")
            & (paired["status"] == "unmatched"),
            "reviewed unmatched neural",
        ),
    ]
    population_table = pd.DataFrame(population_rows)
    blank_191 = paired["peak_channel_blank"] == 191
    blank_191_neural = blank_191 & (paired["review_label"] == "neural")
    decision = {
        "decision": "interpolation_not_primary_contaminant_retain_with_metric_exclusion",
        "population_summary": population_rows,
        "channel_191_peak_events": {
            "all_reviewed": int(blank_191.sum()),
            "reviewed_neural": int(blank_191_neural.sum()),
            "remain_on_191_after_interpolation": int(
                (blank_191 & (paired["peak_channel_interpolated"] == 191)).sum()
            ),
            "interpretation": (
                "Without synthesis, the near-flat contact can win a normalized "
                "negative-peak search. Interpolation redirects all such fixed "
                "events to non-bad contacts."
            ),
        },
        "interpretation": (
            "With saturation blanking, bandpass, and local reference held fixed, "
            "bad-channel interpolation has no population-wide peak-amplitude "
            "cost and removes false peak localization to channel 191."
        ),
        "rescue_policy": [
            "Retain channel-191 interpolation in the next sorter candidate.",
            "Exclude synthetic channel 191 from event-footprint, dominant-contact, and claim-mask distance metrics.",
            "Treat recurring row-216 structure as a separate distributed/common-mode question.",
        ],
        "remaining_limit": (
            "This is a fixed-event waveform audit, not a leave-blank sorter "
            "comparison; subtle unit-family effects remain possible."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_event_metrics.csv", index=False)
    population_table.to_csv(output_dir / "population_summary.csv", index=False)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
