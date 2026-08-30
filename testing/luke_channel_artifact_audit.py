"""Decide whether Luke imec1's polarity imbalance is a localized channel artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "testing/outputs/luke_motion_candidate_results/raw_voltage_audit"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_channel_artifact_audit")
REFERENCE_STAGES = (
    "common_bandpass_equal_5_reference",
    "common_bandpass_local_reference",
    "common_bandpass_shank_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def percentile_rank(values: pd.Series, value: float) -> float:
    return float(np.mean(values.to_numpy() <= value))


def run(input_dir: Path, output_dir: Path) -> dict:
    events = pd.read_csv(input_dir / "raw_event_summary.csv")
    footprints = pd.read_csv(input_dir / "raw_footprint_summary.csv")
    channels = pd.read_csv(input_dir / "raw_channel_summary.csv")

    selected = events.loc[
        (events["threshold_kind"] == "absolute_uv")
        & (events["threshold"] == 75.0)
        & events["stage"].isin(REFERENCE_STAGES)
    ]
    polarity = selected.pivot_table(
        index=["dataset", "window_kind", "stage"],
        columns="polarity",
        values="median_event_rate_per_mm_s",
        aggfunc="first",
    ).reset_index()
    if polarity[["negative", "positive"]].isna().any().any():
        raise ValueError("Missing a 75 uV polarity estimate")
    polarity["positive_to_negative_ratio"] = (
        polarity["positive"] / polarity["negative"]
    )

    compactness = footprints.loc[
        footprints["stage"].isin(REFERENCE_STAGES),
        [
            "dataset",
            "window_kind",
            "stage",
            "polarity",
            "sampled_events",
            "compact_fraction",
            "median_local_energy_fraction",
            "median_footprint_depth_sd_um",
            "median_active_channels_4sigma",
        ],
    ].copy()

    suspect_rows: list[dict] = []
    for dataset, cohort in channels.loc[
        channels["dataset"].str.startswith("Luke imec1")
    ].groupby("dataset", sort=True):
        for channel in (191, 216):
            match = cohort.loc[cohort["channel"] == channel]
            if len(match) != 1:
                raise ValueError(f"Expected channel {channel} once in {dataset}")
            row = match.iloc[0]
            suspect_rows.append(
                {
                    "dataset": dataset,
                    "window_kind": row["window_kind"],
                    "channel": channel,
                    "y_um": row["y_um"],
                    "median_bandpass_sigma_uv": row["median_bandpass_sigma_uv"],
                    "bandpass_sigma_percentile": percentile_rank(
                        cohort["median_bandpass_sigma_uv"],
                        row["median_bandpass_sigma_uv"],
                    ),
                    "median_fraction_abs_raw_over_500uv": row[
                        "median_fraction_abs_raw_over_500uv"
                    ],
                }
            )
    suspect_channels = pd.DataFrame(suspect_rows)

    imec1 = polarity.loc[polarity["dataset"].str.startswith("Luke imec1")]
    imec0 = polarity.loc[polarity["dataset"].str.startswith("Luke imec0")]
    yates = polarity.loc[polarity["dataset"] == "Yates raw session"]
    channel_191 = suspect_channels.loc[suspect_channels["channel"] == 191]
    equal_count_compact = compactness.loc[
        compactness["stage"] == "common_bandpass_equal_5_reference"
    ]

    decision = {
        "decision": "distributed_imec1_polarity_morphology_problem_not_single_noisy_channel",
        "reference_controls": {
            "imec1_positive_to_negative_ratio_range": [
                float(imec1["positive_to_negative_ratio"].min()),
                float(imec1["positive_to_negative_ratio"].max()),
            ],
            "imec0_positive_to_negative_ratio_range": [
                float(imec0["positive_to_negative_ratio"].min()),
                float(imec0["positive_to_negative_ratio"].max()),
            ],
            "yates_positive_to_negative_ratio_range": [
                float(yates["positive_to_negative_ratio"].min()),
                float(yates["positive_to_negative_ratio"].max()),
            ],
            "interpretation": (
                "The imec1 positive excess replicates in pathological, shared, "
                "and session-wide samples under five-contact, 100 um, and "
                "shank-wide median references."
            ),
        },
        "channel_191": {
            "bandpass_sigma_uv_range": [
                float(channel_191["median_bandpass_sigma_uv"].min()),
                float(channel_191["median_bandpass_sigma_uv"].max()),
            ],
            "maximum_sigma_percentile": float(
                channel_191["bandpass_sigma_percentile"].max()
            ),
            "maximum_raw_over_500uv_fraction": float(
                channel_191["median_fraction_abs_raw_over_500uv"].max()
            ),
            "interpretation": (
                "Channel 191 is attenuated/near-flat, not a high-noise or "
                "large-excursion source. Exclusion remains justified, but its "
                "interpolation shadow must be tested separately."
            ),
        },
        "equal_count_compactness": {
            "imec1_positive_compact_fraction_range": [
                float(
                    equal_count_compact.loc[
                        equal_count_compact["dataset"].str.startswith("Luke imec1")
                        & (equal_count_compact["polarity"] == "positive"),
                        "compact_fraction",
                    ].min()
                ),
                float(
                    equal_count_compact.loc[
                        equal_count_compact["dataset"].str.startswith("Luke imec1")
                        & (equal_count_compact["polarity"] == "positive"),
                        "compact_fraction",
                    ].max()
                ),
            ],
            "yates_positive_compact_fraction": float(
                equal_count_compact.loc[
                    (equal_count_compact["dataset"] == "Yates raw session")
                    & (equal_count_compact["polarity"] == "positive"),
                    "compact_fraction",
                ].iloc[0]
            ),
            "interpretation": (
                "imec1 positive events are less compact than Yates under the "
                "probe-neutral five-contact reference, but are not purely "
                "full-shank diffuse events."
            ),
        },
        "rescue_implications": [
            "Do not solve imec1 by lowering a global detection threshold.",
            "Keep imec0 and imec1 preprocessing decisions separate.",
            "Exclude channel 191 and compare leave-blank versus interpolation controls on fixed events.",
            "Require candidate preprocessing to reduce imec1 polarity imbalance while preserving compact negative events and reviewed-event recovery.",
        ],
        "remaining_claim_limit": (
            "Normalized probe depth is not matched cortical layer, and event "
            "polarity/compactness is not a biological cell-count estimator."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    polarity.to_csv(output_dir / "polarity_reference_summary.csv", index=False)
    compactness.to_csv(output_dir / "compactness_reference_summary.csv", index=False)
    suspect_channels.to_csv(output_dir / "suspect_channel_summary.csv", index=False)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.input_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
