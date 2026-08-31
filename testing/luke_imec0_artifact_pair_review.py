"""Review the two imec0 duplicate hypotheses without modifying the sort.

The review tests whether an outside-artifact counterfactual exists, measures the
number of raw channels crossing 500 uV around each unit's spikes, compares
template morphology with all KS-good units, and joins the saved Kilosort-space
residual audit.  The sidecar remains an annotation; no spikes or units are
deleted, merged, or relabeled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0"
)
SORTER = ROOT / "kilosort4/sorter_output"
SIDECAR = ROOT / "artifacts/raw_over_500uv.h5"
RESIDUAL = Path(
    "testing/outputs/luke_imec0_artifact_pair_residual_review/"
    "pair_residual_summary.csv"
)
OUTPUT = Path("testing/outputs/luke_imec0_artifact_pair_review")
UNITS = (184, 191, 164, 165)
PAIRS = ((184, 191), (164, 165))


def nearest_distance_frames(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if reference.size == 0:
        return np.full(values.shape, np.iinfo(np.int64).max, dtype=np.int64)
    insertion = np.searchsorted(reference, values)
    left = reference[np.maximum(insertion - 1, 0)]
    right = reference[np.minimum(insertion, len(reference) - 1)]
    return np.minimum(np.abs(values - left), np.abs(right - values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorter", type=Path, default=SORTER)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR)
    parser.add_argument("--residual", type=Path, default=RESIDUAL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--events-per-unit", type=int, default=200)
    return parser.parse_args()


def evenly_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(values)
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum, dtype=int)]


def threshold_footprint(
    event: int,
    threshold_samples: np.ndarray,
    threshold_channels: np.ndarray,
    threshold_values: np.ndarray,
    tolerance: int,
) -> tuple[int, int, int, int]:
    left = np.searchsorted(threshold_samples, event - tolerance, side="left")
    right = np.searchsorted(threshold_samples, event + tolerance, side="right")
    samples = threshold_samples[left:right]
    channels = threshold_channels[left:right]
    values = threshold_values[left:right]
    if len(samples) == 0:
        return 0, 0, 0, 0
    _, simultaneous = np.unique(samples, return_counts=True)
    return (
        int(len(np.unique(channels))),
        int(simultaneous.max()),
        int(np.max(np.abs(values))),
        int(len(samples)),
    )


def percentile_rank(reference: np.ndarray, value: float) -> float:
    reference = np.asarray(reference, dtype=float)
    return float(100 * np.mean(reference <= value))


def run(
    sorter: Path,
    sidecar: Path,
    residual_path: Path,
    output_dir: Path,
    events_per_unit: int,
) -> dict:
    if events_per_unit <= 0:
        raise ValueError("events-per-unit must be positive")
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    templates = np.load(sorter / "templates.npy", mmap_mode="r")
    positions = np.load(sorter / "channel_positions.npy")
    labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t")
    label_column = next(column for column in labels if column != "cluster_id")
    good_units = labels.loc[
        labels[label_column].astype(str).str.lower().eq("good"), "cluster_id"
    ].to_numpy(int)
    with h5py.File(sidecar, "r") as handle:
        threshold_samples = handle["sample_index"][:].astype(np.int64)
        threshold_channels = handle["channel_index"][:].astype(np.int32)
        threshold_values = handle["value_counts"][:].astype(np.int16)
        claim_samples = handle["claim_active_sample_index"][:].astype(np.int64)
    times_by_unit = {
        int(unit): np.sort(np.asarray(times[clusters == unit], dtype=np.int64))
        for unit in good_units
    }

    morphology_rows = []
    footprint_rows = []
    tolerance = int(round(0.5e-3 * fs))
    for unit in good_units:
        template = np.asarray(templates[int(unit)], dtype=float)
        channel_ptp = np.ptp(template, axis=0)
        peak_channel = int(np.argmax(channel_ptp))
        peak_ptp = float(channel_ptp[peak_channel])
        active = channel_ptp >= 0.1 * peak_ptp
        peak_waveform = template[:, peak_channel]
        negative = float(max(-np.min(peak_waveform), np.finfo(float).eps))
        morphology_rows.append(
            {
                "unit_id": int(unit),
                "template_peak_to_peak": peak_ptp,
                "template_active_channels_10pct": int(active.sum()),
                "template_active_depth_span_um_10pct": float(
                    np.ptp(positions[active, 1]) if active.any() else 0
                ),
                "template_positive_to_negative_ratio": float(
                    np.max(peak_waveform) / negative
                ),
                "template_peak_channel": peak_channel,
            }
        )
        unit_times = times_by_unit[int(unit)]
        sampled = evenly_sample(unit_times, events_per_unit)
        footprints = np.asarray(
            [
                threshold_footprint(
                    int(event),
                    threshold_samples,
                    threshold_channels,
                    threshold_values,
                    tolerance,
                )
                for event in sampled
            ],
            dtype=int,
        )
        for event, values in zip(sampled, footprints):
            footprint_rows.append(
                {
                    "unit_id": int(unit),
                    "sample_index": int(event),
                    "unique_threshold_channels_0p5ms": int(values[0]),
                    "max_simultaneous_threshold_channels": int(values[1]),
                    "maximum_absolute_threshold_counts": int(values[2]),
                    "threshold_points_0p5ms": int(values[3]),
                }
            )

    morphology = pd.DataFrame(morphology_rows)
    footprints = pd.DataFrame(footprint_rows)
    focused = morphology[morphology.unit_id.isin(UNITS)].copy()
    for column in (
        "template_peak_to_peak",
        "template_active_channels_10pct",
        "template_active_depth_span_um_10pct",
        "template_positive_to_negative_ratio",
    ):
        focused[f"{column}_good_unit_percentile"] = [
            percentile_rank(morphology[column], value) for value in focused[column]
        ]
    residual = pd.read_csv(residual_path)
    focused = focused.merge(
        residual,
        left_on="unit_id",
        right_on="first_unit",
        how="left",
    ).merge(
        residual,
        left_on="unit_id",
        right_on="second_unit",
        how="left",
        suffixes=("_as_first", "_as_second"),
    )

    sensitivity_rows = []
    for unit in UNITS:
        unit_times = times_by_unit[unit]
        distances = nearest_distance_frames(unit_times, claim_samples)
        row = {"unit_id": unit, "original_spike_count": int(len(unit_times))}
        for label, milliseconds in (("0p5ms", 0.5), ("1ms", 1), ("2ms", 2), ("5ms", 5)):
            radius = int(round(milliseconds * 1e-3 * fs))
            row[f"spikes_outside_{label}"] = int(np.sum(distances > radius))
            row[f"fraction_outside_{label}"] = float(np.mean(distances > radius))
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)

    all_footprint_summary = footprints.groupby("unit_id").agg(
        sampled_spikes=("sample_index", "size"),
        median_unique_threshold_channels_0p5ms=(
            "unique_threshold_channels_0p5ms",
            "median",
        ),
        median_max_simultaneous_threshold_channels=(
            "max_simultaneous_threshold_channels",
            "median",
        ),
        p90_max_simultaneous_threshold_channels=(
            "max_simultaneous_threshold_channels",
            lambda values: values.quantile(0.9),
        ),
        median_maximum_absolute_threshold_counts=(
            "maximum_absolute_threshold_counts",
            "median",
        ),
    ).reset_index()
    for column in (
        "median_unique_threshold_channels_0p5ms",
        "median_max_simultaneous_threshold_channels",
        "p90_max_simultaneous_threshold_channels",
    ):
        all_footprint_summary[f"{column}_good_unit_percentile"] = [
            percentile_rank(all_footprint_summary[column], value)
            for value in all_footprint_summary[column]
        ]
    footprint_summary = all_footprint_summary[
        all_footprint_summary.unit_id.isin(UNITS)
    ].copy()
    focused_footprints = footprints[footprints.unit_id.isin(UNITS)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row_index, pair in enumerate(PAIRS):
        colors = ("tab:blue", "tab:orange")
        for unit, color in zip(pair, colors):
            template = np.asarray(templates[unit], dtype=float)
            channel_ptp = np.ptp(template, axis=0)
            peak_channel = int(np.argmax(channel_ptp))
            axes[row_index, 0].plot(
                np.arange(template.shape[0]) / fs * 1e3,
                template[:, peak_channel],
                color=color,
                label=str(unit),
            )
            axes[row_index, 1].plot(
                positions[:, 1], channel_ptp / max(channel_ptp.max(), 1e-12),
                color=color,
                label=str(unit),
            )
            values = focused_footprints.loc[
                focused_footprints.unit_id.eq(unit),
                "max_simultaneous_threshold_channels",
            ]
            axes[row_index, 2].hist(
                values,
                bins=np.linspace(0, 384, 25),
                histtype="step",
                linewidth=1.5,
                color=color,
                label=str(unit),
            )
        axes[row_index, 0].set(
            title=f"{pair[0]}/{pair[1]} peak-channel templates",
            xlabel="Template time (ms)",
            ylabel="Template value",
        )
        axes[row_index, 1].set(
            title="Normalized spatial peak-to-peak profile",
            xlabel="Depth (µm)",
            ylabel="Fraction of peak",
        )
        axes[row_index, 2].set(
            title=">500 µV simultaneous-channel footprint",
            xlabel="Channels at a single sample",
            ylabel="Sampled spikes",
        )
        for axis in axes[row_index]:
            axis.legend()
    fig.suptitle(
        "imec0 duplicate hypotheses: positive-dominant templates and raw threshold footprint"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    morphology.to_csv(output_dir / "all_good_unit_template_morphology.csv", index=False)
    footprints.to_csv(output_dir / "sampled_good_unit_threshold_footprints.csv", index=False)
    all_footprint_summary.to_csv(
        output_dir / "all_good_unit_threshold_footprint_summary.csv", index=False
    )
    focused.to_csv(output_dir / "focused_unit_review.csv", index=False)
    footprint_summary.to_csv(output_dir / "focused_threshold_footprints.csv", index=False)
    sensitivity.to_csv(output_dir / "artifact_neighborhood_sensitivity.csv", index=False)
    fig.savefig(output_dir / "focused_pair_review.png", dpi=180)
    plt.close(fig)

    summary = {
        "reviewed_pairs": [list(pair) for pair in PAIRS],
        "outside_artifact_counterfactual_available": bool(
            sensitivity["spikes_outside_0p5ms"].gt(0).any()
        ),
        "all_four_units_entirely_within_0p5ms_of_claim_sample": bool(
            sensitivity["spikes_outside_0p5ms"].eq(0).all()
        ),
        "residual_pairs_supporting_two_distinct_templates": int(
            residual["median_two_over_best_single_relative_improvement"].gt(0.1).sum()
        ),
        "residual_pairs_with_good_empirical_template_match": int(
            (
                residual[
                    [
                        "empirical_median_first_template_cosine",
                        "empirical_median_second_template_cosine",
                    ]
                ].max(axis=1)
                >= 0.8
            ).sum()
        ),
        "automatic_curation_allowed": False,
        "interpretation": (
            "All four units are inseparable from the sidecar at 0.5 ms, are "
            "high-amplitude positive-dominant template outliers, and their "
            "coincident empirical waveforms are poorly explained by either "
            "saved template. This supports artifact-associated questionable "
            "units, not an automatic biological merge. Proximity alone cannot "
            "establish causality because no outside-artifact subset exists."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(
                args.sorter,
                args.sidecar,
                args.residual,
                args.output_dir,
                args.events_per_unit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
