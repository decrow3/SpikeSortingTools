"""Refine imec0 similar-good-template pairs with CCG and artifact evidence.

This audit is diagnostic only. It does not merge, delete, or relabel units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from testing.luke_full_strip_pair_ccg_audit import (
    deduplicate_times,
    pair_count_within,
    refractory_fraction,
    template_cosine_best_shift,
)
from testing.luke_full_strip_pair_residual_audit import one_to_one_centers


SORTER = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/"
    "kilosort4/sorter_output"
)
PAIRS = Path(
    "testing/outputs/luke_full_probe_rescue_diagnostics_imec0_rescue/"
    "similar_template_pairs.csv"
)
SIDECAR = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/"
    "artifacts/raw_over_500uv.h5"
)
OUTPUT = Path("testing/outputs/luke_imec0_similar_pair_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorter", type=Path, default=SORTER)
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def nearest_distance_frames(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if reference.size == 0:
        return np.full(values.shape, np.iinfo(np.int64).max, dtype=np.int64)
    insertion = np.searchsorted(reference, values)
    left_index = np.maximum(insertion - 1, 0)
    right_index = np.minimum(insertion, reference.size - 1)
    return np.minimum(
        np.abs(values - reference[left_index]),
        np.abs(reference[right_index] - values),
    )


def proximity_fractions(distances: np.ndarray, fs: float) -> dict[str, float]:
    return {
        f"artifact_proximity_{label}_fraction": float(np.mean(distances <= frames))
        if distances.size
        else np.nan
        for label, frames in (
            ("0p5ms", int(round(0.5e-3 * fs))),
            ("1ms", int(round(1e-3 * fs))),
            ("2ms", int(round(2e-3 * fs))),
            ("5ms", int(round(5e-3 * fs))),
        )
    }


def run(sorter: Path, pair_path: Path, sidecar: Path, output_dir: Path) -> dict:
    pairs = pd.read_csv(pair_path)
    pairs = pairs[pairs.both_good.astype(bool)].copy()
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    templates = np.load(sorter / "templates.npy", mmap_mode="r")
    labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t")
    label_column = next(column for column in labels if column != "cluster_id")
    good_units = labels.loc[
        labels[label_column].astype(str).str.lower().eq("good"), "cluster_id"
    ].to_numpy(int)
    with h5py.File(sidecar, "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError(f"Artifact sidecar is not complete: {sidecar}")
        artifact_samples = handle["claim_active_sample_index"][:].astype(np.int64)

    cluster_values = np.asarray(clusters)
    order = np.argsort(cluster_values, kind="stable")
    sorted_clusters = cluster_values[order]
    groups = np.split(order, np.flatnonzero(np.diff(sorted_clusters)) + 1)
    times_by_unit = {
        int(cluster_values[group[0]]): np.asarray(times[group], dtype=np.int64)
        for group in groups
    }
    central_tolerance = int(round(0.5e-3 * fs))
    inner_shoulder = int(round(1e-3 * fs))
    outer_shoulder = int(round(5e-3 * fs))
    dedup_tolerance = int(round(0.25e-3 * fs))
    refractory = int(round(1.5e-3 * fs))

    unit_rows = []
    for unit in good_units:
        unit_times = times_by_unit[int(unit)]
        distances = nearest_distance_frames(unit_times, artifact_samples)
        unit_rows.append(
            {
                "unit_id": int(unit),
                "spike_count": int(unit_times.size),
                **proximity_fractions(distances, fs),
            }
        )
    unit_result = pd.DataFrame(unit_rows)
    unit_artifact_2ms = unit_result.set_index("unit_id")[
        "artifact_proximity_2ms_fraction"
    ]

    pair_rows = []
    for pair in pairs.itertuples(index=False):
        first_unit, second_unit = int(pair.unit_first), int(pair.unit_second)
        first_times = times_by_unit[first_unit]
        second_times = times_by_unit[second_unit]
        centers = one_to_one_centers(first_times, second_times, central_tolerance)
        center_distances = nearest_distance_frames(centers, artifact_samples)
        first_distances = nearest_distance_frames(first_times, artifact_samples)
        second_distances = nearest_distance_frames(second_times, artifact_samples)
        pooled_distances = np.r_[first_distances, second_distances]
        central_count = pair_count_within(first_times, second_times, central_tolerance)
        inner_count = pair_count_within(first_times, second_times, inner_shoulder)
        outer_count = pair_count_within(first_times, second_times, outer_shoulder)
        shoulder_count = max(0, outer_count - inner_count)
        central_to_shoulder = central_count / max(shoulder_count / 8.0, 1e-12)
        matched_fraction = len(centers) / max(min(len(first_times), len(second_times)), 1)
        jaccard = len(centers) / max(
            len(first_times) + len(second_times) - len(centers), 1
        )
        waveform_cosine, waveform_shift = template_cosine_best_shift(
            templates[first_unit], templates[second_unit]
        )
        merged = np.sort(np.r_[first_times, second_times])
        deduplicated = deduplicate_times(merged, dedup_tolerance)
        merged_rpv = refractory_fraction(deduplicated, refractory)
        center_proximity = proximity_fractions(center_distances, fs)
        pooled_proximity = proximity_fractions(pooled_distances, fs)
        center_2ms = center_proximity["artifact_proximity_2ms_fraction"]
        pooled_2ms = pooled_proximity["artifact_proximity_2ms_fraction"]
        artifact_enriched = bool(
            len(centers) >= 20
            and center_2ms >= pooled_2ms + 0.02
            and center_2ms >= 2 * max(pooled_2ms, 1e-12)
        )
        artifact_associated = bool(pooled_2ms >= 0.5 or center_2ms >= 0.5)
        strong_duplicate = bool(
            matched_fraction >= 0.5
            and jaccard >= 0.1
            and central_to_shoulder >= 5.0
            and waveform_cosine >= 0.8
            and merged_rpv <= 0.02
        )
        partial_duplicate = bool(
            matched_fraction >= 0.25
            and jaccard >= 0.1
            and central_to_shoulder >= 5.0
            and waveform_cosine >= 0.8
            and merged_rpv <= 0.02
        )
        pair_rows.append(
            {
                **pair._asdict(),
                "first_spike_count": len(first_times),
                "second_spike_count": len(second_times),
                "one_to_one_coincident_events": len(centers),
                "smaller_unit_match_fraction": matched_fraction,
                "coincident_jaccard": jaccard,
                "ccg_central_to_shoulder_density_ratio": central_to_shoulder,
                "template_waveform_best_cosine": waveform_cosine,
                "template_waveform_best_shift_samples": waveform_shift,
                "merged_refractory_fraction_after_0p25ms_dedup": merged_rpv,
                "merged_spike_reduction_fraction_0p25ms": 1
                - len(deduplicated) / len(merged),
                "first_unit_artifact_2ms_fraction": float(
                    unit_artifact_2ms[first_unit]
                ),
                "second_unit_artifact_2ms_fraction": float(
                    unit_artifact_2ms[second_unit]
                ),
                **{
                    f"coincident_{key}": value
                    for key, value in center_proximity.items()
                },
                **{
                    f"pooled_pair_spikes_{key}": value
                    for key, value in pooled_proximity.items()
                },
                "coincident_artifact_enriched": artifact_enriched,
                "artifact_associated_pair": artifact_associated,
                "strong_duplicate_hypothesis": strong_duplicate,
                "partial_or_strong_duplicate_hypothesis": partial_duplicate,
            }
        )
    pair_result = pd.DataFrame(pair_rows).sort_values(
        ["strong_duplicate_hypothesis", "coincident_artifact_enriched", "template_similarity"],
        ascending=False,
    )
    summary = {
        "input_similar_good_pairs": int(len(pair_result)),
        "unique_good_units_in_pairs": int(
            len(set(pair_result.unit_first).union(pair_result.unit_second))
        ),
        "strong_duplicate_hypotheses": int(
            pair_result.strong_duplicate_hypothesis.sum()
        ),
        "artifact_enriched_coincident_pairs": int(
            pair_result.coincident_artifact_enriched.sum()
        ),
        "artifact_associated_pairs": int(pair_result.artifact_associated_pair.sum()),
        "strong_and_artifact_associated_pairs": int(
            (
                pair_result.strong_duplicate_hypothesis
                & pair_result.artifact_associated_pair
            ).sum()
        ),
        "partial_or_strong_duplicate_hypotheses": int(
            pair_result.partial_or_strong_duplicate_hypothesis.sum()
        ),
        "partial_or_strong_and_artifact_associated_pairs": int(
            (
                pair_result.partial_or_strong_duplicate_hypothesis
                & pair_result.artifact_associated_pair
            ).sum()
        ),
        "median_good_unit_artifact_2ms_fraction": float(
            unit_result.artifact_proximity_2ms_fraction.median()
        ),
        "median_similar_pair_coincident_artifact_2ms_fraction": float(
            pair_result.coincident_artifact_proximity_2ms_fraction.median()
        ),
        "automatic_curation_allowed": False,
        "interpretation_guardrail": (
            "Similarity, CCG, and artifact proximity rank hypotheses; they do not "
            "establish biological identity or authorize merging."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    unit_result.to_csv(output_dir / "good_unit_artifact_proximity.csv", index=False)
    pair_result.to_csv(output_dir / "similar_good_pair_audit.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args.sorter, args.pairs, args.sidecar, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
