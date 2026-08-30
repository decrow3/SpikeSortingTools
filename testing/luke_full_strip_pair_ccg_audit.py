"""Refine high-priority full-strip excess-coincident unit pairs.

Near-synchronous firing is not assumed to be duplicate peeling.  This audit
adds full-session CCG peak shape, direct template waveform similarity, and
refractory behavior after a hypothetical 0.25 ms deduplicated merge.  The
result is a ranked hypothesis set for waveform/residual review, not automatic
curation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SORTER = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1/sorts/core_depth_strip/"
    "single_ks_preprocessing_claim_off/sorter_output"
)
PAIR_CSV = Path(
    "testing/outputs/luke_full_strip_diagnostic_audit/"
    "near_coincident_unit_pairs.csv"
)
OUTPUT = Path("testing/outputs/luke_full_strip_pair_ccg_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorter", type=Path, default=SORTER)
    parser.add_argument("--pairs", type=Path, default=PAIR_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def pair_count_within(first: np.ndarray, second: np.ndarray, tolerance: int) -> int:
    """Count all cross-train pairs within a symmetric sample tolerance."""
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    lower = np.searchsorted(second, first - tolerance, side="left")
    upper = np.searchsorted(second, first + tolerance, side="right")
    return int(np.sum(upper - lower))


def template_cosine_best_shift(
    first: np.ndarray, second: np.ndarray, maximum_shift: int = 3
) -> tuple[float, int]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    best = (-np.inf, 0)
    for shift in range(-maximum_shift, maximum_shift + 1):
        if shift < 0:
            a, b = first[-shift:], second[:shift]
        elif shift > 0:
            a, b = first[:-shift], second[shift:]
        else:
            a, b = first, second
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        value = float(np.sum(a * b) / denominator) if denominator else np.nan
        if np.isfinite(value) and value > best[0]:
            best = (value, shift)
    return float(best[0]), int(best[1])


def refractory_fraction(times: np.ndarray, refractory_samples: int) -> float:
    values = np.sort(np.asarray(times, dtype=np.int64))
    return float(np.mean(np.diff(values) < refractory_samples)) if len(values) > 1 else np.nan


def deduplicate_times(times: np.ndarray, tolerance: int) -> np.ndarray:
    values = np.sort(np.asarray(times, dtype=np.int64))
    if len(values) < 2:
        return values
    keep = np.ones(len(values), dtype=bool)
    last = values[0]
    for index in range(1, len(values)):
        if values[index] - last <= tolerance:
            keep[index] = False
        else:
            last = values[index]
    return values[keep]


def run_audit(sorter: Path, pair_csv: Path, output_dir: Path) -> dict:
    pairs = pd.read_csv(pair_csv)
    pairs = pairs[pairs.high_priority_pair.astype(bool)].copy()
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    templates = np.load(sorter / "templates.npy", mmap_mode="r")
    units = np.unique(clusters).astype(int)
    order = np.argsort(np.asarray(clusters), kind="stable")
    sorted_clusters = np.asarray(clusters)[order]
    groups = np.split(order, np.flatnonzero(np.diff(sorted_clusters)) + 1)
    times_by_unit = {
        int(np.asarray(clusters)[group[0]]): np.sort(
            np.asarray(times[group], dtype=np.int64)
        )
        for group in groups
    }
    if not set(pairs.first_unit).union(pairs.second_unit).issubset(set(units)):
        raise ValueError("Pair table references units absent from the sort")
    central_tolerance = int(round(0.5e-3 * fs))
    inner_shoulder = int(round(1.0e-3 * fs))
    outer_shoulder = int(round(5.0e-3 * fs))
    dedup_tolerance = int(round(0.25e-3 * fs))
    refractory = int(round(1.5e-3 * fs))
    rows = []
    for pair in pairs.itertuples(index=False):
        first_times = times_by_unit[int(pair.first_unit)]
        second_times = times_by_unit[int(pair.second_unit)]
        central_count = pair_count_within(
            first_times, second_times, central_tolerance
        )
        inner_count = pair_count_within(
            first_times, second_times, inner_shoulder
        )
        outer_count = pair_count_within(
            first_times, second_times, outer_shoulder
        )
        shoulder_count = max(0, outer_count - inner_count)
        # Central is 1 ms wide; the two 1--5 ms shoulders total 8 ms.
        expected_central_from_shoulder = shoulder_count / 8.0
        central_to_shoulder = central_count / max(
            expected_central_from_shoulder, 1e-12
        )
        merged = np.sort(np.r_[first_times, second_times])
        deduplicated = deduplicate_times(merged, dedup_tolerance)
        waveform_cosine, waveform_shift = template_cosine_best_shift(
            templates[int(pair.first_unit)], templates[int(pair.second_unit)]
        )
        first_rpv = refractory_fraction(first_times, refractory)
        second_rpv = refractory_fraction(second_times, refractory)
        merged_rpv = refractory_fraction(deduplicated, refractory)
        strong = bool(
            pair.smaller_unit_match_fraction >= 0.5
            and pair.jaccard_agreement >= 0.1
            and central_to_shoulder >= 5.0
            and waveform_cosine >= 0.8
            and merged_rpv <= 0.02
        )
        rows.append(
            {
                **pair._asdict(),
                "ccg_central_count_all_pairs": central_count,
                "ccg_shoulder_count_1_to_5ms": shoulder_count,
                "ccg_central_to_shoulder_density_ratio": central_to_shoulder,
                "template_waveform_best_cosine": waveform_cosine,
                "template_waveform_best_shift_samples": waveform_shift,
                "first_unit_refractory_fraction": first_rpv,
                "second_unit_refractory_fraction": second_rpv,
                "merged_spikes_before_dedup": len(merged),
                "merged_spikes_after_0_25ms_dedup": len(deduplicated),
                "merged_refractory_fraction_after_dedup": merged_rpv,
                "strong_duplicate_hypothesis": strong,
                "classification": (
                    "strong_duplicate_hypothesis_needs_residual_review"
                    if strong
                    else "ambiguous_synchrony_collision_or_template_split"
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["strong_duplicate_hypothesis", "jaccard_agreement"],
        ascending=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "pair_ccg_template_metrics.csv", index=False)
    summary = {
        "input_high_priority_pairs": int(len(pairs)),
        "strong_duplicate_hypotheses": int(result.strong_duplicate_hypothesis.sum()),
        "strong_hypotheses_with_any_ks_good": int(
            (
                result.strong_duplicate_hypothesis
                & (result.first_ks_good | result.second_ks_good)
            ).sum()
        ),
        "ambiguous_pairs": int((~result.strong_duplicate_hypothesis).sum()),
        "median_ccg_central_to_shoulder_density_ratio": float(
            result.ccg_central_to_shoulder_density_ratio.median()
        ),
        "median_template_waveform_best_cosine": float(
            result.template_waveform_best_cosine.median()
        ),
        "automatic_curation_allowed": False,
        "next_gate": "Reconstruct conditioned/KS-preprocessed waveforms and residual reduction for strong hypotheses before any merge or deletion.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(run_audit(args.sorter, args.pairs, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
