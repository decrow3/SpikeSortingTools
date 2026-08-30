"""Reconstruct Kilosort-preprocessed waveforms for strong pair hypotheses.

Only pairs already passing the CCG/template/refractory screen are evaluated.
For sampled one-to-one coincident spikes, the script compares best one-template
and nonnegative two-template residual energy after applying Kilosort's saved
inverse-whitening transform, matching the exported ``templates.npy`` space.
It writes evidence only and never edits the sort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


SORTER = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1/sorts/core_depth_strip/"
    "single_ks_preprocessing_claim_off/sorter_output"
)
PAIR_CSV = Path(
    "testing/outputs/luke_full_strip_pair_ccg_audit/"
    "pair_ccg_template_metrics.csv"
)
OUTPUT = Path("testing/outputs/luke_full_strip_pair_residual_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorter", type=Path, default=SORTER)
    parser.add_argument("--pairs", type=Path, default=PAIR_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--events-per-pair", type=int, default=64)
    return parser.parse_args()


def one_to_one_centers(
    first: np.ndarray, second: np.ndarray, tolerance: int
) -> np.ndarray:
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    i = j = 0
    centers = []
    while i < len(first) and j < len(second):
        delta = int(first[i]) - int(second[j])
        if abs(delta) <= tolerance:
            centers.append(int(round((int(first[i]) + int(second[j])) / 2)))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return np.asarray(centers, dtype=np.int64)


def shift_template(template: np.ndarray, shift: int) -> np.ndarray:
    value = np.asarray(template)
    shifted = np.zeros_like(value)
    if shift < 0:
        shifted[:shift] = value[-shift:]
    elif shift > 0:
        shifted[shift:] = value[:-shift]
    else:
        shifted[:] = value
    return shifted


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float).ravel()
    b = np.asarray(second, dtype=float).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else np.nan


def best_shift(template: np.ndarray, waveform: np.ndarray, maximum: int = 3) -> tuple[np.ndarray, int, float]:
    candidates = []
    for shift in range(-maximum, maximum + 1):
        shifted = shift_template(template, shift)
        candidates.append((cosine(shifted, waveform), shift, shifted))
    value, shift, shifted = max(candidates, key=lambda item: item[0])
    return shifted, int(shift), float(value)


def normalized_residual(waveform: np.ndarray, design: np.ndarray) -> tuple[float, np.ndarray]:
    y = np.asarray(waveform, dtype=float).ravel()
    matrix = np.asarray(design, dtype=float).reshape(len(y), -1)
    coefficients, _ = nnls(matrix, y)
    residual = y - matrix @ coefficients
    denominator = np.dot(y, y)
    return (
        float(np.dot(residual, residual) / denominator) if denominator else np.nan,
        coefficients,
    )


def run_audit(
    sorter: Path, pair_csv: Path, output_dir: Path, events_per_pair: int
) -> dict:
    if events_per_pair <= 0:
        raise ValueError("events-per-pair must be positive")
    import torch
    from kilosort.io import bfile_from_ops, load_ops

    pairs = pd.read_csv(pair_csv)
    pairs = pairs[pairs.strong_duplicate_hypothesis.astype(bool)].copy()
    ops = load_ops(sorter / "ops.npy", device=torch.device("cpu"))
    bfile = bfile_from_ops(ops=ops, device=torch.device("cpu"))
    fs = float(ops["fs"])
    nt = int(ops["nt"])
    nt0min = int(ops["nt0min"])
    total_samples = int(bfile.n_samples)
    tolerance = int(round(0.5e-3 * fs))
    times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    templates = np.load(sorter / "templates.npy", mmap_mode="r")
    whitening_mat_inv = np.load(sorter / "whitening_mat_inv.npy")
    times_by_unit = {
        int(unit): np.sort(np.asarray(times[clusters == unit], dtype=np.int64))
        for unit in np.unique(clusters)
    }
    pair_rows = []
    event_rows = []
    context = 2048
    for pair in pairs.itertuples(index=False):
        centers = one_to_one_centers(
            times_by_unit[int(pair.first_unit)],
            times_by_unit[int(pair.second_unit)],
            tolerance,
        )
        centers = centers[(centers >= context) & (centers < total_samples - context)]
        selected = centers[
            np.linspace(0, len(centers) - 1, min(events_per_pair, len(centers)), dtype=int)
        ]
        waveforms = []
        for center in selected:
            start = int(center - context)
            filtered = bfile[start : int(center + context)].cpu().numpy()
            # Match kilosort.data_tools.get_spike_waveforms(): BinaryFiltered
            # is whitened, while exported templates are compared after the
            # saved inverse-whitening transform.
            filtered = (whitening_mat_inv @ filtered).T
            offset = context - nt0min
            waveforms.append(filtered[offset : offset + nt])
        waveforms = np.asarray(waveforms, dtype=float)
        median_waveform = np.median(waveforms, axis=0)
        first_template, first_shift, first_cosine = best_shift(
            templates[int(pair.first_unit)], median_waveform
        )
        second_template, second_shift, second_cosine = best_shift(
            templates[int(pair.second_unit)], median_waveform
        )
        first_vector = first_template.ravel()
        second_vector = second_template.ravel()
        design_two = np.column_stack([first_vector, second_vector])
        per_event = []
        for event_index, (center, waveform) in enumerate(zip(selected, waveforms)):
            first_residual, first_coef = normalized_residual(
                waveform, first_vector[:, None]
            )
            second_residual, second_coef = normalized_residual(
                waveform, second_vector[:, None]
            )
            two_residual, two_coef = normalized_residual(waveform, design_two)
            best_single = min(first_residual, second_residual)
            improvement = (best_single - two_residual) / max(best_single, 1e-12)
            per_event.append((first_residual, second_residual, two_residual, improvement))
            event_rows.append(
                {
                    "first_unit": int(pair.first_unit),
                    "second_unit": int(pair.second_unit),
                    "event_index": event_index,
                    "sample_index": int(center),
                    "time_s": float(center / fs),
                    "first_template_residual_fraction": first_residual,
                    "second_template_residual_fraction": second_residual,
                    "two_template_residual_fraction": two_residual,
                    "two_over_best_single_relative_improvement": improvement,
                    "first_coefficient": float(first_coef[0]),
                    "second_coefficient": float(second_coef[0]),
                    "two_first_coefficient": float(two_coef[0]),
                    "two_second_coefficient": float(two_coef[1]),
                }
            )
        per_event_array = np.asarray(per_event)
        median_improvement = float(np.median(per_event_array[:, 3]))
        supports_redundancy = bool(
            max(first_cosine, second_cosine) >= 0.8
            and min(first_cosine, second_cosine) >= 0.7
            and median_improvement <= 0.10
        )
        pair_rows.append(
            {
                "first_unit": int(pair.first_unit),
                "second_unit": int(pair.second_unit),
                "n_sampled_coincident_events": len(selected),
                "first_template_shift_samples": first_shift,
                "second_template_shift_samples": second_shift,
                "empirical_median_first_template_cosine": first_cosine,
                "empirical_median_second_template_cosine": second_cosine,
                "median_first_template_residual_fraction": float(np.median(per_event_array[:, 0])),
                "median_second_template_residual_fraction": float(np.median(per_event_array[:, 1])),
                "median_two_template_residual_fraction": float(np.median(per_event_array[:, 2])),
                "median_two_over_best_single_relative_improvement": median_improvement,
                "p90_two_over_best_single_relative_improvement": float(np.quantile(per_event_array[:, 3], 0.9)),
                "residual_supports_redundant_templates": supports_redundancy,
                "interpretation": (
                    "redundant-template hypothesis strengthened; manual waveform review still required"
                    if supports_redundancy
                    else "two-template or waveform evidence remains ambiguous; do not merge"
                ),
            }
        )
    pair_result = pd.DataFrame(pair_rows)
    event_result = pd.DataFrame(event_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_result.to_csv(output_dir / "pair_residual_summary.csv", index=False)
    event_result.to_csv(output_dir / "event_residual_metrics.csv", index=False)
    summary = {
        "input_strong_pair_hypotheses": int(len(pairs)),
        "pairs_with_residual_support_for_redundancy": int(
            pair_result.residual_supports_redundant_templates.sum()
        ),
        "pairs_remaining_ambiguous": int(
            (~pair_result.residual_supports_redundant_templates).sum()
        ),
        "events_reconstructed": int(len(event_result)),
        "automatic_curation_allowed": False,
        "preprocessing_space": (
            "Exact saved Kilosort CAR and FFT high-pass operators, followed by "
            "the saved inverse-whitening transform used by Kilosort's waveform "
            "export utility; no motion shift"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_audit(args.sorter, args.pairs, args.output_dir, args.events_per_pair),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
