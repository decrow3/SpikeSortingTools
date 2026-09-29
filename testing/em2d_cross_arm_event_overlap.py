#!/usr/bin/env python3
"""Conservative cross-arm unit association from cached event times."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sort(receipt: dict[str, object]) -> dict[str, object]:
    path = Path(receipt["path"])
    if sha256(path) != receipt["sha256"]:
        raise RuntimeError(f"sorting hash differs: {path}")
    with np.load(path, allow_pickle=False) as saved:
        times = np.asarray(saved["times_samples"], dtype=np.int64)
        labels = np.asarray(saved["labels"], dtype=np.int64)
        channels = np.asarray(saved["channels"], dtype=np.int64)
        fs = float(saved["sampling_frequency"])
    keep = labels >= 0
    times, labels, channels = times[keep], labels[keep], channels[keep]
    order = np.argsort(times, kind="stable")
    return {
        "times": times[order],
        "labels": labels[order],
        "channels": channels[order],
        "sampling_frequency": fs,
        "path": str(path),
        "sha256": receipt["sha256"],
    }


def count_one_to_one(a: np.ndarray, b: np.ndarray, tolerance: int) -> int:
    """Maximum cardinality ordered matching within an integer tolerance."""
    i = j = count = 0
    while i < len(a) and j < len(b):
        delta = int(a[i]) - int(b[j])
        if abs(delta) <= tolerance:
            count += 1
            i += 1
            j += 1
        elif delta < -tolerance:
            i += 1
        else:
            j += 1
    return count


def candidate_pairs(
    a_times: np.ndarray,
    a_labels: np.ndarray,
    b_times: np.ndarray,
    b_labels: np.ndarray,
    *,
    tolerance: int,
    minimum: int,
) -> set[tuple[int, int]]:
    by_time: dict[int, set[int]] = defaultdict(set)
    for time, label in zip(b_times, b_labels):
        by_time[int(time)].add(int(label))
    counts: Counter[tuple[int, int]] = Counter()
    for time, label in zip(a_times, a_labels):
        seen = set()
        for candidate_time in range(int(time) - tolerance, int(time) + tolerance + 1):
            seen.update(by_time.get(candidate_time, ()))
        counts.update((int(label), other) for other in seen)
    return {pair for pair, count in counts.items() if count >= minimum}


def unit_arrays(values: dict[str, object]) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    times = np.asarray(values["times"])
    labels = np.asarray(values["labels"])
    channels = np.asarray(values["channels"])
    units = {}
    medians = {}
    for unit in np.unique(labels):
        selected = labels == unit
        units[int(unit)] = times[selected]
        medians[int(unit)] = float(np.median(channels[selected]))
    return units, medians


def score_pairs(
    reference: dict[str, object],
    comparator: dict[str, object],
    *,
    tolerance: int,
    minimum: int,
) -> pd.DataFrame:
    a_units, a_channels = unit_arrays(reference)
    b_units, b_channels = unit_arrays(comparator)
    candidates = candidate_pairs(
        np.asarray(reference["times"]),
        np.asarray(reference["labels"]),
        np.asarray(comparator["times"]),
        np.asarray(comparator["labels"]),
        tolerance=tolerance,
        minimum=minimum,
    )
    rows = []
    for a, b in sorted(candidates):
        matched = count_one_to_one(a_units[a], b_units[b], tolerance)
        if matched < minimum:
            continue
        na, nb = len(a_units[a]), len(b_units[b])
        rows.append(
            {
                "reference_unit": a,
                "comparator_unit": b,
                "reference_events": na,
                "comparator_events": nb,
                "matched_events": matched,
                "f1": 2 * matched / (na + nb),
                "reference_recall": matched / na,
                "comparator_recall": matched / nb,
                "overlap_coefficient": matched / min(na, nb),
                "median_channel_difference": b_channels[b] - a_channels[a],
            }
        )
    return pd.DataFrame(rows)


def best_rows(scores: pd.DataFrame, side: str) -> pd.DataFrame:
    unit = f"{side}_unit"
    other = "comparator_unit" if side == "reference" else "reference_unit"
    if scores.empty:
        return pd.DataFrame(columns=[unit, other, "f1", "second_f1", "ambiguous"])
    rows = []
    for value, group in scores.groupby(unit):
        ordered = group.sort_values(
            ["f1", "matched_events", other], ascending=[False, False, True]
        )
        best = ordered.iloc[0].to_dict()
        second = float(ordered.iloc[1].f1) if len(ordered) > 1 else np.nan
        best["second_f1"] = second
        best["ambiguous"] = bool(np.isfinite(second) and best["f1"] - second <= 0.05)
        rows.append(best)
    return pd.DataFrame(rows)


def summarize(
    scores: pd.DataFrame,
    reference: dict[str, object],
    comparator: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    ref_best = best_rows(scores, "reference")
    cmp_best = best_rows(scores, "comparator")
    reverse = {
        int(row.comparator_unit): int(row.reference_unit) for row in cmp_best.itertuples()
    }
    reciprocal_mask = [
        reverse.get(int(row.comparator_unit)) == int(row.reference_unit)
        for row in ref_best.itertuples()
    ]
    reciprocal = ref_best[np.asarray(reciprocal_mask, dtype=bool)].copy()
    reciprocal["reciprocal"] = True
    ref_units = np.unique(reference["labels"])
    cmp_units = np.unique(comparator["labels"])
    matched = int(reciprocal.matched_events.sum()) if len(reciprocal) else 0
    f1 = reciprocal.f1.to_numpy(float) if len(reciprocal) else np.asarray([])
    summary = {
        "reference_units": int(len(ref_units)),
        "comparator_units": int(len(cmp_units)),
        "candidate_pairs": int(len(scores)),
        "reciprocal_pairs": int(len(reciprocal)),
        "reference_units_without_candidate": int(len(ref_units) - ref_best.reference_unit.nunique()),
        "comparator_units_without_candidate": int(len(cmp_units) - cmp_best.comparator_unit.nunique()),
        "ambiguous_reference_best": int(ref_best.ambiguous.sum()) if len(ref_best) else 0,
        "ambiguous_comparator_best": int(cmp_best.ambiguous.sum()) if len(cmp_best) else 0,
        "reciprocal_f1_p10": float(np.quantile(f1, 0.1)) if len(f1) else None,
        "reciprocal_f1_median": float(np.median(f1)) if len(f1) else None,
        "reciprocal_f1_p90": float(np.quantile(f1, 0.9)) if len(f1) else None,
        "reciprocal_pairs_f1_ge_0p5": int(np.count_nonzero(f1 >= 0.5)),
        "reciprocal_pairs_f1_ge_0p8": int(np.count_nonzero(f1 >= 0.8)),
        "matched_events_in_reciprocal_pairs": matched,
        "matched_fraction_reference_events": matched / len(reference["times"]),
        "matched_fraction_comparator_events": matched / len(comparator["times"]),
    }
    best = pd.concat(
        [
            ref_best.assign(direction="reference_to_comparator"),
            cmp_best.assign(direction="comparator_to_reference"),
        ],
        ignore_index=True,
        sort=False,
    )
    return summary, best, reciprocal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    reference = load_sort(config["reference_arm"])
    tolerance = int(config["matching"]["primary_tolerance_samples"])
    minimum = int(config["matching"]["minimum_candidate_matches"])
    args.output.mkdir(parents=True)
    result = {
        "schema": "em2d-cross-arm-event-overlap-result-v1",
        "status": "complete",
        "config_sha256": sha256(args.config),
        "comparisons": {},
        "interpretation": config["interpretation"],
    }
    for receipt in config["comparators"]:
        comparator = load_sort(receipt)
        if comparator["sampling_frequency"] != reference["sampling_frequency"]:
            raise RuntimeError("sampling frequencies differ")
        name = receipt["name"]
        scores = score_pairs(reference, comparator, tolerance=tolerance, minimum=minimum)
        summary, best, reciprocal = summarize(scores, reference, comparator)
        exact_scores = score_pairs(reference, comparator, tolerance=0, minimum=minimum)
        exact_summary, _, _ = summarize(exact_scores, reference, comparator)
        scores.to_csv(args.output / f"{name}_PAIR_SCORES.csv", index=False)
        best.to_csv(args.output / f"{name}_UNIT_BEST.csv", index=False)
        reciprocal.to_csv(args.output / f"{name}_RECIPROCAL.csv", index=False)
        result["comparisons"][name] = {
            "role": receipt["role"],
            "tolerance_samples": tolerance,
            "summary": summary,
            "exact_sample_sensitivity": exact_summary,
        }
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    products = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.name not in {"MANIFEST.json", "COMPLETE.json"}
    ]
    (args.output / "MANIFEST.json").write_text(
        json.dumps({"products": products}, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            {"status": "complete", "manifest_sha256": sha256(args.output / "MANIFEST.json")},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
