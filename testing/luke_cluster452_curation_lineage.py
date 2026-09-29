"""Reconstruct exact pre-to-post-curation lineage for the cluster 452 case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_final_labels(
    raw_clusters: np.ndarray,
    duplicate_rows: np.ndarray,
    merge_groups: list[list[int]],
    removed_units: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror ``pipeline.curation._apply_curation_and_save`` label operations."""
    keep = np.ones(len(raw_clusters), dtype=bool)
    keep[np.asarray(duplicate_rows, dtype=int)] = False
    original = np.asarray(raw_clusters[keep], dtype=np.int64)
    working = original.copy()
    merged_units = {unit for group in merge_groups for unit in group}
    new_ids = int(working.max()) + 1 + np.arange(len(merge_groups), dtype=np.int64)
    for new_id, group in zip(new_ids, merge_groups):
        working[np.isin(working, group)] = new_id
    unique, intermediate = np.unique(working, return_inverse=True)
    remove_rows = [
        index for index, unit in enumerate(unique)
        if int(unit) in (removed_units - merged_units)
    ]
    keep_after_removal = ~np.isin(intermediate, remove_rows)
    _, final = np.unique(intermediate[keep_after_removal], return_inverse=True)
    return original[keep_after_removal], final, np.flatnonzero(keep)[keep_after_removal]


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    raw = Path(config["sorter_output"])
    curated = Path(config["curated_output"])
    decisions_path = Path(config["curation_decisions"])
    decisions = np.load(decisions_path, allow_pickle=True).item()
    raw_clusters = np.load(raw / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    raw_times = np.load(raw / "spike_times.npy", mmap_mode="r").reshape(-1)
    duplicate_rows = np.asarray(decisions["duped_spikes"], dtype=np.int64)
    groups = [list(map(int, group)) for group in decisions["merge_unit_groups"]]
    original, final, _ = reconstruct_final_labels(
        raw_clusters, duplicate_rows, groups, set(map(int, decisions["removed_units"]))
    )
    target = int(config["target_final_cluster"])
    contributors = sorted(map(int, np.unique(original[final == target])))
    source_mask = np.isin(raw_clusters, contributors)
    duplicate_mask = np.zeros(len(raw_clusters), dtype=bool)
    duplicate_mask[duplicate_rows] = True
    fs = float(config["sampling_frequency_hz"])
    rows = []
    for window in config["windows"]:
        start = int(round(float(window["start_s"]) * fs))
        stop = int(round(float(window["stop_s"]) * fs))
        present = source_mask & (raw_times >= start) & (raw_times < stop)
        total = int(np.count_nonzero(present))
        removed = int(np.count_nonzero(present & duplicate_mask))
        rows.append({**window, "precuration_events": total,
                     "duplicate_rows_removed": removed,
                     "removed_fraction": float(removed / total) if total else None})
    reference = [row["removed_fraction"] for row in rows if row["role"] == "reference"]
    failing = [row["removed_fraction"] for row in rows if row["role"] == "failing"]
    removal_increase = float(min(failing) - max(reference))
    threshold = float(config["minimum_failing_removal_fraction_increase"])
    curated_count = int(np.count_nonzero(
        np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1) == target
    ))
    expected_count = int(np.count_nonzero(final == target))
    if curated_count != expected_count:
        raise RuntimeError(f"reconstructed target count {expected_count} != curated count {curated_count}")
    result = {
        "schema": config["schema"], "status": "complete",
        "verdict": "curation_exclusion_supported" if removal_increase >= threshold else "curation_exclusion_not_supported",
        "config_sha256": sha256(config_path), "target_final_cluster": target,
        "raw_contributors": contributors,
        "merge_groups_containing_contributor": [group for group in groups if set(group) & set(contributors)],
        "raw_target_events": int(np.count_nonzero(source_mask)),
        "duplicate_rows_removed_all": int(np.count_nonzero(source_mask & duplicate_mask)),
        "curated_target_events": curated_count, "windows": rows,
        "minimum_failing_minus_maximum_reference_removed_fraction": removal_increase,
        "required_increase": threshold, "sort_launched": False,
        "raw_voltage_read": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster452_curation_lineage.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster452_curation_lineage_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
