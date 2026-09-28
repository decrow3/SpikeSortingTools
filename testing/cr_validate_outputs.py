"""Independent structural validation of CR count-controlled RF outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cr_count_controlled_family_rf import load_context


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-force", type=Path, required=True)
    parser.add_argument("--no-force", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stimulus-cache", type=Path, required=True)
    parser.add_argument("--gaze-csv", type=Path, required=True)
    parser.add_argument("--interval-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    context = load_context(args, build_stimulus=False)
    frozen = pd.read_csv(args.output / "FROZEN_FAMILY_SUPPORT.csv")
    results = args.output / "results"
    metrics = pd.read_csv(results / "COUNT_CONTROLLED_PARENT_CHILD_RF.csv")
    completion = json.loads((results / "COMPLETE.json").read_text())
    product_checks = {
        item["path"]: (
            (results / item["path"]).stat().st_size == item["bytes"]
            and sha256(results / item["path"]) == item["sha256"]
        )
        for item in completion["products"]
    }
    selected = np.load(results / "SELECTED_EVENT_ROW_IDS.npz", allow_pickle=False)
    failures = []
    checked = 0
    for row in frozen[frozen.qualifies].itertuples(index=False):
        parent = int(row.all_force_parent)
        child = int(row.no_force_child)
        for seed in (1729, 1730, 1731):
            for fold in (0, 1):
                expected = int(getattr(row, f"thinned_fold{fold}"))
                for role in ("parent", "child"):
                    key = f"s{seed}_p{parent}_c{child}_{role}_fold{fold}"
                    ids = np.asarray(selected[key], dtype=np.int64)
                    checked += 1
                    if len(ids) != expected or len(np.unique(ids)) != len(ids):
                        failures.append(f"{key}: count/uniqueness")
                        continue
                    if role == "parent":
                        labels = context["af_labels"]
                        valid = context["af_valid"]
                        frames = context["af_frame"]
                        target = parent
                    else:
                        labels = context["nf_labels"]
                        valid = context["nf_valid"]
                        frames = context["nf_frame"]
                        target = child
                    if not np.all(labels[ids] == target):
                        failures.append(f"{key}: labels")
                    if not np.all(valid[ids]):
                        failures.append(f"{key}: response support")
                    if not np.all(context["frame_fold"][frames[ids]] == fold):
                        failures.append(f"{key}: fold")
    sta = np.load(results / "BOTH_CHILD_FOLD_STAS.npz", allow_pickle=False)
    audit = {
        "status": "pass" if not failures and all(product_checks.values()) else "fail",
        "all_result_product_hashes_match": all(product_checks.values()),
        "selected_arrays_checked": checked,
        "selected_array_failures": failures,
        "families": int(frozen.all_force_parent.nunique()),
        "comparisons": int(len(frozen)),
        "qualifying_comparisons": int(frozen.qualifies.sum()),
        "metric_rows": int(len(metrics)),
        "metric_rows_expected": int(frozen.qualifies.sum() * 3),
        "selected_array_count": len(selected.files),
        "selected_array_count_expected": int(frozen.qualifies.sum() * 3 * 2 * 2),
        "sta_array_count": len(sta.files),
        "sta_array_count_expected": int(
            frozen.groupby("all_force_parent").qualifies.all().sum() * 3 * 2 * 2 * 2
        ),
        "outer_holdout_opened": False,
    }
    audit["status"] = "pass" if (
        audit["status"] == "pass"
        and audit["metric_rows"] == audit["metric_rows_expected"]
        and audit["selected_array_count"] == audit["selected_array_count_expected"]
        and audit["sta_array_count"] == audit["sta_array_count_expected"]
    ) else "fail"
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
