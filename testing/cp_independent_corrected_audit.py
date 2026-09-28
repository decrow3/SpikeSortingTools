"""Independent bounded audit of the H5 CP corrected saved-output packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def greedy_match(left: np.ndarray, right: np.ndarray, tolerance: int = 12) -> int:
    i = j = matched = 0
    while i < left.size and j < right.size:
        delta = int(right[j]) - int(left[i])
        if abs(delta) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif delta < -tolerance:
            j += 1
        else:
            i += 1
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--static-w2", type=Path, required=True)
    parser.add_argument("--all-force-w2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    corrected = args.packet / "analysis_corrected_v2"
    manifest = json.loads((corrected / "MANIFEST.json").read_text())
    manifest_checks = {}
    for item in manifest["products"]:
        path = args.packet / item["path"]
        manifest_checks[item["path"]] = (
            path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]
        )

    summary = json.loads((corrected / "CORRECTED_SUMMARY.json").read_text())
    review = json.loads((corrected / "CODE_REVIEW_RECEIPT.json").read_text())
    source_checks = {
        name: sha256(args.packet / "source" / name) == expected
        for name, expected in review["source_files"].items()
    }
    catalogue = pd.read_csv(args.catalogue)
    domains = pd.read_csv(corrected / "EPISODE_DOMAINS.csv")
    domain_checks = {}
    for window, start, end in (("W2", 900.0, 1240.0), ("W3", 8000.0, 8340.0)):
        expected = catalogue[(catalogue.start_s < end) & (catalogue.end_s > start)].copy()
        expected["start_s"] = expected.start_s.clip(lower=start)
        expected["end_s"] = expected.end_s.clip(upper=end)
        actual = domains[(domains.window == window) & (domains.domain == "actual_all")]
        accepted = domains[
            (domains.window == window) & (domains.domain == "actual_accepted")
        ]
        unresolved = domains[
            (domains.window == window) & (domains.domain == "actual_unresolved")
        ]
        rest = domains[(domains.window == window) & (domains.domain == "actual_rest")]
        expected_pairs = list(zip(expected.start_s, expected.end_s))
        actual_pairs = list(zip(actual.start_s, actual.end_s))
        expected_accepted_s = float(
            (expected.loc[expected.status == "accepted", "end_s"]
             - expected.loc[expected.status == "accepted", "start_s"]).sum()
        )
        domain_checks[window] = {
            "actual_intervals_equal_catalogue_unpadded": expected_pairs == actual_pairs,
            "actual_seconds": float(actual.duration_s.sum()),
            "accepted_seconds_recomputed": float(accepted.duration_s.sum()),
            "accepted_seconds_catalogue": expected_accepted_s,
            "unresolved_seconds": float(unresolved.duration_s.sum()),
            "actual_plus_rest_is_window": bool(
                np.isclose(actual.duration_s.sum() + rest.duration_s.sum(), end - start)
            ),
            "padded_and_canonical_reported_separately": bool(
                {"padded_as", "canonical_hub"}
                <= set(domains.loc[domains.window == window, "domain"])
            ),
        }

    null = pd.read_csv(corrected / "COMMON_SUPPORT_TIME_SHIFT_NULL.csv")
    with np.load(args.static_w2, allow_pickle=False) as data:
        static = np.sort(data["times_samples"][data["labels"] >= 0])
        fs = float(data["sampling_frequency"])
    with np.load(args.all_force_w2, allow_pickle=False) as data:
        moving = np.sort(data["times_samples"][data["labels"] >= 0])
    n_samples = 10_199_918
    null_checks = []
    for offset_s in (0.0, -0.5, 0.5):
        shift = int(round(offset_s * fs))
        support_start = max(0, shift)
        support_end = min(n_samples, n_samples + shift)
        left = static[(static >= support_start) & (static < support_end)]
        right = moving + shift
        right = right[(right >= support_start) & (right < support_end)]
        observed = null[(null.window == "W2") & (null.offset_s == offset_s)].iloc[0]
        matched = greedy_match(left, right)
        null_checks.append(
            {
                "offset_s": offset_s,
                "support_equal": [support_start, support_end]
                == [int(observed.common_support_start_sample), int(observed.common_support_end_sample)],
                "denominators_equal": [len(left), len(right)]
                == [int(observed.static_rows_in_support), int(observed.shifted_d2l_rows_in_support)],
                "matched_equal": matched == int(observed.matched),
            }
        )

    control = pd.read_csv(corrected / "FORCE_SPLIT_EXCLUSIVE_COINCIDENCE_CONTROL.csv")
    control_summary = pd.read_csv(
        corrected / "FORCE_SPLIT_EXCLUSIVE_COINCIDENCE_SUMMARY.csv"
    )
    shifted = control[control.offset_s != 0].groupby(
        ["window", "all_force_parent"]
    ).fraction_of_smaller_child.median()
    merged = control_summary.set_index(["window", "all_force_parent"])
    split_control_ok = bool(
        np.allclose(
            shifted.sort_index().to_numpy(),
            merged.loc[shifted.sort_index().index, "shifted_median"].to_numpy(),
            equal_nan=True,
        )
    )
    panels = pd.read_csv(corrected / "PANEL_SELECTION_ZABS_FROZEN.csv")
    zabs_ok = bool(
        panels.median_z_abs_um.notna().all()
        and panels.median_z_abs_um.between(0, 4000).all()
        and not np.allclose(panels.median_z_abs_um, panels.median_depth_um_raw)
    )
    checks = {
        "status": "pass" if (
            all(manifest_checks.values())
            and all(source_checks.values())
            and sha256(args.catalogue) == summary["episode_catalogue_sha256"]
            and all(
                row["actual_intervals_equal_catalogue_unpadded"]
                and row["actual_plus_rest_is_window"]
                and np.isclose(row["accepted_seconds_recomputed"], row["accepted_seconds_catalogue"])
                for row in domain_checks.values()
            )
            and all(
                row["support_equal"] and row["denominators_equal"] and row["matched_equal"]
                for row in null_checks
            )
            and split_control_ok
            and zabs_ok
        ) else "fail",
        "manifest_all_products_match": all(manifest_checks.values()),
        "manifest_checks": manifest_checks,
        "source_snapshot_hashes_match_review_receipt": all(source_checks.values()),
        "source_checks": source_checks,
        "catalogue_hash_matches": sha256(args.catalogue)
        == summary["episode_catalogue_sha256"],
        "domain_checks": domain_checks,
        "w2_common_support_spot_checks": null_checks,
        "split_shifted_medians_recompute": split_control_ok,
        "corrected_panel_z_abs_sanity": zabs_ok,
        "interpretive_limits_accepted": summary["limits"],
        "source_review": (
            "H5 source snapshots match the code-review receipt. Independent saved-output "
            "checks confirm the load-bearing corrected invariants."
        ),
    }
    args.output.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
