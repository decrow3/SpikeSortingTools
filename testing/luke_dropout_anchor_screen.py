"""Screen every cached rescue-good dropout case for independent event loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_cluster452_identity_replay import (
    extract_candidates,
    interval,
    match_summary,
    select_anchor,
    sha256,
    template_depths,
)


def case_decision(anchor_pass: bool, reference_fraction: float, failing_fraction: float,
                  minimum_loss: float) -> tuple[float, bool]:
    loss = float(reference_fraction - failing_fraction)
    return loss, bool(anchor_pass and loss >= minimum_loss)


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    cases_path = Path(config["cases_csv"])
    if sha256(cases_path) != config["cases_csv_sha256"]:
        raise RuntimeError("candidate case table hash differs from frozen contract")
    cases = pd.read_csv(cases_path)
    rule = config["case_rule"]
    cases = cases.loc[
        cases.sort_id.eq(rule["sort_id"]) & cases.KSLabel.astype(str).eq(rule["KSLabel"])
    ].sort_values("cluster_id")
    if cases.empty:
        raise RuntimeError("frozen case rule selected no rows")

    rescue_root = Path(config["inputs"]["rescue_curated"])
    legacy_root = Path(config["inputs"]["legacy_curated"])
    rescue_depth = template_depths(rescue_root)
    legacy_depth = template_depths(legacy_root)
    fs = float(config["sampling_frequency_hz"])
    duration_frames = int(round(float(config["duration_s"]) * fs))
    tolerance = int(round(float(config["tolerance_ms"]) * 1e-3 * fs))
    radius = float(config["maximum_depth_difference_um"])
    minimum_loss = float(config["material_independent_loss_fraction"])
    rows = []
    for case in cases.itertuples(index=False):
        target = int(case.cluster_id)
        target_depth = float(rescue_depth[target])
        rescue_ids = np.flatnonzero(np.abs(rescue_depth - target_depth) <= radius)
        legacy_ids = np.flatnonzero(np.abs(legacy_depth - target_depth) <= radius)
        rescue_times, rescue_labels = extract_candidates(rescue_root, rescue_ids)
        legacy_times, legacy_labels = extract_candidates(legacy_root, legacy_ids)
        ref_start, ref_stop = int(round(case.start_s * fs)), int(round(case.reference_end_s * fs))
        fail_start, fail_stop = int(round(case.failing_start_s * fs)), int(round(case.end_s * fs))
        rescue_ref = interval(rescue_times, ref_start, ref_stop)
        anchor_rows = []
        for legacy_id in legacy_ids:
            anchor_all = legacy_times[legacy_labels == legacy_id]
            anchor = anchor_all[interval(anchor_all, ref_start, ref_stop)]
            observed = match_summary(
                anchor, rescue_times[rescue_ref], rescue_labels[rescue_ref], tolerance
            )
            null = []
            for shift_s in config["circular_shift_null_s"]:
                shifted = np.sort((anchor + int(round(float(shift_s) * fs))) % duration_frames)
                null.append(match_summary(shifted, rescue_times, rescue_labels, tolerance)["matched_fraction"])
            null_median = float(np.median(null))
            dominant = bool(observed["label_counts"] and
                int(max(observed["label_counts"], key=observed["label_counts"].get)) == target)
            anchor_rows.append({
                "legacy_cluster": int(legacy_id),
                "reference_anchor_events": observed["anchor_events"],
                "reference_matched_fraction": observed["matched_fraction"],
                "reference_target_fraction": observed["label_fractions"].get(str(target), 0.0),
                "null_median_fraction": null_median,
                "excess_over_null": observed["matched_fraction"] - null_median,
                "target_is_dominant": dominant,
            })
        selected, anchor_pass = select_anchor(anchor_rows, config["anchor_rule"])
        anchor_all = legacy_times[legacy_labels == int(selected["legacy_cluster"])]
        reference_anchor = anchor_all[interval(anchor_all, ref_start, ref_stop)]
        failing_anchor = anchor_all[interval(anchor_all, fail_start, fail_stop)]
        rescue_fail = interval(rescue_times, fail_start, fail_stop)
        reference_match = match_summary(
            reference_anchor, rescue_times[rescue_ref], rescue_labels[rescue_ref], tolerance
        )
        failing_match = match_summary(
            failing_anchor, rescue_times[rescue_fail], rescue_labels[rescue_fail], tolerance
        )
        independent_loss, advances = case_decision(
            anchor_pass, reference_match["matched_fraction"], failing_match["matched_fraction"], minimum_loss
        )
        rows.append({
            "cluster_id": target, "fitted_missingness_change_pp": float(case.difference_pp),
            "reference_missing_pct": float(case.reference_missing_pct),
            "failing_missing_pct": float(case.failing_missing_pct),
            "selected_legacy_anchor": int(selected["legacy_cluster"]),
            "anchor_pass": bool(anchor_pass),
            "reference_anchor_events": reference_match["anchor_events"],
            "failing_anchor_events": failing_match["anchor_events"],
            "reference_any_rescue_fraction": reference_match["matched_fraction"],
            "failing_any_rescue_fraction": failing_match["matched_fraction"],
            "independent_loss_fraction": independent_loss,
            "reference_target_fraction": reference_match["label_fractions"].get(str(target), 0.0),
            "failing_target_fraction": failing_match["label_fractions"].get(str(target), 0.0),
            "null_median_fraction": selected["null_median_fraction"],
            "reference_excess_over_null": selected["excess_over_null"],
            "advances_to_intervention_localization": advances,
        })
    table = pd.DataFrame(rows)
    advancing = table.loc[table.advances_to_intervention_localization, "cluster_id"].astype(int).tolist()
    result = {
        "schema": config["schema"], "status": "complete",
        "config_sha256": sha256(config_path), "case_count": int(len(table)),
        "anchor_pass_count": int(table.anchor_pass.sum()),
        "advancing_cluster_ids": advancing,
        "verdict": "corroborated_cases_found" if advancing else "no_cached_good_case_has_material_independent_loss",
        "sort_launched": False, "raw_voltage_read": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    table.to_csv(output / "case_screen.csv", index=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_dropout_anchor_screen.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_dropout_anchor_screen_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
