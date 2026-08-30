"""Synthesize Luke claim-mask evidence into a rescue decision.

The short-window sweep is the causal comparison: every Kilosort run uses the
same cached recording and differs only in the cross-peel claim parameters.  The
existing full-session comparison is retained as an external replication of the
production setting, but is labelled separately because it includes curation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SHORT_WINDOW = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/"
    "claimmask_window_sweep/claimmask_window_sweep_scores.csv"
)
DEFAULT_FULL_SESSION = Path(
    "testing/outputs/luke_multichannel_event_validation/imec1/"
    "sort_variant_event_recovery.csv"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_claimmask_rescue_audit")

EXPECTED_WINDOWS = {"shared_template": 35, "registration_outlier": 27}
EXPECTED_SETTINGS = {
    "claim_off",
    "claim_ms0p1_um25",
    "claim_ms0p1_um50",
    "claim_ms0p25_um25",
    "claim_ms0p25_um50",
    "claim_ms0p25_um75",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-window", type=Path, default=DEFAULT_SHORT_WINDOW)
    parser.add_argument("--full-session", type=Path, default=DEFAULT_FULL_SESSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_short_window(scores: pd.DataFrame) -> pd.DataFrame:
    required = {
        "window",
        "setting",
        "claim_ms",
        "claim_um",
        "population",
        "n_events",
        "observed_recovery",
        "n_spikes",
        "n_units",
        "n_ks_good",
        "median_contamination_pct",
        "cross_unit_near_coincident_fraction",
        "median_unit_refractory_violation_fraction",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Short-window scores are missing columns: {sorted(missing)}")
    if scores.duplicated(["window", "setting", "population"]).any():
        raise ValueError("Duplicate window/setting/population rows detected")

    visual = scores[scores.population == "visual_neural_unmatched"].copy()
    if set(visual.window) != set(EXPECTED_WINDOWS):
        raise ValueError(f"Unexpected windows: {sorted(set(visual.window))}")
    if set(visual.setting) != EXPECTED_SETTINGS:
        raise ValueError(f"Unexpected settings: {sorted(set(visual.setting))}")
    if len(visual) != len(EXPECTED_WINDOWS) * len(EXPECTED_SETTINGS):
        raise ValueError(f"Expected 12 visual-neural rows, found {len(visual)}")
    for window, n_events in EXPECTED_WINDOWS.items():
        observed = set(visual.loc[visual.window == window, "n_events"].astype(int))
        if observed != {n_events}:
            raise ValueError(f"Unexpected event count for {window}: {sorted(observed)}")
    numeric = required - {"window", "setting", "population"}
    if visual[list(numeric)].isna().any().any():
        raise ValueError("Visual-neural decision rows contain missing numeric values")
    return visual


def summarize_short_window(scores: pd.DataFrame) -> pd.DataFrame:
    visual = validate_short_window(scores)
    rows = []
    for window, subset in visual.groupby("window", sort=False):
        baseline_rows = subset[subset.setting == "claim_off"]
        if len(baseline_rows) != 1:
            raise ValueError(f"Expected one claim-off baseline for {window}")
        baseline = baseline_rows.iloc[0]
        for _, row in subset.iterrows():
            rows.append(
                {
                    "window": window,
                    "setting": row.setting,
                    "claim_ms": float(row.claim_ms),
                    "claim_um": float(row.claim_um),
                    "n_events": int(row.n_events),
                    "recovered_events": int(round(row.observed_recovery * row.n_events)),
                    "observed_recovery": float(row.observed_recovery),
                    "recovery_change_pp": 100.0
                    * (row.observed_recovery - baseline.observed_recovery),
                    "near_coincident_fraction": float(
                        row.cross_unit_near_coincident_fraction
                    ),
                    "near_coincident_relative_change": float(
                        row.cross_unit_near_coincident_fraction
                        / baseline.cross_unit_near_coincident_fraction
                        - 1.0
                    ),
                    "spike_retention": float(row.n_spikes / baseline.n_spikes),
                    "n_units": int(row.n_units),
                    "n_ks_good": int(row.n_ks_good),
                    "ks_good_change": int(row.n_ks_good - baseline.n_ks_good),
                    "median_contamination_pct": float(row.median_contamination_pct),
                    "contamination_change_pp": float(
                        row.median_contamination_pct
                        - baseline.median_contamination_pct
                    ),
                    "median_refractory_violation_fraction": float(
                        row.median_unit_refractory_violation_fraction
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["window", "claim_ms", "claim_um"])


def validate_full_session(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "variant",
        "population",
        "n_events",
        "observed_recovery",
        "recovery_above_null",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Full-session scores are missing columns: {sorted(missing)}")
    visual = rows[rows.population == "visual_neural_unmatched"].copy()
    selected = visual[visual.variant.isin(["patched", "dredge"])]
    if set(selected.variant) != {"patched", "dredge"} or len(selected) != 2:
        raise ValueError("Need exactly one patched and one dredge full-session row")
    if set(selected.n_events.astype(int)) != {62}:
        raise ValueError("Full-session comparison must use the same 62-event cohort")
    return selected.sort_values("variant")


def build_decision(short: pd.DataFrame, full: pd.DataFrame) -> dict:
    nonzero = short[short.setting != "claim_off"]
    recovery_deltas = nonzero.pivot(
        index="setting", columns="window", values="recovery_change_pp"
    )
    all_nonzero_reduce_both = bool((recovery_deltas < 0).all(axis=1).all())

    mild = short[short.setting == "claim_ms0p1_um25"].copy()
    current = short[short.setting == "claim_ms0p25_um75"].copy()
    full_index = full.set_index("variant")
    full_ratio = float(
        full_index.loc["patched", "observed_recovery"]
        / full_index.loc["dredge", "observed_recovery"]
    )
    return {
        "decision": "reject_current_claim_mask_for_luke_rescue",
        "production_setting": {"claim_ms": 0.25, "claim_um": 75.0},
        "causal_short_window_result": {
            "all_nonzero_masks_reduce_visual_neural_recovery_in_both_windows": all_nonzero_reduce_both,
            "production_recovery_change_pp_by_window": {
                row.window: float(row.recovery_change_pp)
                for row in current.itertuples()
            },
            "production_spike_retention_by_window": {
                row.window: float(row.spike_retention) for row in current.itertuples()
            },
            "mildest_mask_recovery_change_pp_by_window": {
                row.window: float(row.recovery_change_pp) for row in mild.itertuples()
            },
            "mildest_mask_near_coincident_relative_change_by_window": {
                row.window: float(row.near_coincident_relative_change)
                for row in mild.itertuples()
            },
        },
        "full_session_replication": {
            "population": "62 visually neural unmatched events",
            "patched_recovery": float(
                full_index.loc["patched", "observed_recovery"]
            ),
            "claim_off_dredge_recovery": float(
                full_index.loc["dredge", "observed_recovery"]
            ),
            "patched_to_claim_off_recovery_ratio": full_ratio,
            "caveat": "This comparison includes downstream curation; use it as replication, not the primary causal estimate.",
        },
        "interpretation": (
            "The current 0.25 ms/75 um cross-peel mask removes duplicate-like "
            "activity by suppressing a large fraction of all detections, including "
            "reviewed neural events. It is not a rescue setting for Luke."
        ),
        "next_test": (
            "If claim masking is revisited, test only the mild 0.10 ms/25 um "
            "setting on the no-external-voltage-correction baseline, and require "
            "event-level non-inferiority plus collision and unit-family continuity checks."
        ),
        "scope_caveat": (
            "The causal sweep used the already DREDGE-warped binary. It rejects the "
            "current mask on that input but does not establish whether a milder, "
            "template-family-aware mask could help the no-warp production baseline."
        ),
    }


def run(short_path: Path, full_path: Path, output_dir: Path) -> dict:
    short_source = pd.read_csv(short_path)
    full_source = pd.read_csv(full_path)
    short = summarize_short_window(short_source)
    full = validate_full_session(full_source)
    decision = build_decision(short, full)
    output_dir.mkdir(parents=True, exist_ok=True)
    short.to_csv(output_dir / "short_window_claimmask_summary.csv", index=False)
    full.to_csv(output_dir / "full_session_claimmask_summary.csv", index=False)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    args = parse_args()
    decision = run(args.short_window, args.full_session, args.output_dir)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
