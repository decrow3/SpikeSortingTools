"""Combine replicated preprocessing evidence into a Luke rescue decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SHARED = Path("testing/outputs/luke_preprocessing_family_continuity")
DEFAULT_PATHOLOGICAL = Path(
    "testing/outputs/luke_preprocessing_family_continuity_pathological"
)
DEFAULT_CONDITIONING = Path(
    "testing/outputs/luke_motion_candidate_results/conditioning_replication_summary.csv"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_preprocessing_rescue_decision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--pathological", type=Path, default=DEFAULT_PATHOLOGICAL)
    parser.add_argument("--conditioning", type=Path, default=DEFAULT_CONDITIONING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_window(name: str, directory: Path) -> tuple[dict, pd.DataFrame]:
    decision = json.loads((directory / "decision.json").read_text())
    sensitivity = pd.read_csv(directory / "family_threshold_sensitivity.csv")
    if not decision["decision"].startswith("advance_single_pass"):
        raise ValueError(f"{name} does not support advancement: {decision['decision']}")
    gains = decision["ks_good_family_gain_by_agreement_threshold"]
    if not gains or min(gains.values()) <= 0:
        raise ValueError(f"{name} family gain is not positive at every threshold")
    return decision, sensitivity


def run(
    shared_dir: Path,
    pathological_dir: Path,
    conditioning_path: Path,
    output_dir: Path,
) -> dict:
    shared, shared_sensitivity = read_window("shared", shared_dir)
    pathological, pathological_sensitivity = read_window(
        "pathological", pathological_dir
    )
    conditioning = pd.read_csv(conditioning_path)
    expected_conditions = {
        "Current no motion",
        "Single KS preprocessing",
        "Current conditioning",
    }
    if not expected_conditions.issubset(set(conditioning.condition)):
        raise ValueError("Conditioning replication summary is incomplete")

    def event_recovery(condition: str, window: str) -> str:
        matches = conditioning.loc[
            (conditioning["condition"] == condition)
            & (conditioning["window"] == window),
            "recovered",
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one recovery row for {condition!r} in {window!r}; "
                f"found {len(matches)}"
            )
        return str(matches.iloc[0])

    window_rows = []
    for name, result in (("shared", shared), ("pathological", pathological)):
        window_rows.append(
            {
                "window": name,
                "duration_s": result["duration_s"],
                "current_effective_ks_good_families": result[
                    "ks_good_effective_families"
                ]["current"],
                "single_pass_effective_ks_good_families": result[
                    "ks_good_effective_families"
                ]["single_pass"],
                "family_gain": result["ks_good_effective_families"]["difference"],
                "family_gain_fraction": result["ks_good_effective_families"][
                    "relative_change"
                ],
                "min_family_gain_across_thresholds": min(
                    result["ks_good_family_gain_by_agreement_threshold"].values()
                ),
                "max_family_gain_across_thresholds": max(
                    result["ks_good_family_gain_by_agreement_threshold"].values()
                ),
                "single_pass_good_related_fraction": result[
                    "single_pass_ks_good_related_to_current_fraction"
                ],
                "conservative_independent_candidates": result[
                    "conservative_independent_candidate_count"
                ],
                "moderate_independent_candidates": result[
                    "moderate_independent_candidate_count"
                ],
            }
        )
    window_summary = pd.DataFrame(window_rows)

    combined_sensitivity = pd.concat(
        [
            shared_sensitivity.assign(window="shared"),
            pathological_sensitivity.assign(window="pathological"),
        ],
        ignore_index=True,
    )
    decision = {
        "decision": "advance_single_pass_to_full_session_validation_not_production",
        "replication_result": (
            "Single-pass preprocessing increases effective KS-good activity-family "
            "count in both prespecified windows, and the gain remains positive at "
            "every tested agreement threshold."
        ),
        "windows": window_rows,
        "event_recovery_guardrail": {
            "pathological": {
                "current": event_recovery(
                    "Current no motion", "120 s pathological"
                ),
                "single_pass": event_recovery(
                    "Single KS preprocessing", "120 s pathological"
                ),
            },
            "shared": {
                "current": event_recovery(
                    "Current conditioning", "240 s shared"
                ),
                "single_pass": event_recovery(
                    "Single KS preprocessing", "240 s shared"
                ),
            },
            "interpretation": (
                "Single-pass does not materially improve reviewed-event recovery; "
                "its value is a possible separation/effective-yield gain."
            ),
        },
        "candidate_quality_caveat": (
            "Most additional KS-good units are related to a current-sort family or "
            "are marginal/redundant. Only the declared conservative independent "
            "candidates should anchor a full-session continuity test."
        ),
        "production_gate": (
            "Do not adopt until a no-external-voltage-correction full-session run "
            "passes identical merging, unit-family continuity, event recovery, "
            "near-zero-lag duplicate, contamination, and refractory guardrails."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    window_summary.to_csv(output_dir / "window_summary.csv", index=False)
    combined_sensitivity.to_csv(
        output_dir / "combined_threshold_sensitivity.csv", index=False
    )
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    args = parse_args()
    result = run(args.shared, args.pathological, args.conditioning, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
