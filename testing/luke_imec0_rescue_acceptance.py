"""Apply the prespecified imec0 rescue acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CRITERIA = Path(
    "testing/outputs/luke_full_probe_rescue_diagnostics_imec0_legacy/"
    "acceptance_criteria.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, default=CRITERIA)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def evaluate(diagnostics: Path, criteria_path: Path) -> dict:
    summary = json.loads((diagnostics / "summary.json").read_text())
    recovery = pd.read_csv(diagnostics / "event_recovery.csv").set_index("cohort")
    criteria = json.loads(criteria_path.read_text())
    gates = criteria["hard_gates"]
    observed = {
        "n_ks_good_min": summary["n_ks_good"],
        "n_spikes_max": summary["n_spikes"],
        "median_contamination_pct_good_max": summary[
            "median_contamination_pct_good"
        ],
        "median_refractory_violation_fraction_good_max": summary[
            "median_refractory_violation_fraction_good"
        ],
        "median_good_presence_fraction_300s_min": summary[
            "median_good_presence_fraction_300s"
        ],
        "fraction_good_units_presence_ge_0_9_min": summary[
            "good_units_presence_ge_0_9"
        ]
        / summary["n_ks_good"],
        "median_holdout_window_coincidence_excess_max": summary[
            "median_holdout_window_coincidence_excess"
        ],
        "sealed_holdout_all_raw_events_recovery_min": recovery.loc[
            "sealed_holdout_all_raw_events", "observed_recovery"
        ],
        "sealed_holdout_middle_depth_recovery_min": recovery.loc[
            "sealed_holdout_depth_third=2", "observed_recovery"
        ],
        "overall_edge_spike_fraction_40um_max": summary[
            "overall_edge_spike_fraction_40um"
        ],
        "nearby_similar_good_good_pairs_per_good_unit_max": summary[
            "nearby_similar_good_good_pairs"
        ]
        / summary["n_ks_good"],
    }
    results = {}
    for name, threshold in gates.items():
        comparator = ">=" if name.endswith("_min") else "<="
        passed = (
            observed[name] >= threshold
            if comparator == ">="
            else observed[name] <= threshold
        )
        results[name] = {
            "observed": observed[name],
            "comparator": comparator,
            "threshold": threshold,
            "passed": bool(passed),
        }
    all_pass = all(item["passed"] for item in results.values())
    if all_pass and summary["n_ks_good"] >= criteria["primary_target"]["n_ks_good"]:
        decision = "adopt_general_pipeline"
    elif all_pass:
        decision = "qualified_stream_policy_requires_quality_review"
    else:
        decision = "reject_universal_default"
    return {
        "decision": decision,
        "all_hard_gates_pass": all_pass,
        "criteria": str(criteria_path),
        "diagnostics": str(diagnostics),
        "gate_results": results,
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args.diagnostics, args.criteria)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
