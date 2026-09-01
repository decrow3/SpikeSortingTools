"""Validate and consolidate the Luke KS4 native-operator audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_ks4_native_operator_audit import ARMS, DEFAULT_OUTPUT


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_and_decide(root: Path = DEFAULT_OUTPUT) -> dict:
    operator = json.loads((root / "result.json").read_text())
    smoothness = json.loads((root / "waveform_depth_smoothness/result.json").read_text())
    metrics = pd.read_csv(root / "case_metrics.csv")
    pairs = pd.read_csv(root / "pair_separability_metrics.csv")
    generators = pd.read_csv(root / "generator_gate_summary.csv")
    matrices = np.load(root / "operator_matrices.npz")

    errors = []
    if operator.get("case_rows") != 2628 or len(metrics) != 2628:
        errors.append("operator case-row count is not 2628")
    if operator.get("pair_rows") != 2610 or len(pairs) != 2610:
        errors.append("pair-separability row count is not 2610")
    if set(metrics.arm) != set(ARMS):
        errors.append("six-arm coverage changed")
    if metrics.isna().any().any() or pairs.isna().any().any():
        errors.append("metric tables contain missing values")
    if set(metrics.loc[metrics.generator != "stationary", "generator"]) != {
        "si_kriging_p2_sigma10",
        "si_kriging_p2_sigma20",
        "si_idw4",
    }:
        errors.append("forward-generator coverage changed")
    moved_displacements = set(
        metrics.loc[
            metrics.arm.eq("moved_no_correction") & metrics.generator.ne("stationary"),
            "displacement_um",
        ].astype(float)
    )
    if moved_displacements != {-20.0, -10.0, -6.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0}:
        errors.append("signed moved-displacement coverage changed")
    if len(matrices.files) != 61:
        errors.append("operator matrix count is not 61")
    if any(not np.isfinite(matrices[name]).all() for name in matrices.files):
        errors.append("operator matrices contain non-finite values")
    if len(generators) != 3:
        errors.append("generator gate summary does not have three rows")
    if operator.get("prospective_holdout_accessed") is not False:
        errors.append("operator result claims prospective holdout access")
    if operator.get("sorter_run") is not False:
        errors.append("operator result claims a sorter run")
    if smoothness.get("prospective_holdout_accessed") is not False:
        errors.append("smoothness result claims prospective holdout access")
    if errors:
        raise RuntimeError("; ".join(errors))

    operator_pass = bool(operator["gate"]["operator_primary_and_tax_pass"])
    smoothness_supported = bool(smoothness["decision"]["smoothness_supported"])
    edge_covered = bool(operator["coverage"]["edge_gate_covered"])
    advancement = bool(operator_pass and smoothness_supported and edge_covered)
    if advancement:
        decision = "operator_prerequisites_pass_but_validation_ladder_still_required"
        stop_reason = None
    else:
        decision = "do_not_advance_to_supplied_dshift_sort"
        stop_reason = (
            "The native operator failed the preregistered primary and zero-tax gates. "
            "Smoothness was unvalidated and the edge challenge cannot rescue a failed "
            "necessary operator condition."
        )
    artifacts = [
        root / "frozen_config.json",
        root / "synthetic_validation.json",
        root / "dry_run_result.json",
        root / "result.json",
        root / "case_metrics.csv",
        root / "pair_separability_metrics.csv",
        root / "generator_gate_summary.csv",
        root / "separability_summary.csv",
        root / "operator_matrices.npz",
        root / "operator_recovery_curves.png",
        root / "template_separability.png",
        root / "zero_shift_tax.png",
        root / "waveform_depth_smoothness/result.json",
        root / "waveform_depth_smoothness/family_manifest.csv",
    ]
    return {
        "schema_version": "luke-ks4-native-operator-final-decision-v1",
        "decision": decision,
        "advancement_authorized": advancement,
        "stop_reason": stop_reason,
        "operator_gate": operator["gate"],
        "smoothness_decision": smoothness["decision"],
        "edge_gate_covered": edge_covered,
        "edge_challenge_status": (
            "not_run_after_failed_necessary_operator_gate" if not edge_covered else "covered"
        ),
        "supplied_dshift_sort_status": "not_run_not_authorized",
        "motion_estimator_coordinate_ladder_status": "remains_separate_active_work",
        "scope": (
            "Rejects the exact KS4 4.0.27 sig_interp=20 native operator under the "
            "three preregistered continuous forward generators; does not reject all "
            "motion handling or coordinate-only correction."
        ),
        "validated_counts": {
            "case_rows": len(metrics),
            "pair_rows": len(pairs),
            "operator_matrices": len(matrices.files),
        },
        "artifact_sha256": {str(path.relative_to(root)): file_sha256(path) for path in artifacts},
        "prospective_holdout_accessed": False,
        "sorter_run": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_and_decide(args.root)
    (args.root / "final_decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
