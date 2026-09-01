import json

import numpy as np
import pandas as pd

from testing.luke_ks4_native_operator_audit import ARMS
from testing.luke_ks4_native_operator_decision import validate_and_decide


def test_decision_rejects_failed_operator_even_if_other_gates_claim_pass(tmp_path):
    root = tmp_path
    smooth = root / "waveform_depth_smoothness"
    smooth.mkdir()
    metrics = pd.DataFrame(
        {
            "arm": list(ARMS) + ["moved_no_correction"] * (2628 - len(ARMS)),
            "generator": ["stationary"] * len(ARMS)
            + ["si_kriging_p2_sigma10"] * (2628 - len(ARMS)),
            "displacement_um": [0.0] * len(ARMS)
            + ([-20.0, -10.0, -6.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 6.0, 10.0, 20.0]
               * ((2628 - len(ARMS)) // 12)
               + [-20.0] * ((2628 - len(ARMS)) % 12)),
        }
    )
    # Include all generators while retaining the exact row count.
    metrics.loc[100:199, "generator"] = "si_kriging_p2_sigma20"
    metrics.loc[200:299, "generator"] = "si_idw4"
    metrics.to_csv(root / "case_metrics.csv", index=False)
    pd.DataFrame({"x": np.arange(2610)}).to_csv(root / "pair_separability_metrics.csv", index=False)
    pd.DataFrame({"generator": ["a", "b", "c"]}).to_csv(root / "generator_gate_summary.csv", index=False)
    pd.DataFrame({"x": [1]}).to_csv(root / "separability_summary.csv", index=False)
    np.savez(root / "operator_matrices.npz", **{f"m{i}": np.eye(1) for i in range(61)})
    (root / "result.json").write_text(
        json.dumps(
            {
                "case_rows": 2628,
                "pair_rows": 2610,
                "gate": {"operator_primary_and_tax_pass": False},
                "coverage": {"edge_gate_covered": True},
                "prospective_holdout_accessed": False,
                "sorter_run": False,
            }
        )
    )
    (smooth / "result.json").write_text(
        json.dumps(
            {
                "decision": {"smoothness_supported": True},
                "prospective_holdout_accessed": False,
            }
        )
    )
    for path in [
        root / "frozen_config.json",
        root / "synthetic_validation.json",
        root / "dry_run_result.json",
        smooth / "family_manifest.csv",
        root / "operator_recovery_curves.png",
        root / "template_separability.png",
        root / "zero_shift_tax.png",
    ]:
        path.write_text("placeholder")
    decision = validate_and_decide(root)
    assert decision["advancement_authorized"] is False
    assert decision["decision"] == "do_not_advance_to_supplied_dshift_sort"
