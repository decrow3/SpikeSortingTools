import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("run_audit.py")
SPEC = importlib.util.spec_from_file_location("noise_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_status_preserves_undefined():
    got = audit.status(pd.Series([np.nan, 4.999, 5.0])).tolist()
    assert got == ["undefined", "finite_pass", "finite_fail"]


def test_standardization_detects_pure_composition_gap():
    rows = []
    for arm, counts in {"REF": {"s0": (90, 0), "s1": (10, 10)}, "A": {"s0": (10, 0), "s1": (90, 90)}}.items():
        for stratum, (n, failures) in counts.items():
            rows.extend({"arm": arm, "factor": stratum, "noise_failure": i < failures, "noise_undefined": False} for i in range(n))
    got = audit.standardized_gap(pd.DataFrame(rows), "factor", ["s0", "s1"])
    assert np.isclose(got["raw_arm_gap_A_minus_REF"], .8)
    assert np.isclose(got["standardized_arm_gap_A_minus_REF"], 0.0)
    assert np.isclose(got["absolute_gap_reduction_fraction"], 1.0)
    assert got["meets_frozen_primary_composition_rule"]


def test_standardization_does_not_erase_within_stratum_gap():
    rows = []
    for arm, failures in {"REF": 20, "A": 60}.items():
        rows.extend({"arm": arm, "factor": "s", "noise_failure": i < failures, "noise_undefined": False} for i in range(100))
    got = audit.standardized_gap(pd.DataFrame(rows), "factor", ["s"])
    assert np.isclose(got["raw_arm_gap_A_minus_REF"], .4)
    assert np.isclose(got["standardized_arm_gap_A_minus_REF"], .4)
    assert not got["meets_frozen_primary_composition_rule"]
