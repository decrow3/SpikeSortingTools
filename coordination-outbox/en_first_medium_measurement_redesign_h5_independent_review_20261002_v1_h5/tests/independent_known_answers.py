"""Bounded synthetic H5 review fixtures; no real data or I/O paths."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from testing.first_medium_measurement_redesign import (
    MatchRule,
    assess_margin_rules,
    conditional_amplitude_diagnostic,
    half_open_cell_indices,
    reference_event_retention,
)


def sort(times, clusters):
    order = np.argsort(times, kind="stable")
    return {
        "st": np.asarray(times, dtype=np.int64)[order],
        "cl": np.asarray(clusters, dtype=np.int64)[order],
    }


rule = MatchRule(0, 0.1, 0.5)

ref = sort([10, 20, 30, 40], [1, 1, 1, 1])
split = sort([10, 20, 30, 40], [8, 8, 8, 9])
result = reference_event_retention(ref, split, [1], rule)
assert result["R"] == 0.75
assert result["credited_candidate_events"] == 3

ref = sort([10, 20, 30, 110], [1, 1, 1, 2])
merged = sort([10, 20, 30, 110], [8, 8, 8, 8])
result = reference_event_retention(ref, merged, [2, 1], rule)
rows = result["unit_retention"].set_index("reference_unit")
assert rows.loc[1, "retention"] == 1.0
assert rows.loc[2, "retention"] == 0.0
assert result["R"] == 0.5 and result["credited_candidate_events"] == 3

assert assess_margin_rules(0.06, 0.05, minimum_effect=0.05)["replacement_passes"] is False
assert half_open_cell_indices([0, 9, 10, 19], [0, 10, 20]).tolist() == [0, 0, 1, 1]
try:
    half_open_cell_indices([20], [0, 10, 20])
except ValueError:
    pass
else:
    raise AssertionError("final stop was not excluded")

tree = ast.parse(Path("source/testing/first_medium_measurement_redesign.py").read_text())
calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
for forbidden in ("open", "load", "save", "read_csv", "read_json", "read_bytes", "write_bytes"):
    assert forbidden not in calls

windows = pd.DataFrame({
    "cluster_id": [1, 1],
    "status": ["finite_interior", "finite_interior"],
    "start_s": [0.0, 60.0],
    "end_s": [60.0, 120.0],
    "missing_pct": [10.0, 10.0],
})
primary = pd.DataFrame({"baseline_cluster": [1], "candidate_cluster": [1]})
domain = conditional_amplitude_diagnostic(
    windows, windows, primary, [1], duration_s=60.0,
    minimum_valid_windows=2, minimum_common_time_fraction=0.5,
)
assert domain["reference_support_union_fraction_full_cohort_time"] == 2.0
assert domain["candidate_support_union_fraction_full_cohort_time"] == 2.0
assert domain["common_time_fraction_full_cohort_time"] == 2.0

ref = sort([10, 20], [1, 1])
candidate = sort([10, 20, 60, 70, 80], [8, 8, 8, 8, 8])
domain = reference_event_retention(ref, candidate, [1], rule, cell_edges_frames=[0, 50])
assert domain["R"] == 0.0
assert domain["correspondence_edges"].candidate_retention.item() == 0.4

print("independent bounded known answers and blocking probes reproduced")
