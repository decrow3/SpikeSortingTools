"""Independent bounded H5 fixtures for the two repaired temporal-domain findings."""

import numpy as np
import pandas as pd

import testing.first_medium_measurement_redesign as module
from testing.first_medium_measurement_redesign import (
    MatchRule,
    assess_margin_rules,
    conditional_amplitude_diagnostic,
    reference_event_retention,
    retention_difference,
)


def sort(times, clusters):
    order = np.argsort(times, kind="stable")
    return {"st": np.asarray(times, dtype=np.int64)[order],
            "cl": np.asarray(clusters, dtype=np.int64)[order]}


def windows(cluster, starts, stops):
    return pd.DataFrame({
        "cluster_id": [cluster] * len(starts),
        "status": ["finite_interior"] * len(starts),
        "start_s": starts, "end_s": stops,
        "missing_pct": [10.0] * len(starts),
    })


rule = MatchRule(0, 0.1, 0.5)
primary = pd.DataFrame({"baseline_cluster": [1], "candidate_cluster": [11]})

reference = pd.concat([
    windows(1, [0.0, 10.0], [10.0, 20.0]),
    windows(999, [19.0], [21.0]),
], ignore_index=True)
candidate = windows(11, [0.0, 10.0], [10.0, 20.0])
try:
    conditional_amplitude_diagnostic(reference, candidate, primary, [1], duration_s=20.0)
except ValueError as error:
    assert "wholly inside" in str(error)
else:
    raise AssertionError("out-of-domain noncohort finite window accepted")

bad_tables = [
    pd.DataFrame({"cluster_id": [1, 1], "status": ["finite_interior"] * 2,
                  "start_s": [0.0, 9.0], "end_s": [10.0, 12.0],
                  "missing_pct": [1.0, 1.0]}),
    pd.DataFrame({"cluster_id": [1], "status": ["finite_interior"],
                  "start_s": [0.0], "end_s": [np.nan], "missing_pct": [1.0]}),
]
for bad in bad_tables:
    try:
        conditional_amplitude_diagnostic(
            bad, candidate, primary, [1], duration_s=20.0,
            minimum_valid_windows=1, minimum_common_time_fraction=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed amplitude table accepted")

for duration in (np.nan, np.inf, 0.0, -0.5):
    try:
        conditional_amplitude_diagnostic(
            candidate, candidate,
            pd.DataFrame({"baseline_cluster": [11], "candidate_cluster": [11]}),
            [11], duration_s=duration, minimum_valid_windows=1,
            minimum_common_time_fraction=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("bad duration accepted")

coverage = conditional_amplitude_diagnostic(
    windows(1, [0.0, 10.0], [10.0, 20.0]), candidate, primary, [1],
    duration_s=20.0, minimum_valid_windows=2, minimum_common_time_fraction=1.0)
for name in ("supported_unit_fraction_full_cohort",
             "reference_support_union_fraction_full_cohort_time",
             "candidate_support_union_fraction_full_cohort_time",
             "common_time_fraction_full_cohort_time"):
    assert 0.0 <= coverage[name] <= 1.0
assert coverage["reference_support_union_fraction_full_cohort_time"] == 1.0

called = False
accepted = module.correspondence


def sentinel(*args, **kwargs):
    global called
    called = True
    raise AssertionError("matcher ran before domain validation")


module.correspondence = sentinel
try:
    try:
        reference_event_retention(
            sort([5, 15], [1, 1]), sort([5, 15, 50], [8, 8, 99]), [1], rule,
            cell_edges_frames=[0, 25, 50])
    except ValueError:
        pass
    else:
        raise AssertionError("exact-stop candidate event accepted")
    assert called is False
finally:
    module.correspondence = accepted

try:
    reference_event_retention(
        sort([-1, 5, 15], [99, 1, 1]), sort([5, 15], [8, 8]), [1], rule,
        cell_edges_frames=[0, 25, 50])
except ValueError:
    pass
else:
    raise AssertionError("negative noncohort REF event accepted")

valid = reference_event_retention(
    sort([0, 25, 49], [1, 1, 1]), sort([0, 25, 49], [8, 8, 8]), [1], rule,
    cell_edges_frames=[0, 25, 50])
assert valid["R"] == 1.0
assert valid["cell_diagnostics"].reference_events.tolist() == [1, 2]

ref = sort([10, 20, 30, 110, 120, 130], [1, 1, 1, 2, 2, 2])
existing = sort([10, 20, 110], [11, 11, 12])
repaired = sort([10, 20, 30, 110, 120, 130], [21, 21, 21, 22, 22, 22])
delta = retention_difference(ref, existing, repaired, [1, 2], rule,
                             bootstrap_replicates=500, bootstrap_seed=20261002)
assert delta["R_existing"] == (2 / 3 + 0.0) / 2
assert delta["R_repaired"] == 1.0
assert np.isclose(delta["DeltaR"], 2 / 3)
assert delta["bootstrap"]["kind"] == "paired REF-unit composition bootstrap"
assert assess_margin_rules(0.06, 0.05, minimum_effect=0.05)["replacement_passes"] is False

print("independent workstream-B repair adversarial checks passed")
