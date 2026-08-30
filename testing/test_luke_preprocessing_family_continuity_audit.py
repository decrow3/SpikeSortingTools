import numpy as np
import pandas as pd

from testing.luke_preprocessing_family_continuity_audit import (
    SortData,
    count_one_to_one_matches,
    family_assignments,
    qualifying_edges,
)


def test_one_to_one_match_does_not_reuse_spikes():
    first = np.array([100, 101, 500])
    second = np.array([100, 500])
    assert count_one_to_one_matches(first, second, tolerance=2) == 2


def _sort(condition, units):
    ids = np.arange(len(units))
    return SortData(
        condition=condition,
        unit_ids=ids,
        times_by_unit={i: np.asarray(times) for i, times in enumerate(units)},
        spike_counts=np.asarray([len(times) for times in units]),
        depths_um=np.asarray([100.0 + 20.0 * i for i in ids]),
        ks_good=np.ones(len(ids), dtype=bool),
        presence_ratio_10s=np.ones(len(ids), dtype=float),
        rate_cv_10s=np.zeros(len(ids), dtype=float),
        max_nearby_template_similarity=np.zeros(len(ids), dtype=float),
        fs=30_000.0,
    )


def test_family_graph_collapses_two_single_units_related_to_one_current_unit():
    current = _sort("current", [[100, 200, 300, 400]])
    single = _sort("single_pass", [[100, 200], [300, 400]])
    pairs = pd.DataFrame(
        [
            {
                "current_unit": 0,
                "single_pass_unit": 0,
                "agreement": 0.5,
                "matched_spikes": 30,
                "observed_expected_ratio": 10.0,
            },
            {
                "current_unit": 0,
                "single_pass_unit": 1,
                "agreement": 0.5,
                "matched_spikes": 30,
                "observed_expected_ratio": 10.0,
            },
        ]
    )
    assert len(qualifying_edges(pairs, 0.2)) == 2
    assigned = family_assignments(current, single, pairs, 0.2)
    assert assigned.family_id.nunique() == 1
    assert assigned[assigned.condition == "single_pass"].family_id.nunique() == 1
