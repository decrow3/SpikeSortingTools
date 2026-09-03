from testing.luke_ladder_l2_baseline import _config_row


def _fake_result(good, sim_pairs, rv, edge, wall, cached):
    return {
        "wall_clock": {"pipeline_s": wall, "sort_was_cached": cached},
        "score": {
            "context": {"ks_good_count": good, "n_clusters": good * 3, "total_spikes": 1000},
            "guardrails": {
                "similar_good_good_pairs": sim_pairs,
                "similar_pairs_per_good_unit": sim_pairs / max(good, 1),
                "refractory_violation_median": rv,
                "refractory_violation_frac_over_1pct": 0.05,
                "edge_spike_fraction_40um": edge,
            },
        },
    }


def test_config_row_pulls_the_baseline_fields():
    row = _config_row(_fake_result(20, 3, 0.0012, 0.015, 41.0, False), "rescue")
    assert row["sorter"] == "rescue"
    assert row["ks_good"] == 20
    assert row["similar_good_pairs"] == 3
    assert row["similar_pairs_per_good"] == 0.15
    assert row["rv_median"] == 0.0012
    assert row["edge_spike_frac_40um"] == 0.015
    assert row["pipeline_s"] == 41.0
    assert row["sort_cached"] is False
