import numpy as np

from testing.cg_anchor_helper import (
    AnchorCandidate, build_candidate_table, force_connected_units,
    freeze_event_rows, score_anchor_reproducibility, select_depth_stratified,
)


def test_force_exclusion_is_union_of_both_native_graphs():
    a = np.eye(4, dtype=bool); b = a.copy()
    a[0, 1] = a[1, 0] = True
    b[2, 3] = b[3, 2] = True
    assert np.array_equal(force_connected_units(a, b), np.ones(4, dtype=bool))


def _candidate(unit, depth_bin, depth, score=1):
    return AnchorCandidate(unit, depth, depth_bin, 0, (0, 1, 2, 3), score,
                           500, 120, 120, 12, 12, True, "")


def test_selection_is_depth_stratified_and_deterministic():
    candidates = [_candidate(0, 0, 0, .5), _candidate(1, 0, 1, 1.0),
                  _candidate(2, 3, 30), _candidate(3, 7, 70)]
    selected = select_depth_stratified(candidates, n_select=3)
    assert [x.unit_id for x in selected] == [1, 2, 3]
    assert [x.unit_id for x in selected] == [x.unit_id for x in select_depth_stratified(candidates, 3)]


def test_masks_precede_balanced_cap_and_rows_are_repeatable():
    n = 500
    rows = np.arange(n); labels = np.zeros(n, dtype=int); times = np.arange(n) * 1000
    rest = np.ones(n, bool); bounds = np.ones(n, bool); collision = np.zeros(n, bool)
    rest[:20] = False; collision[20:40] = True
    kwargs = dict(selected_candidates=[_candidate(0, 0, 0)], event_row_ids=rows,
                  labels=labels, times_samples=times, rest_mask=rest,
                  bounds_mask=bounds, collision_mask=collision, split_sample=250000,
                  sampling_frequency=1000, seed=4)
    first = freeze_event_rows(**kwargs); second = freeze_event_rows(**kwargs)
    assert np.array_equal(first[0]["half1"]["event_row_ids"], second[0]["half1"]["event_row_ids"])
    assert np.all(first[0]["half1"]["event_row_ids"] >= 40)
    assert len(first[0]["half1"]["event_row_ids"]) == 100


def test_reproducibility_requires_complete_finite_block_draws():
    rng = np.random.default_rng(0)
    base = np.sin(np.linspace(0, np.pi * 2, 61))[:, None] * np.arange(1, 7)[None]
    first = base[None] + rng.normal(scale=.05, size=(100, 61, 6))
    second = base[None] + rng.normal(scale=.05, size=(100, 61, 6))
    blocks = np.arange(20).repeat(5)
    result = score_anchor_reproducibility(first, second, blocks_half1=blocks,
        blocks_half2=blocks + 20, support_channels=np.arange(6), n_bootstrap=40, seed=2)
    assert result["qualified"] and result["ci_lower"] >= .90
    assert result["accepted_replicates"] == result["requested_replicates"] == 40
    bad = second.copy(); bad[:, :, 0] = np.nan
    assert score_anchor_reproducibility(first, bad, blocks_half1=blocks,
        blocks_half2=blocks + 20, support_channels=np.arange(6), n_bootstrap=2)["status"] == "unresolved"


def test_candidate_table_excludes_force_group_and_uses_prefiltered_pools():
    n_units, n_events = 2, 480
    labels = np.repeat(np.arange(2), n_events // 2)
    times = np.tile(np.r_[np.arange(120) * 5000, 700000 + np.arange(120) * 5000], 2)
    templates = np.zeros((n_units, 21, 6)); templates[:, 10, 2] = -5
    counts = np.ones((n_units, 6), dtype=int) * 20
    rgeom = np.c_[np.zeros(6), np.arange(6) * 20.]
    force_a = np.eye(2, dtype=bool); force_b = force_a.copy(); force_a[0, 1] = force_a[1, 0] = True
    table = build_candidate_table(labels=labels, times_samples=times,
        rest_mask=np.ones(n_events, bool), bounds_mask=np.ones(n_events, bool),
        collision_mask=np.zeros(n_events, bool), split_sample=650000,
        sampling_frequency=1000, templates_3000=templates,
        spike_counts_by_channel_3000=counts, registered_geom=rgeom,
        physical_geom=rgeom, construction_counts=np.array([240, 240]),
        expanded_force_30000=force_a, expanded_force_3000=force_b)
    assert all(not row.eligible and "force_connected" in row.reason for row in table)
