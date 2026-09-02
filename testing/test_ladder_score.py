import numpy as np
import pandas as pd
import pytest

from testing.ladder_score import (
    NOT_ENDPOINTS,
    SCORE_SCHEMA,
    coincident_mask,
    ground_truth_scores,
    guardrails,
    score_sort,
    symmetric_agreement,
    window_reference_sort,
)
from testing.luke_rescue_unique_units_audit import load_sort

FS = 30_000.0


def _write_sort(
    d,
    spike_times,
    spike_clusters,
    good_ids,
    *,
    similar=None,
    depths=None,
    spike_depths=None,
):
    d.mkdir(parents=True, exist_ok=True)
    st = np.asarray(spike_times, dtype=np.int64)
    cl = np.asarray(spike_clusters, dtype=np.int64)
    order = np.argsort(st, kind="stable")
    np.save(d / "spike_times.npy", st[order])
    np.save(d / "spike_clusters.npy", cl[order])
    n = int(cl.max()) + 1
    pd.DataFrame({
        "cluster_id": np.arange(n),
        "KSLabel": ["good" if c in set(good_ids) else "mua" for c in range(n)],
    }).to_csv(d / "cluster_KSLabel.tsv", sep="\t", index=False)

    n_chan = 8
    chan_y = np.linspace(0.0, 700.0, n_chan)
    np.save(
        d / "channel_positions.npy",
        np.column_stack([np.zeros(n_chan), chan_y]).astype(np.float64),
    )
    sim = np.eye(n, dtype=np.float32) if similar is None else np.asarray(
        similar, dtype=np.float32
    )
    np.save(d / "similar_templates.npy", sim)
    tmpl = np.zeros((n, 10, n_chan), dtype=np.float32)
    dep = depths if depths is not None else [0] * n
    for i in range(n):
        tmpl[i, 5, int(dep[i])] = -5.0  # peak channel = dep[i]
    np.save(d / "templates.npy", tmpl)
    if spike_depths is None:
        spike_depths = np.full(st.size, 350.0)
    np.save(
        d / "spike_positions.npy",
        np.column_stack([np.zeros(st.size), np.asarray(spike_depths)[order]]).astype(
            np.float32
        ),
    )
    (d / "params.py").write_text(f"sample_rate = {FS}\n")
    return d


def test_coincident_mask_matches_within_tolerance_only():
    a = np.array([100, 200, 300, 400])
    b = np.array([102, 250, 401])
    m = coincident_mask(a, b, tol=5)
    assert list(m) == [True, False, False, True]
    assert not coincident_mask(a, np.array([]), tol=5).any()


def test_ground_truth_perfect_recovery_is_counted():
    truth_st = np.arange(100, 100_000, 500)  # 200 spikes
    st = np.concatenate([truth_st, np.arange(50, 100_000, 900)])  # +background unit
    cl = np.concatenate([np.zeros(truth_st.size), np.ones(st.size - truth_st.size)])
    sort = {
        "st": st.astype(np.int64),
        "cl": cl.astype(np.int64),
        "label": {0: "good", 1: "mua"},
        "good": {0},
    }
    out = ground_truth_scores(sort, {"u1": truth_st}, FS, duration_s=4.0)
    u = out["units"][0]
    assert u["accuracy"] == 1.0
    assert u["split"] is False and u["merge"] is False
    assert u["recovered"] is True
    assert out["headline_units_recovered"] == 1


def test_ground_truth_split_is_not_recovered():
    truth_st = np.arange(0, 120_000, 200)  # 600 spikes over 4 s
    half = truth_st.size // 2
    st = truth_st.copy()
    cl = np.zeros(truth_st.size, dtype=np.int64)
    cl[half:] = 1  # same train, two output clusters
    sort = {"st": st, "cl": cl, "label": {0: "good", 1: "good"}, "good": {0, 1}}
    out = ground_truth_scores(sort, {"u1": truth_st}, FS, duration_s=4.0)
    u = out["units"][0]
    assert u["n_output_units_capturing"] == 2
    assert u["split"] is True
    assert u["recovered"] is False
    assert out["headline_units_recovered"] == 0


def test_ground_truth_merge_is_flagged_for_both_trains():
    a = np.arange(0, 120_000, 400)
    b = np.arange(200, 120_000, 400)
    st = np.concatenate([a, b])
    cl = np.zeros(st.size, dtype=np.int64)  # one output unit swallows both
    sort = {"st": st.astype(np.int64), "cl": cl, "label": {0: "good"}, "good": {0}}
    out = ground_truth_scores(sort, {"a": a, "b": b}, FS, duration_s=4.0)
    assert all(u["merge"] for u in out["units"])
    assert out["headline_units_recovered"] == 0


def test_symmetric_agreement_reports_both_sides_and_absent(tmp_path):
    shared = np.arange(1000, 200_000, 500)
    cand_unique = np.arange(1200, 200_000, 1700)
    ref_unique = np.arange(300_000, 400_000, 500)  # entirely after any cand spike

    cand_st = np.concatenate([shared, cand_unique])
    cand_cl = np.concatenate([np.zeros(shared.size), np.ones(cand_unique.size)])
    cand = _write_sort(tmp_path / "cand", cand_st, cand_cl, good_ids={0, 1})

    ref_st = np.concatenate([shared, ref_unique])
    ref_cl = np.concatenate([np.zeros(shared.size), np.ones(ref_unique.size)])
    ref = _write_sort(tmp_path / "ref", ref_st, ref_cl, good_ids={0, 1})

    agree = symmetric_agreement(load_sort(cand), load_sort(ref))
    assert agree["matched_good_pairs"] == 1  # the shared train
    assert agree["gained_good"] == 1 and agree["lost_good"] == 1
    assert agree["net_good"] == 0
    # ref unit 1 has no counterpart in cand at all -> absent at detection
    assert agree["lost_absent_at_detection"] == 1


def test_window_reference_sort_rebases_frames_and_drops_empty_units():
    ref = {
        "st": np.array([100, 5_000, 10_500, 11_000, 90_000], dtype=np.int64),
        "cl": np.array([0, 0, 1, 1, 2], dtype=np.int64),
        "label": {0: "good", 1: "good", 2: "good"},
        "good": {0, 1, 2},
    }
    w = window_reference_sort(ref, start_frame=10_000, end_frame=20_000)
    assert set(w["good"]) == {1}  # unit 0 before, unit 2 after the window
    assert w["st"].tolist() == [500, 1000]  # re-based to window start


def test_symmetric_agreement_survives_zero_matches(tmp_path):
    c_st = np.arange(0, 50_000, 250)
    r_st = np.arange(50_000, 100_000, 250)  # disjoint time range
    cand = _write_sort(
        tmp_path / "c", c_st, np.zeros(c_st.size), good_ids={0}
    )
    ref = _write_sort(
        tmp_path / "r", r_st, np.zeros(r_st.size), good_ids={0}
    )
    # disjoint trains -> no mutual best match; must not raise
    agree = symmetric_agreement(load_sort(cand), load_sort(ref))
    assert agree["matched_good_pairs"] == 0
    assert agree["gained_good"] == 1 and agree["lost_good"] == 1


def test_guardrails_flags_similar_pair_and_edge_spikes(tmp_path):
    st = np.tile(np.arange(0, 300_000, 400), 2)
    cl = np.repeat([0, 1], st.size // 2)
    sim = np.array([[1.0, 0.95], [0.95, 1.0]], dtype=np.float32)
    d = _write_sort(
        tmp_path / "s",
        st,
        cl,
        good_ids={0, 1},
        similar=sim,
        depths=[3, 3],  # same peak channel -> within depth window
        spike_depths=np.full(st.size, 5.0),  # near the shallow edge
    )
    g = guardrails(d, load_sort(d), FS)
    assert g["similar_good_good_pairs"] == 1
    assert g["similar_pairs_per_good_unit"] == 0.5
    assert g["edge_spike_fraction_40um"] == 1.0
    assert g["refractory_violation_median"] >= 0.0


def test_score_sort_end_to_end_contract(tmp_path):
    truth_st = np.arange(500, 400_000, 600)
    extra = np.sort(np.random.default_rng(0).integers(0, 400_000, 200))
    st = np.concatenate([truth_st, extra])
    cl = np.concatenate([np.zeros(truth_st.size), np.ones(extra.size)])
    d = _write_sort(tmp_path / "cand", st, cl, good_ids={0})
    ref = _write_sort(
        tmp_path / "ref", truth_st, np.zeros(truth_st.size), good_ids={0}
    )

    out = score_sort(
        d,
        truth={"u1": truth_st},
        reference=ref,
        runtime_s=120.0,
    )
    assert out["schema"] == SCORE_SCHEMA
    assert out["headline"] == out["primary"]["headline_units_recovered"] == 1
    assert out["secondary"]["matched_good_pairs"] == 1
    assert set(NOT_ENDPOINTS).issubset(out["context"]["not_endpoints"] + list(NOT_ENDPOINTS))
    assert "NOT promotion endpoints" in out["context"]["_warning"]
    assert out["runtime"]["runtime_s_per_recording_s"] is not None
    assert out["context"]["ks_good_count"] == 1


def test_score_sort_requires_a_sampling_rate(tmp_path):
    d = tmp_path / "no_params"
    _write_sort(d, np.arange(0, 1000, 10), np.zeros(100), good_ids={0})
    (d / "params.py").unlink()
    with pytest.raises(ValueError, match="sampling rate"):
        score_sort(d)
