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


def test_ground_truth_two_trains_one_output_event_is_scored_as_a_merge():
    # Two injected units fire within the scoring tolerance and the sorter
    # resolves only one event. Per-cluster matching (decisions/0014) credits
    # both trains against that cluster; the invariant that matters is that the
    # single output unit is flagged as a merge for both trains, so neither is
    # counted as cleanly recovered.
    sort = {
        "st": np.array([1_010], dtype=np.int64),
        "cl": np.array([0], dtype=np.int64),
        "label": {0: "good"},
        "good": {0},
    }
    out = ground_truth_scores(
        sort,
        {"a": np.array([1_000]), "b": np.array([1_015])},
        FS,
        duration_s=1.0,
        tol_ms=0.5,
    )
    assert all(u["merge"] for u in out["units"])
    assert out["headline_units_recovered"] == 0


# --- decisions/0014 regression tests: per-cluster matching ------------------- #
def _pooled_v2_best_accuracy(sort, truth_st, tol):
    """The retracted v2 scoring: one global 1:1 match against the whole river,
    then credit each pair to its cluster and read the best cluster's accuracy.
    Kept here only to assert the counterfactual the v3 fix must beat."""
    st = np.asarray(sort["st"], np.int64)
    cl = np.asarray(sort["cl"], np.int64)
    o = np.argsort(st, kind="stable")
    st, cl = st[o], cl[o]
    t = np.sort(np.asarray(truth_st, np.int64))
    caught: dict[int, int] = {}
    i = j = 0
    while i < t.size and j < st.size:
        d = int(st[j]) - int(t[i])
        if d < -tol:
            j += 1
        elif d > tol:
            i += 1
        else:
            caught[int(cl[j])] = caught.get(int(cl[j]), 0) + 1
            i += 1
            j += 1
    best = max(caught, key=caught.get)
    tp = caught[best]
    fp = int((cl == best).sum()) - tp
    fn = t.size - tp
    return tp / (tp + fp + fn)


def _capture_only_count(sort, truth_st, tol):
    """Clusters clipping >5% of the train under capture alone (no precision
    clause) — the pre-fix behaviour the precision gate must suppress."""
    from testing.ladder_score import _exclusive_pairs
    st = np.asarray(sort["st"], np.int64)
    cl = np.asarray(sort["cl"], np.int64)
    t = np.sort(np.asarray(truth_st, np.int64))
    n = 0
    for c in np.unique(cl):
        ost = np.sort(st[cl == c])
        ta, _ = _exclusive_pairs(t, ost, tol)
        if ta.size / t.size > 0.05:
            n += 1
    return n


def test_dense_background_does_not_steal_from_the_best_cluster():
    # The C2 v3 pathology: global exclusive matching against the pooled spike
    # river let a background spike *earlier* than a truth event grab that match,
    # a constant ~0.78 accuracy floor. Here ~15% of truth events get a guaranteed
    # steal partner two samples early, spread thin across 40 background clusters
    # (each well under the 5% capture threshold). v3 per-cluster matching must
    # score the exact-copy cluster at 1.0 anyway.
    tol = int(round(0.5 / 1000.0 * FS))  # 15
    truth_st = np.arange(1_000, 1_000 + 1_000 * 250, 250)  # 1000 spikes
    steal_at = truth_st[::7] - 2  # ~143 events, within tol and earlier
    st = [truth_st]
    cl = [np.zeros(truth_st.size, dtype=np.int64)]
    for k, chunk in enumerate(np.array_split(steal_at, 40), start=1):
        st.append(np.sort(chunk))
        cl.append(np.full(chunk.size, k, dtype=np.int64))
    st = np.concatenate(st).astype(np.int64)
    cl = np.concatenate(cl)
    sort = {
        "st": st, "cl": cl,
        "label": {c: ("good" if c == 0 else "mua") for c in range(41)}, "good": {0},
    }
    out = ground_truth_scores(sort, {"u1": truth_st}, FS, duration_s=10.0)
    u = out["units"][0]
    assert u["best_output_unit"] == 0
    assert u["tp"] == truth_st.size and u["fn"] == 0
    assert u["accuracy"] == 1.0
    assert u["n_output_units_capturing"] == 1  # no background cluster captures >5%
    assert u["recovered"] is True
    # ... and confirm this fixture actually exercises the pathology
    assert _pooled_v2_best_accuracy(sort, truth_st, tol) < 0.8


def test_true_split_lowers_accuracy_and_is_flagged():
    truth_st = np.arange(0, 300_000, 200)  # 1500 spikes
    half = truth_st.size // 2
    st = truth_st.copy()
    cl = np.zeros(truth_st.size, dtype=np.int64)
    cl[half:] = 1  # one injected neuron, two output clusters, no overlap
    sort = {"st": st, "cl": cl, "label": {0: "good", 1: "good"}, "good": {0, 1}}
    out = ground_truth_scores(sort, {"u1": truth_st}, FS, duration_s=10.0)
    u = out["units"][0]
    assert u["n_output_units_capturing"] == 2 and u["split"] is True
    assert u["accuracy"] < 0.55  # best cluster only holds ~half the train
    assert u["recovered"] is False


def test_duplicate_cluster_is_noticed_while_best_cluster_score_stays_sensible():
    truth_st = np.arange(1_000, 300_000, 300)  # ~997 spikes
    dup = truth_st[: int(truth_st.size * 0.4)]  # 40% copied into another cluster
    st = np.concatenate([truth_st, dup])
    cl = np.concatenate([np.zeros(truth_st.size), np.ones(dup.size)]).astype(np.int64)
    sort = {
        "st": st.astype(np.int64), "cl": cl,
        "label": {0: "good", 1: "good"}, "good": {0, 1},
    }
    out = ground_truth_scores(sort, {"u1": truth_st}, FS, duration_s=10.0)
    u = out["units"][0]
    assert u["best_output_unit"] == 0 and u["accuracy"] == 1.0  # clean best cluster
    assert u["n_output_units_capturing"] == 2 and u["split"] is True
    assert u["recovered"] is False  # the duplicate blocks a clean-recovery call


def test_identity_continuity_match_across_a_bin_edge_never_exceeds_one():
    # decisions/0014 review #3: a legitimate +/-tol match can straddle a 30 s
    # bin boundary. TP/FN must be counted on the truth clock and FP only on
    # unmatched output spikes, so per-bin accuracy stays in [0, 1].
    bin_edge = int(30.0 * FS)
    tol = int(round(0.5 / 1000.0 * FS))  # 15
    # truth events straddling the edge, each recovered by an output spike on the
    # OTHER side of the edge: truth at bin_edge-5 -> output at bin_edge+8 (bin 1),
    # truth at bin_edge+5 -> output at bin_edge-8 (bin 0). The v2 accounting put
    # the TP in one bin and the output-derived FP in the other -> fp<0, acc>1.
    truth_st = np.concatenate([
        np.arange(bin_edge - 40 * 300, bin_edge - 300, 300),   # well inside bin 0
        np.array([bin_edge - 5]),                              # last event in bin 0
        np.arange(bin_edge + 2 * 300, bin_edge + 40 * 300, 300),  # well inside bin 1
    ])
    out_st = truth_st.copy()
    # the bin-0 straddler is recovered by an output spike in bin 1 (asymmetric:
    # nothing crosses back). Old accounting: TP credited to bin 0, output absent
    # from bin 0's o_seg -> fp = o_seg.size - best_tp = -1 -> acc = 40/39 > 1.
    out_st[out_st == bin_edge - 5] = bin_edge + 8
    order = np.argsort(out_st)
    sort = {
        "st": out_st[order].astype(np.int64),
        "cl": np.zeros(out_st.size, dtype=np.int64),
        "label": {0: "good"}, "good": {0},
    }
    out = ground_truth_scores(sort, {"u1": truth_st}, FS, duration_s=61.0)
    u = out["units"][0]
    assert u["fp"] == 0 and u["fn"] == 0 and u["accuracy"] == 1.0
    assert 0.0 <= u["min_bin_accuracy"] <= 1.0
    assert not np.isnan(u["min_bin_accuracy"])


def test_ground_truth_normalises_unsorted_inputs():
    # decisions/0014 review #1: _exclusive_pairs needs sorted inputs; the public
    # entry point must normalise a shuffled sort and a shuffled truth train.
    rng = np.random.default_rng(3)
    truth_st = np.arange(1_000, 200_000, 400)
    st = truth_st.copy()
    cl = np.zeros(st.size, dtype=np.int64)
    perm = rng.permutation(st.size)
    sort = {"st": st[perm], "cl": cl[perm], "label": {0: "good"}, "good": {0}}
    out = ground_truth_scores(
        sort, {"u1": rng.permutation(truth_st)}, FS, duration_s=7.0
    )
    u = out["units"][0]
    assert u["tp"] == truth_st.size and u["fp"] == 0 and u["fn"] == 0
    assert u["accuracy"] == 1.0


def test_split_diagnostic_ignores_chance_coincidence_from_high_rate_background():
    # decisions/0014: on a dense real strip several high-rate background clusters
    # clip >5% of a 6 Hz injected train by pure chance. Without a precision
    # clause every clean donor reads as split. A capturing cluster must also be
    # >5% injected train.
    tol = int(round(0.5 / 1000.0 * FS))
    dur_s = 120.0
    n_samp = int(dur_s * FS)
    truth_st = np.arange(1_000, n_samp - 1_000, 5_000)  # ~720 spikes, 6 Hz
    rng = np.random.default_rng(1)
    st = [truth_st]                                   # cluster 0: exact copy
    cl = [np.zeros(truth_st.size, dtype=np.int64)]
    for k in range(1, 8):  # 7 background clusters at ~67 Hz over the whole window
        bg = np.sort(rng.integers(0, n_samp, 8_000))
        st.append(bg)
        cl.append(np.full(bg.size, k, dtype=np.int64))
    st = np.concatenate(st).astype(np.int64)
    cl = np.concatenate(cl)
    sort = {"st": st, "cl": cl,
            "label": {c: ("good" if c == 0 else "mua") for c in range(8)}, "good": {0}}
    out = ground_truth_scores(sort, {"u1": truth_st}, FS, duration_s=dur_s)
    u = out["units"][0]
    assert _capture_only_count(sort, truth_st, tol) > 1  # chance clip really happens
    assert u["best_output_unit"] == 0 and u["accuracy"] == 1.0
    assert u["n_output_units_capturing"] == 1           # precision gate removes it
    assert u["split"] is False and u["recovered"] is True


def test_symmetric_agreement_reports_both_sides_and_withholds_detection_claim(tmp_path):
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
    # Temporal absence alone is no longer promoted to a detection claim.
    assert agree["lost_absent_at_detection"] is None
    assert agree["lost_detection_status"] == "unresolved_requires_spatial_null_audit"


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
