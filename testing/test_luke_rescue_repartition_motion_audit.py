import numpy as np

from testing.luke_rescue_repartition_motion_audit import PRESPEC, _score_family

FS = 30_000.0


def _other_from_assignment(anchor_st, frag_of_spike):
    """A minimal 'other' sort where anchor spike i lands in cluster frag_of_spike[i]."""
    st = np.asarray(anchor_st, dtype=np.int64)
    cl = np.asarray(frag_of_spike, dtype=np.int64)
    order = np.argsort(st)
    ids = set(int(c) for c in np.unique(cl))
    return {
        "st": st[order],
        "cl": cl[order],
        "label": {c: "good" for c in ids},
        "good": ids,
    }


def test_prespec_is_frozen_shape():
    assert PRESPEC["schema"] == "luke-rescue-repartition-motion-v2"
    assert set(PRESPEC["family_sampling"]["sides"]) == {
        "legacy_lost_dispersed",
        "rescue_gained_dispersed",
    }
    assert PRESPEC["discriminator"]["temporal_overlap_successive_max"] == 0.20


def test_successive_motion_tracking_clean_is_motion_fragmentation():
    n = 6000
    anchor_st = np.arange(0, n * 3000, 3000, dtype=np.int64)  # 10 Hz, 600 s
    frag = np.where(np.arange(n) < n // 2, 0, 1)  # first half unit 0, second unit 1
    other = _other_from_assignment(anchor_st, frag)
    depth = np.linspace(1200.0, 1230.0, n)  # 30 µm downward drift
    t = anchor_st / FS
    motion = (np.linspace(0, 620, 64), np.linspace(0.0, 31.0, 64))  # tracks depth

    out = _score_family(anchor_st, depth, other, FS, motion)
    assert out["successive"] and out["clean_merge"]
    assert out["motion_tracking"]
    assert out["classification"] == "motion_fragmentation"


def test_coexisting_fragments_are_over_splitting():
    n = 6000
    anchor_st = np.arange(0, n * 3000, 3000, dtype=np.int64)
    frag = np.tile([0, 1], n // 2)  # both units present in every bin
    other = _other_from_assignment(anchor_st, frag)
    depth = np.full(n, 1500.0)
    out = _score_family(anchor_st, depth, other, FS, None)
    assert out["coexist"]
    assert out["classification"] == "over_splitting"


def test_successive_clean_but_flat_depth_is_its_own_bucket():
    n = 6000
    anchor_st = np.arange(0, n * 3000, 3000, dtype=np.int64)
    frag = np.where(np.arange(n) < n // 2, 0, 1)
    other = _other_from_assignment(anchor_st, frag)
    depth = np.full(n, 1500.0)  # no trajectory
    out = _score_family(anchor_st, depth, other, FS, None)
    assert out["successive"] and out["clean_merge"]
    assert not out["motion_tracking"] and not out["monotonic_depth"]
    assert out["classification"] == "successive_clean_no_motion_signal"


def test_single_fragment_family_is_skipped():
    n = 2000
    anchor_st = np.arange(0, n * 3000, 3000, dtype=np.int64)
    other = _other_from_assignment(anchor_st, np.zeros(n))
    assert _score_family(anchor_st, np.full(n, 1000.0), other, FS, None) is None


def test_contaminated_merge_is_not_motion_fragmentation():
    n = 6000
    # add refractory violations: pairs 0.4 ms apart
    base = np.arange(0, n * 3000, 3000, dtype=np.int64)
    anchor_st = np.sort(np.concatenate([base, base + 12]))  # 12 samples = 0.4 ms
    frag = np.where(np.arange(anchor_st.size) < anchor_st.size // 2, 0, 1)
    other = _other_from_assignment(anchor_st, frag)
    depth = np.linspace(1200.0, 1230.0, anchor_st.size)
    motion = (np.linspace(0, 620, 64), np.linspace(0.0, 31.0, 64))
    out = _score_family(anchor_st, depth, other, FS, motion)
    assert not out["clean_merge"]
    assert out["classification"] != "motion_fragmentation"


def test_merge_cleanliness_includes_unmatched_fragment_spikes():
    n = 6000
    anchor_st = np.arange(0, n * 3000, 3000, dtype=np.int64)
    frag = np.where(np.arange(n) < n // 2, 0, 1)
    other = _other_from_assignment(anchor_st, frag)
    # Extra events do not match the anchor but would be present after merging.
    extra = anchor_st[: n // 2] + 12
    other["st"] = np.concatenate([other["st"], extra])
    other["cl"] = np.concatenate(
        [other["cl"], np.zeros(extra.size, dtype=np.int64)]
    )
    order = np.argsort(other["st"], kind="stable")
    other["st"], other["cl"] = other["st"][order], other["cl"][order]

    out = _score_family(
        anchor_st, np.linspace(1200.0, 1230.0, n), other, FS, None
    )
    assert not out["clean_merge"]
    assert out["merged_fragment_spikes"] > out["n_spikes"]
