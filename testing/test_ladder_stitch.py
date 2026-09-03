import json

import numpy as np
import pandas as pd
import pytest

from testing.ladder_stitch import (
    STITCH_SCHEMA,
    StitchConfig,
    _connected_components,
    apply_stitch,
    merged_rv_fraction,
    shifted_cosine,
    stitch_families,
    temporal_overlap,
)

FS = 30_000.0


def _template(peak_chan, n_chan=24, n_samp=61, amp=-100.0):
    t = np.zeros((n_samp, n_chan), dtype=np.float32)
    prof = np.array([-0.1, -0.4, -1.0, -0.5, -0.15], dtype=np.float32) * -amp
    for dc, w in ((-1, 0.4), (0, 1.0), (1, 0.4)):
        c = peak_chan + dc
        if 0 <= c < n_chan:
            t[28:33, c] = prof * w
    return t


def _write_sort(d, units, n_chan=24):
    """units: {id: (spike_samples, template)}."""
    d.mkdir(parents=True, exist_ok=True)
    st = np.concatenate([np.sort(s) for s, _ in units.values()])
    cl = np.concatenate([np.full(len(s), i) for i, (s, _) in zip(units, units.values())])
    order = np.argsort(st)
    np.save(d / "spike_times.npy", st[order].astype(np.int64))
    np.save(d / "spike_clusters.npy", cl[order].astype(np.int64))
    n = max(units) + 1
    tmpl = np.zeros((n, 61, n_chan), dtype=np.float32)
    for i, (_, t) in units.items():
        tmpl[i] = t
    np.save(d / "templates.npy", tmpl)
    np.save(
        d / "channel_positions.npy",
        np.column_stack([np.zeros(n_chan), np.arange(n_chan) * 20.0]).astype(np.float64),
    )
    pd.DataFrame({"cluster_id": range(n), "KSLabel": ["good"] * n}).to_csv(
        d / "cluster_KSLabel.tsv", sep="\t", index=False
    )
    (d / "cluster_group.tsv").write_text(
        (d / "cluster_KSLabel.tsv").read_text()
    )
    (d / "params.py").write_text(f"sample_rate = {FS}\n")
    return d


def test_shifted_cosine_matches_a_depth_shifted_copy():
    a = _template(10)
    b = _template(13)  # same shape, 3 channels deeper
    assert shifted_cosine(a, b) > 0.99
    noise = np.random.default_rng(0).normal(0, 30, a.shape).astype("float32")
    assert shifted_cosine(a, noise) < 0.5


def test_temporal_overlap_is_zero_for_successive_and_one_for_identical():
    first = np.arange(0, 60 * FS, 3000, dtype=np.int64)
    second = np.arange(60 * FS, 120 * FS, 3000, dtype=np.int64)
    assert temporal_overlap(first, second, FS, 30.0) < 0.05
    assert temporal_overlap(first, first, FS, 30.0) == pytest.approx(1.0)


def test_merged_rv_fraction_flags_colliding_trains():
    a = np.arange(0, 100 * 3000, 3000, dtype=np.int64)
    b = a + 10  # 0.33 ms apart -> refractory violations on merge
    assert merged_rv_fraction(a, b, FS, 1.5) > 0.4
    assert merged_rv_fraction(a, a + 1500, FS, 1.5) == 0.0  # 50 ms apart, no violations


def test_connected_components_groups_transitively():
    groups = _connected_components([(1, 2), (2, 3), (5, 6)], {1, 2, 3, 4, 5, 6})
    groups = sorted(sorted(g) for g in groups)
    assert groups == [[1, 2, 3], [5, 6]]


@pytest.fixture
def fragmented_sort(tmp_path):
    rng = np.random.default_rng(0)
    # unit 0 + unit 1: one drifting neuron — same shape, 3 ch apart, successive epochs
    u0 = np.arange(1000, 60 * int(FS), 2500, dtype=np.int64)
    u1 = np.arange(60 * int(FS), 118 * int(FS), 2500, dtype=np.int64)
    # unit 2: a different, unrelated neuron that fires throughout, far away
    u2 = np.sort(rng.integers(0, 118 * int(FS), 3000)).astype(np.int64)
    return _write_sort(
        tmp_path / "sort",
        {
            0: (u0, _template(10)),
            1: (u1, _template(13)),
            2: (u2, _template(20)),
        },
    )


def test_stitch_families_merges_the_drifting_pair_only(fragmented_sort):
    out = stitch_families(fragmented_sort, StitchConfig(min_spikes=50))
    assert out["schema"] == STITCH_SCHEMA
    assert out["families"] == [[0, 1]]
    assert out["n_good_after"] == 2  # 3 -> 2
    assert out["n_units_absorbed"] == 1


def test_stitch_leaves_a_clean_sort_alone(tmp_path):
    # two well-separated, co-firing, dissimilar units — nothing to stitch
    rng = np.random.default_rng(1)
    a = np.sort(rng.integers(0, 100 * int(FS), 2000)).astype(np.int64)
    b = np.sort(rng.integers(0, 100 * int(FS), 2000)).astype(np.int64)
    d = _write_sort(tmp_path / "s", {0: (a, _template(4)), 1: (b, _template(19))})
    out = stitch_families(d, StitchConfig(min_spikes=50))
    assert out["families"] == []
    assert out["n_good_after"] == out["n_good_before"] == 2


def test_apply_stitch_relabels_and_drops_absorbed_units(fragmented_sort, tmp_path):
    out_dir = tmp_path / "stitched"
    result = apply_stitch(fragmented_sort, out_dir, StitchConfig(min_spikes=50))
    assert result["families"] == [[0, 1]]

    new_cl = np.load(out_dir / "spike_clusters.npy")
    assert set(np.unique(new_cl)) == {0, 2}  # unit 1 folded into 0 (0 has more spikes)
    labels = pd.read_csv(out_dir / "cluster_KSLabel.tsv", sep="\t")
    assert 1 not in set(labels["cluster_id"])
    assert json.loads((out_dir / "stitch_result.json").read_text())["n_families"] == 1


def test_apply_stitch_refuses_mnt(fragmented_sort):
    with pytest.raises(ValueError, match="/mnt"):
        apply_stitch(fragmented_sort, "/mnt/x")


def test_mutual_best_partner_edges_do_not_collapse_unrelated_successive_units(tmp_path):
    # Four low-rate units that each fire in their own epoch (pairwise
    # "successive" and refractory-clean) but have *different* template shapes.
    # A transitive closure over the pairwise relation would merge all four;
    # the mutual-best-partner rule must not, because none is another's best
    # cosine match.
    fs = int(FS)
    units = {}
    for i, chan in enumerate((3, 8, 14, 20)):
        s = np.arange(i * 25 * fs + 1000, (i + 1) * 25 * fs, 2500, dtype=np.int64)
        t = _template(chan)
        # give each a distinct waveform so cross-cosines stay low
        t[28:33, chan] *= 1.0 + 0.5 * i
        units[i] = (s, t)
    d = _write_sort(tmp_path / "s", units)
    out = stitch_families(d, StitchConfig(min_spikes=20, depth_window_um=400.0))
    assert all(len(f) <= 2 for f in out["families"])
    assert out["n_units_absorbed"] <= 1
