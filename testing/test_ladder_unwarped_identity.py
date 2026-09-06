"""Known-answer fixtures for testing/ladder_unwarped_identity.py.

Each test below is aimed at one specific way the linker could be wrong, and is
built so the *right* answer is known by construction rather than by running the
code and writing down what it said. They are deliberately small: the point is
to catch the named failure, not to re-review the module.
"""

import numpy as np
import pytest

from testing.ladder_unwarped_identity import (
    CandidateLink,
    EpochObservation,
    IdentityRefusal,
    MotionDeclaration,
    ReplayInput,
    UnwarpedIdentityConfig,
    assign_rows_to_families,
    build_candidate_links,
    epochs_covering,
    extract_epoch_observations,
    refractory_violation_fraction,
    run_unwarped_identity_replay,
    solve_families,
    waveform_cosine,
)

FS = 30000.0
N_CHANNELS = 8
CHANNEL_POSITIONS = np.stack(
    [np.zeros(N_CHANNELS), np.arange(N_CHANNELS) * 20.0], axis=1
)


def _template_bank() -> np.ndarray:
    """Three templates on the same eight physical channels.

    0 and 1 are the same waveform (cosine 1.0); 2 is the inverted waveform
    (cosine -1.0) sitting on exactly the same channels, so nothing but the
    waveform itself distinguishes it.
    """
    n_samples = 10
    trough = -np.exp(-0.5 * ((np.arange(n_samples) - 5.0) / 1.5) ** 2)
    spatial = np.array([0.1, 0.5, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0])
    bank = np.zeros((3, n_samples, N_CHANNELS), dtype=np.float64)
    bank[0] = np.outer(trough, spatial)
    bank[1] = np.outer(trough, spatial)
    bank[2] = np.outer(-trough, spatial)
    return bank


def _inputs(spikes, *, fs=FS):
    """``spikes`` is a list of (time_s, cluster, depth_um, amplitude, template)."""
    times = np.array([s[0] for s in spikes], dtype=np.float64)
    return ReplayInput(
        row_id=np.arange(len(spikes), dtype=np.int64),
        sample=np.round(times * fs).astype(np.int64),
        cluster=np.array([s[1] for s in spikes], dtype=np.int64),
        depth_um=np.array([s[2] for s in spikes], dtype=np.float64),
        amplitude=np.array([s[3] for s in spikes], dtype=np.float64),
        template=np.array([s[4] for s in spikes], dtype=np.int64),
        template_bank=_template_bank(),
        channel_positions_um=CHANNEL_POSITIONS,
        fs_hz=fs,
    )


def _train(start_s, stop_s, n, cluster, depth, amplitude, template):
    return [
        (t, cluster, depth, amplitude, template)
        for t in np.linspace(start_s, stop_s, n, endpoint=False)
    ]


CONFIG = UnwarpedIdentityConfig(
    epoch_duration_s=120.0,
    epoch_overlap_s=30.0,
    min_spikes_per_epoch=5,
    max_spatial_distance_um=30.0,
    max_amplitude_ratio=2.0,
    min_waveform_cosine=0.9,
    ambiguity_threshold_ratio=0.85,
    max_refractory_violation_increase=0.01,
)

ABSENT = MotionDeclaration(mode="declared_absent", rationale="fixture")


def _link_between(links, cluster_a, cluster_b):
    (found,) = [l for l in links if l.cluster_a == cluster_a and l.cluster_b == cluster_b]
    return found


# --------------------------------------------------------------------------- #
# the epoch grid is the recording's, not the interval's
# --------------------------------------------------------------------------- #
def test_epoch_grid_is_anchored_to_the_recording_clock():
    """The same spike falls in the same epoch whichever bounded run reaches it."""
    wide = epochs_covering((0.0, 600.0), CONFIG)
    narrow = epochs_covering((180.0, 480.0), CONFIG)
    assert wide == [0, 1, 2, 3, 4, 5]
    # 180 = 2 * step, so the narrow run starts on grid epoch 2 rather than
    # renumbering its own first epoch as 0.
    assert narrow == [2, 3, 4]
    assert set(narrow) <= set(wide)


def test_only_whole_epochs_inside_the_declared_interval_are_processed():
    """A smoke run must not read outside its declared interval."""
    indices = epochs_covering((6350.0, 7050.0), CONFIG)
    for idx in indices:
        start = idx * CONFIG.epoch_step_s
        assert start >= 6350.0 and start + CONFIG.epoch_duration_s <= 7050.0
    assert epochs_covering((0.0, 119.0), CONFIG) == []


# --------------------------------------------------------------------------- #
# waveform compatibility on a common physical channel representation
# --------------------------------------------------------------------------- #
def test_waveform_gate_refuses_a_link_every_other_gate_would_accept():
    """Same depth, same amplitude, opposite waveform: only the waveform separates them."""
    spikes = _train(10.0, 100.0, 20, 0, 500.0, 12.0, 0) + _train(
        130.0, 200.0, 20, 1, 500.0, 12.0, 2
    )
    inputs = _inputs(spikes)
    observations = extract_epoch_observations(
        inputs, epochs_covering((0.0, 210.0), CONFIG), inputs.depth_um, CONFIG
    )
    link = _link_between(build_candidate_links(observations, inputs, CONFIG), 0, 1)

    assert link.spatial_distance_um == 0.0
    assert link.amplitude_ratio == pytest.approx(1.0)
    assert link.waveform_cosine == pytest.approx(-1.0)
    assert not link.accepted
    assert link.rejected_because == "waveform_cosine"

    # ... and it is the waveform that did it: the identical fixture with a
    # matching template instead of the inverted one links.
    matching = _inputs(
        _train(10.0, 100.0, 20, 0, 500.0, 12.0, 0)
        + _train(130.0, 200.0, 20, 1, 500.0, 12.0, 1)
    )
    matched_observations = extract_epoch_observations(
        matching, epochs_covering((0.0, 210.0), CONFIG), matching.depth_um, CONFIG
    )
    matched = _link_between(build_candidate_links(matched_observations, matching, CONFIG), 0, 1)
    assert matched.waveform_cosine == pytest.approx(1.0)
    assert matched.accepted


def test_waveform_comparison_uses_physical_channels_not_private_indices():
    """Two clusters peaking on different channels are compared on a shared set."""
    bank = _template_bank()
    near = EpochObservation(
        epoch_idx=0, cluster_id=0, start_s=0.0, stop_s=120.0, num_spikes=10,
        firing_rate_hz=1.0, mean_observed_depth_um=40.0, mean_tissue_depth_um=40.0,
        median_amplitude=1.0, refractory_violation_fraction=0.0, peak_channel=2,
        mean_waveform=bank[0],
    )
    same_shape_shifted = EpochObservation(
        epoch_idx=1, cluster_id=1, start_s=90.0, stop_s=210.0, num_spikes=10,
        firing_rate_hz=1.0, mean_observed_depth_um=40.0, mean_tissue_depth_um=40.0,
        median_amplitude=1.0, refractory_violation_fraction=0.0, peak_channel=3,
        mean_waveform=np.roll(bank[0], 1, axis=1),
    )
    # 60 um around channels 2 and 3 covers channels 0-6, so both waveforms are
    # expressed on the same physical channels before being compared.
    cosine = waveform_cosine(near, same_shape_shifted, CHANNEL_POSITIONS, 60.0)
    assert cosine is not None and 0.0 < cosine < 1.0

    # The shared set is defined in micrometres on the probe, so shrinking the
    # neighbourhood changes which channels are compared -- not whether the two
    # sparse channel lists happen to be indexed the same way.
    assert waveform_cosine(near, same_shape_shifted, CHANNEL_POSITIONS, 5.0) != cosine
    # A waveform with no energy on the shared channels yields no evidence, and
    # no evidence refuses the link rather than passing it.
    flat = EpochObservation(**{**vars(near), "mean_waveform": np.zeros_like(bank[0])})
    assert waveform_cosine(flat, same_shape_shifted, CHANNEL_POSITIONS, 60.0) is None


def test_a_link_with_no_waveform_evidence_is_refused_not_passed():
    obs_a = EpochObservation(
        epoch_idx=0, cluster_id=0, start_s=0.0, stop_s=120.0, num_spikes=10,
        firing_rate_hz=1.0, mean_observed_depth_um=40.0, mean_tissue_depth_um=40.0,
        median_amplitude=1.0, refractory_violation_fraction=0.0, peak_channel=2,
        row_ids=np.arange(10), mean_waveform=None,
    )
    obs_b = EpochObservation(**{**vars(obs_a), "epoch_idx": 1, "cluster_id": 1,
                                "row_ids": np.arange(10, 20)})
    inputs = _inputs(_train(0.0, 200.0, 20, 0, 40.0, 1.0, 0))
    links = build_candidate_links([obs_a, obs_b], inputs, CONFIG)
    assert [l.rejected_because for l in links] == ["waveform_unavailable"]
    assert not any(l.accepted for l in links)


# --------------------------------------------------------------------------- #
# exclusivity in both directions
# --------------------------------------------------------------------------- #
def test_one_source_may_claim_only_one_destination():
    """Destination-only exclusivity would let cluster 0 absorb both successors."""
    spikes = (
        _train(10.0, 100.0, 20, 0, 500.0, 12.0, 0)
        + _train(130.0, 200.0, 20, 1, 500.0, 12.0, 1)   # distance 0  -> score 1.00
        + _train(130.5, 200.5, 20, 2, 530.0, 12.0, 1)   # distance 30 -> score 0.25
    )
    inputs = _inputs(spikes)
    observations = extract_epoch_observations(
        inputs, epochs_covering((0.0, 210.0), CONFIG), inputs.depth_um, CONFIG
    )
    links = build_candidate_links(observations, inputs, CONFIG)

    accepted = [l for l in links if l.accepted]
    assert len(accepted) == 1 and accepted[0].cluster_b == 1
    assert _link_between(links, 0, 2).rejected_because == "source_already_claimed"


def test_one_destination_may_be_claimed_by_only_one_source():
    spikes = (
        _train(10.0, 100.0, 20, 0, 500.0, 12.0, 0)
        + _train(10.5, 100.5, 20, 2, 530.0, 12.0, 1)
        + _train(130.0, 200.0, 20, 1, 500.0, 12.0, 1)
    )
    inputs = _inputs(spikes)
    observations = extract_epoch_observations(
        inputs, epochs_covering((0.0, 210.0), CONFIG), inputs.depth_um, CONFIG
    )
    links = build_candidate_links(observations, inputs, CONFIG)

    accepted = [l for l in links if l.accepted]
    assert len(accepted) == 1 and accepted[0].cluster_a == 0
    assert _link_between(links, 2, 1).rejected_because == "destination_already_claimed"


def test_ambiguous_links_leave_both_sides_separate():
    """Two near-identical successors are not resolved by picking the winner."""
    spikes = (
        _train(10.0, 100.0, 20, 0, 500.0, 12.0, 0)
        + _train(130.0, 200.0, 20, 1, 500.0, 12.0, 1)
        + _train(130.5, 200.5, 20, 2, 500.5, 12.0, 1)
    )
    inputs = _inputs(spikes)
    observations = extract_epoch_observations(
        inputs, epochs_covering((0.0, 210.0), CONFIG), inputs.depth_um, CONFIG
    )
    links = build_candidate_links(observations, inputs, CONFIG)

    assert not any(l.accepted for l in links)
    assert {l.rejected_because for l in links if l.cluster_a == 0} == {"ambiguous_source"}

    _, family_of_cluster, _ = assign_rows_to_families(inputs, [])
    assert len(set(family_of_cluster.values())) == 3  # nothing merged


# --------------------------------------------------------------------------- #
# deduplication by original event identity, not by timestamp
# --------------------------------------------------------------------------- #
def test_two_distinct_spikes_at_one_sample_stay_visible_to_the_refractory_check():
    """Deduplicating on timestamps would delete one of them and pass the gate.

    Cluster 0 and cluster 1 fire on *exactly* the same samples throughout the
    epoch-0/epoch-1 overlap. As distinct rows their union has an interval of 0
    between every pair, so the refractory fraction is 1.0 and the link is
    refused. Collapsing the union with ``np.unique`` on times instead would
    leave a clean, evenly spaced train and accept the merge.
    """
    coincident = np.linspace(95.0, 115.0, 20, endpoint=False)
    spikes = (
        [(t, 0, 500.0, 12.0, 0) for t in coincident]
        + [(t, 1, 500.0, 12.0, 1) for t in coincident]
    )
    inputs = _inputs(spikes)
    observations = extract_epoch_observations(
        inputs, epochs_covering((0.0, 210.0), CONFIG), inputs.depth_um, CONFIG
    )
    link = _link_between(build_candidate_links(observations, inputs, CONFIG), 0, 1)

    # 40 distinct rows on 20 distinct samples: sorted, every second interval
    # is exactly 0, so 20 of the 39 intervals are violations.
    assert link.pair_refractory_fraction == pytest.approx(20.0 / 39.0)
    assert link.pair_refractory_increase == pytest.approx(20.0 / 39.0)  # both clean apart
    assert link.rejected_because == "epoch_pair_refractory_increase"

    # the timestamp-collapsing version of the same union looks spotless
    unique_times = np.unique(inputs.sample)
    assert refractory_violation_fraction(unique_times, FS, CONFIG.refractory_period_ms) == 0.0


def test_every_original_row_survives_assignment_exactly_once():
    coincident = np.linspace(95.0, 115.0, 20, endpoint=False)
    spikes = (
        [(t, 0, 500.0, 12.0, 0) for t in coincident]
        + [(t, 1, 520.0, 12.0, 1) for t in coincident]
    )
    inputs = _inputs(spikes)
    observations = extract_epoch_observations(
        inputs, epochs_covering((0.0, 210.0), CONFIG), inputs.depth_um, CONFIG
    )
    links = build_candidate_links(observations, inputs, CONFIG)
    row_family, _, meta = assign_rows_to_families(inputs, [l for l in links if l.accepted])

    # 40 distinct rows over 20 distinct samples: all 40 are assigned.
    assert len(row_family) == 40
    assert np.unique(inputs.sample).size == 20
    assert meta["dedup_key"] == "original_spike_row_id"
    assert set(row_family) == set(inputs.row_id.tolist())


def test_overlapping_epochs_do_not_split_a_cluster():
    """A row in two epochs has one family because its *cluster* has one family."""
    spikes = _train(10.0, 200.0, 60, 0, 500.0, 12.0, 0)
    inputs = _inputs(spikes)
    observations = extract_epoch_observations(
        inputs, epochs_covering((0.0, 210.0), CONFIG), inputs.depth_um, CONFIG
    )
    # the overlap [90, 120) puts some rows in both epoch 0 and epoch 1
    in_both = set(observations[0].row_ids.tolist()) & set(observations[1].row_ids.tolist())
    assert in_both

    row_family, family_of_cluster, meta = assign_rows_to_families(inputs, [])
    assert len(row_family) == 60
    assert meta["num_unassigned_rows"] == 0
    assert {row_family[r] for r in in_both} == {family_of_cluster[0]}
    assert len(set(row_family.values())) == 1


# --------------------------------------------------------------------------- #
# refractory cleanliness on the train that is actually exported
# --------------------------------------------------------------------------- #
def _observation(epoch_idx, cluster_id, rows, *, depth=500.0):
    lo = epoch_idx * CONFIG.epoch_step_s
    return EpochObservation(
        epoch_idx=epoch_idx, cluster_id=cluster_id, start_s=lo,
        stop_s=lo + CONFIG.epoch_duration_s, num_spikes=len(rows), firing_rate_hz=1.0,
        mean_observed_depth_um=depth, mean_tissue_depth_um=depth, median_amplitude=12.0,
        refractory_violation_fraction=0.0, peak_channel=2,
        row_ids=np.asarray(sorted(rows), dtype=np.int64), mean_waveform=_template_bank()[0],
    )


def _merge(cluster_a, cluster_b, *, increase=0.0, score=1.0):
    return CandidateLink(
        epoch_a=0, cluster_a=cluster_a, epoch_b=1, cluster_b=cluster_b,
        spatial_distance_um=0.0, amplitude_ratio=1.0, waveform_cosine=1.0,
        pair_refractory_fraction=increase, pair_refractory_increase=increase,
        link_score=score, is_merge=True, accepted=True,
    )


def test_a_merge_that_dirties_the_exported_train_is_pruned():
    """The gated quantity is the increase the merge causes, on the exported rows."""
    clean_a = [(t, 1, 500.0, 12.0, 0) for t in np.linspace(130.0, 190.0, 60)]
    # cluster 2 interleaves 0.5 ms after every one of cluster 1's spikes: apart
    # each is spotless, merged the train is nothing but violations.
    clean_b = [(t + 0.0005, 2, 500.0, 12.0, 1) for t in np.linspace(130.0, 190.0, 60)]
    inputs = _inputs(clean_a + clean_b)

    _, family_of_cluster, active, meta = solve_families([], [_merge(1, 2)], inputs, CONFIG)
    assert active == []
    assert meta["num_pruned_merges"] == 1
    assert family_of_cluster[1] != family_of_cluster[2]
    assert meta["input_partition_preserved"]


def test_pruning_continues_past_a_family_it_cannot_fix():
    """The reproducer: a dirty standalone cluster must not stop the loop.

    Cluster 9 is filthy on its own and no link caused it, so nothing can be
    pruned for it. Clusters 1 and 2 are a genuinely bad merge sitting in a
    different family. A loop that returns as soon as the *worst* breaching
    family has no removable link leaves that merge accepted.
    """
    dirty_standalone = []
    t = 300.0
    for _ in range(40):                      # 0.2 ms pairs: ~50% violations
        dirty_standalone += [(t, 9, 900.0, 12.0, 0), (t + 0.0002, 9, 900.0, 12.0, 0)]
        t += 1.0
    clean_a = [(x, 1, 500.0, 12.0, 0) for x in np.linspace(130.0, 190.0, 60)]
    clean_b = [(x + 0.0005, 2, 500.0, 12.0, 1) for x in np.linspace(130.0, 190.0, 60)]
    inputs = _inputs(clean_a + clean_b + dirty_standalone)

    _, family_of_cluster, active, meta = solve_families([], [_merge(1, 2)], inputs, CONFIG)

    # cluster 9 is left alone -- preserved, not certified, and reported
    standalone = family_of_cluster[9]
    assert meta["exported_train_refractory_fraction"][str(standalone)] > 0.4
    assert meta["exported_train_refractory_increase"][str(standalone)] == 0.0
    # ...and the bad merge in the other family is still pruned
    assert meta["num_pruned_merges"] == 1
    assert active == []
    assert family_of_cluster[1] != family_of_cluster[2]


def test_a_high_baseline_cluster_is_preserved_without_being_certified():
    """Retaining an imperfect cluster is not a claim that it is a clean neuron."""
    dirty = []
    t = 130.0
    for _ in range(40):
        dirty += [(t, 5, 500.0, 12.0, 0), (t + 0.0002, 5, 500.0, 12.0, 0)]
        t += 1.0
    inputs = _inputs(dirty)
    row_family, family_of_cluster, active, meta = solve_families([], [], inputs, CONFIG)

    assert len(row_family) == len(dirty)          # every event survives
    assert set(family_of_cluster) == {5}          # one cluster, one family
    assert meta["input_partition_preserved"]
    assert meta["families_breaching_without_a_prunable_link"] == []
    family = family_of_cluster[5]
    # high absolute violation fraction, reported; zero increase, so not pruned
    assert meta["exported_train_refractory_fraction"][str(family)] > 0.4
    assert meta["exported_train_refractory_increase"][str(family)] == 0.0
    assert "not a claim" in meta["single_cluster_families_are_preserved_not_certified"]


# --------------------------------------------------------------------------- #
# the preservation invariant, under everything that broke v1 at once
# --------------------------------------------------------------------------- #
def test_the_input_partition_survives_every_condition_that_broke_v1(tmp_path):
    """One fixture carrying all four v1 failure conditions simultaneously.

    * a high-violation input cluster (37', ~50% of its ISIs under 1.5 ms), whose
      self-links v1 refused and whose refusal shredded it;
    * therefore refused self-links across all epoch boundaries;
    * overlapping epochs, so rows are seen twice;
    * ambiguous neighbouring clusters at matching depth and amplitude, so the
      merge candidates exist and are contested;
    * and boundary/low-count rows that sit in no eligible epoch at all.

    Every original event must survive, and no original cluster may fragment.
    """
    rng = np.random.default_rng(11)
    spikes = []
    # a dirty, long-lived cluster spanning every epoch
    t = 5.0
    while t < 200.0:
        spikes += [(t, 37, 500.0, 12.0, 0), (t + 0.0002, 37, 500.0, 12.0, 0)]
        t += 0.5
    # two neighbours that are near-identical to each other -> ambiguous merges
    for cid, depth in ((41, 515.0), (42, 515.4)):
        spikes += [
            (x + rng.uniform(0, 0.001), cid, depth, 12.0, 1)
            for x in np.linspace(5.0, 200.0, 300)
        ]
    # a cluster with too few spikes per epoch to produce any observation, and
    # rows outside every whole epoch in the processing interval
    spikes += [(x, 77, 505.0, 12.0, 0) for x in (1.0, 2.0, 3.0, 202.0, 205.0, 208.0)]
    inputs = _inputs(sorted(spikes, key=lambda s: s[0]))

    result = run_unwarped_identity_replay(
        inputs, motion=ABSENT, config=CONFIG,
        processing_interval_s=(0.0, 210.0), output_dir=tmp_path,
    )
    manifest = result["manifest"]
    row_family = result["row_family"]

    # the dirty cluster's self-links were all refused, as in v1
    self_links = [l for l in result["links"] if l.cluster_a == 37 and l.cluster_b == 37]
    assert self_links
    assert all(l.rejected_because == "self_link_no_output_effect" for l in self_links)
    assert not any(l.accepted for l in self_links)

    # every original event survives, exactly once
    assert len(row_family) == inputs.row_id.size
    assert set(row_family) == set(inputs.row_id.tolist())
    assert manifest["assignment"]["num_unassigned_rows"] == 0

    # no original cluster fragmented: one family per cluster, and the map from
    # cluster to family is the identity up to renumbering
    by_cluster = {}
    for row, family in row_family.items():
        cluster = int(inputs.cluster[int(np.flatnonzero(inputs.row_id == row)[0])])
        by_cluster.setdefault(cluster, set()).add(family)
    assert all(len(f) == 1 for f in by_cluster.values()), by_cluster
    assert manifest["num_original_clusters"] == 4
    assert manifest["input_partition_preserved"] is True
    assert manifest["num_families"] == 4

    # the low-count cluster never produced an observation and still came through
    assert 77 in by_cluster
    assert not any(o.cluster_id == 77 for o in result["observations"])


def test_with_no_accepted_merge_the_export_is_the_input_renumbered(tmp_path):
    spikes = (
        _train(10.0, 200.0, 60, 3, 100.0, 12.0, 0)
        + _train(10.0, 200.0, 60, 8, 900.0, 40.0, 1)      # far away, no merge
    )
    inputs = _inputs(spikes)
    result = run_unwarped_identity_replay(
        inputs, motion=ABSENT, config=CONFIG,
        processing_interval_s=(0.0, 210.0), output_dir=tmp_path,
    )
    assert result["manifest"]["num_families_built_from_a_merge"] == 0
    mapping = result["family_of_cluster"]
    assert len(set(mapping.values())) == len(mapping) == 2
    families = np.array([mapping[int(c)] for c in inputs.cluster])
    # identical partition: two rows share a family iff they shared a cluster
    same_family = families[:, None] == families[None, :]
    same_cluster = inputs.cluster[:, None] == inputs.cluster[None, :]
    assert np.array_equal(same_family, same_cluster)


# --------------------------------------------------------------------------- #
# refusals: fabricated depths and implicit motion
# --------------------------------------------------------------------------- #
def test_missing_depths_are_a_refusal_not_a_depth_of_zero():
    spikes = _train(10.0, 100.0, 20, 0, 500.0, 12.0, 0)
    inputs = _inputs(spikes)
    with pytest.raises(IdentityRefusal, match="finite real depths"):
        ReplayInput(**{**vars(inputs), "depth_um": np.full(20, np.nan)})


def test_a_motion_aware_arm_must_name_its_qualified_field():
    with pytest.raises(IdentityRefusal, match="never silently substituted"):
        MotionDeclaration(mode="qualified_field")
    with pytest.raises(IdentityRefusal, match="requires the field's sha256"):
        MotionDeclaration(mode="qualified_field", displacement_um=np.zeros(3))
    with pytest.raises(IdentityRefusal, match="must not carry a displacement"):
        MotionDeclaration(mode="declared_absent", displacement_um=np.zeros(3))
    with pytest.raises(IdentityRefusal, match="motion mode must be one of"):
        MotionDeclaration(mode="whatever_is_lying_around")


def test_an_unsupported_spike_under_a_qualified_field_is_a_refusal(tmp_path):
    spikes = _train(10.0, 200.0, 40, 0, 500.0, 12.0, 0)
    inputs = _inputs(spikes)
    displacement = np.zeros(40)
    displacement[7] = np.nan  # the field does not support this spike
    motion = MotionDeclaration(
        mode="qualified_field",
        displacement_um=displacement,
        field_identity={"sha256": "0" * 64, "path": "fixture.npz"},
    )
    with pytest.raises(IdentityRefusal, match="not a spike with zero displacement"):
        run_unwarped_identity_replay(
            inputs, motion=motion, config=CONFIG,
            processing_interval_s=(0.0, 210.0), output_dir=tmp_path,
        )


# --------------------------------------------------------------------------- #
# end to end on the small fixture
# --------------------------------------------------------------------------- #
def test_replay_writes_its_artifacts_and_records_why_it_linked_what_it_did(tmp_path):
    spikes = _train(10.0, 200.0, 60, 0, 500.0, 12.0, 0)
    inputs = _inputs(spikes)
    result = run_unwarped_identity_replay(
        inputs, motion=ABSENT, config=CONFIG,
        processing_interval_s=(0.0, 210.0), output_dir=tmp_path,
    )
    manifest = result["manifest"]

    assert manifest["execution_mode"] == "retained_sort_replay"
    assert manifest["motion"]["motion_aware"] is False
    assert manifest["epoch_indices"] == [0, 1]
    assert manifest["num_input_rows"] == 60
    assert manifest["assignment"]["num_assigned_rows"] == 60
    assert manifest["assignment"]["num_unassigned_rows"] == 0
    # one input cluster in, one family out, unchanged
    assert manifest["num_original_clusters"] == 1
    assert manifest["num_families"] == 1
    assert manifest["families_built_from_a_merge"] == []
    assert manifest["input_partition_preserved"] is True
    assert set(manifest["link_rejections"]) >= {"waveform_cosine", "self_link_no_output_effect"}
    for name in ("epoch_observations.csv", "candidate_links.csv", "family_membership.csv"):
        assert (tmp_path / name).exists()


def test_a_processing_interval_too_short_for_one_epoch_is_a_refusal(tmp_path):
    inputs = _inputs(_train(10.0, 100.0, 20, 0, 500.0, 12.0, 0))
    with pytest.raises(IdentityRefusal, match="no whole"):
        run_unwarped_identity_replay(
            inputs, motion=ABSENT, config=CONFIG,
            processing_interval_s=(0.0, 100.0), output_dir=tmp_path,
        )
