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
    max_refractory_violation_fraction=0.01,
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

    _, node_family, _ = assign_rows_to_families(observations, [])
    assert len({node_family[o.key] for o in observations}) == len(observations)


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
    assert link.rejected_because == "epoch_pair_refractory"

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
    row_family, _, meta = assign_rows_to_families(observations, [l for l in links if l.accepted])

    # 40 distinct rows over 20 distinct samples: all 40 are assigned.
    assert len(row_family) == 40
    assert np.unique(inputs.sample).size == 20
    assert meta["dedup_key"] == "original_spike_row_id"
    assert set(row_family) == set(inputs.row_id.tolist())


def test_overlap_rows_are_counted_once_and_go_to_the_earliest_epoch():
    spikes = _train(10.0, 200.0, 60, 0, 500.0, 12.0, 0)
    inputs = _inputs(spikes)
    observations = extract_epoch_observations(
        inputs, epochs_covering((0.0, 210.0), CONFIG), inputs.depth_um, CONFIG
    )
    # the overlap [90, 120) puts some rows in both epoch 0 and epoch 1
    in_both = set(observations[0].row_ids.tolist()) & set(observations[1].row_ids.tolist())
    assert in_both

    row_family, _, meta = assign_rows_to_families(observations, [])
    assert len(row_family) == 60
    earliest_family = min(
        f for k, f in assign_rows_to_families(observations, [])[1].items() if k[0] == 0
    )
    assert {row_family[r] for r in in_both} == {earliest_family}


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


def test_the_refractory_gate_is_rechecked_on_the_train_the_export_writes():
    """A pairwise-clean link whose *exported* family train breaches is pruned.

    The pair gate scores the two observations' whole row sets. The export does
    not get those rows: the overlap-assignment rule hands cluster 1's shared
    rows to its own earlier-epoch observation, and the family that is left is
    small enough that the same single violation breaches the 1% gate. Checking
    only the pair would have shipped it.
    """
    # cluster 0: 60 clean rows inside epoch 0 only
    c0 = [(t, 0, 500.0, 12.0, 0) for t in np.linspace(1.0, 89.0, 60)]
    # cluster 1: 60 rows in the epoch 0/1 overlap, plus 5 in epoch 1 alone,
    # two of which are 0.5 ms apart -- one refractory violation.
    shared = [(t, 1, 500.0, 12.0, 1) for t in np.linspace(90.0, 119.0, 60)]
    tail_times = [130.0, 150.0, 150.0005, 170.0, 190.0]
    tail = [(t, 1, 500.0, 12.0, 1) for t in tail_times]
    inputs = _inputs(c0 + shared + tail)

    obs_c0_ep0 = _observation(0, 0, range(0, 60))
    obs_c1_ep0 = _observation(0, 1, range(60, 120))
    obs_c1_ep1 = _observation(1, 1, range(60, 125))
    observations = [obs_c0_ep0, obs_c1_ep0, obs_c1_ep1]

    link = CandidateLink(
        epoch_a=0, cluster_a=0, epoch_b=1, cluster_b=1,
        spatial_distance_um=0.0, amplitude_ratio=1.0, waveform_cosine=1.0,
        pair_refractory_fraction=1.0 / 124.0, link_score=1.0, accepted=True,
    )
    # the pair gate saw 125 rows and one violation, and passed
    assert link.pair_refractory_fraction < CONFIG.max_refractory_violation_fraction

    row_family, node_family, active, meta = solve_families(
        observations, [link], inputs, CONFIG
    )
    assert active == []
    assert meta["num_pruned_links"] == 1
    assert link.rejected_because == "exported_train_refractory"
    # The violation itself belongs to the retained sort, so it does not vanish
    # when the link does. What must not happen is it being passed off as clean:
    # the surviving family that still carries it is named, on the exported
    # train, as breaching with no link left to prune.
    assert meta["families_breaching_without_a_prunable_link"]
    assert max(meta["exported_train_refractory_fraction"].values()) > 0.01


def test_a_single_dirty_original_cluster_is_reported_not_silently_passed():
    """No link caused it, so no pruning can fix it; it must not be hidden."""
    dirty = [130.0, 150.0, 150.0002, 170.0, 190.0, 195.0, 195.0002, 200.0]
    inputs = _inputs([(t, 1, 500.0, 12.0, 1) for t in dirty])
    observations = [_observation(1, 1, range(len(dirty)))]
    _, _, active, meta = solve_families(observations, [], inputs, CONFIG)

    assert active == []
    assert meta["families_breaching_without_a_prunable_link"]
    assert max(meta["exported_train_refractory_fraction"].values()) > 0.01


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
    # one cluster carried across the epoch boundary: one family, and it is not
    # a *new* identity, so it is not counted as built from a link
    assert manifest["num_families"] == 1
    assert manifest["families_built_from_a_link"] == []
    assert set(manifest["link_rejections"]) >= {"waveform_cosine", "ambiguous_source"}
    for name in ("epoch_observations.csv", "candidate_links.csv", "family_membership.csv"):
        assert (tmp_path / name).exists()


def test_a_processing_interval_too_short_for_one_epoch_is_a_refusal(tmp_path):
    inputs = _inputs(_train(10.0, 100.0, 20, 0, 500.0, 12.0, 0))
    with pytest.raises(IdentityRefusal, match="no whole"):
        run_unwarped_identity_replay(
            inputs, motion=ABSENT, config=CONFIG,
            processing_interval_s=(0.0, 100.0), output_dir=tmp_path,
        )
