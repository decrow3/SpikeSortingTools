"""Unit tests for testing/ladder_unwarped_identity.py (Option B candidate)."""

import numpy as np
import pytest

from testing.ladder_unwarped_identity import (
    AcceptedLink,
    EpochObservation,
    UnwarpedIdentityConfig,
    build_candidate_links,
    compute_refractory_violation_rate,
    extract_epoch_observations,
    run_unwarped_identity_pipeline,
    solve_identity_tracks,
)


def test_refractory_violation_rate():
    times = np.array([0.0, 0.001, 0.005, 0.0055, 0.010])
    rate = compute_refractory_violation_rate(times, refractory_period_ms=1.5)
    # diffs: 0.001 (1ms < 1.5ms), 0.004 (4ms), 0.0005 (0.5ms < 1.5ms), 0.0045 (4.5ms)
    # Total diffs: 4. Violations: 2. Rate: 2/4 = 0.5
    assert abs(rate - 0.5) < 1e-6


def test_extract_epoch_observations_and_links(tmp_path):
    config = UnwarpedIdentityConfig(
        epoch_duration_s=10.0,
        epoch_overlap_s=2.0,
        max_spatial_distance_um=20.0,
        min_spikes_per_epoch=5,
    )

    # Synthetic neuron 1 moving smoothly
    np.random.seed(42)
    t1_ep0 = np.linspace(0.5, 9.5, 20)
    t1_ep1 = np.linspace(8.5, 17.5, 20)

    times = np.concatenate([t1_ep0, t1_ep1])
    clusters = np.zeros(len(times), dtype=int)
    depths = np.concatenate([np.full_like(t1_ep0, 100.0), np.full_like(t1_ep1, 105.0)])
    displacements = np.zeros_like(times)

    obs = extract_epoch_observations(
        times, clusters, depths, depths, recording_duration_s=18.0, config=config
    )
    assert len(obs) >= 2

    links = build_candidate_links(obs, times, clusters, config)
    assert len(links) >= 1

    tracks, clean_links, meta = solve_identity_tracks(obs, links, times, clusters, config)
    assert len(set(tracks.values())) == 1

    res = run_unwarped_identity_pipeline(
        times, clusters, depths, displacements, recording_duration_s=18.0, output_dir=tmp_path, config=config
    )
    assert (tmp_path / "unwarped_identity_manifest.json").exists()
    assert (tmp_path / "spike_tracks.npy").exists()
    assert res["manifest"]["num_unique_tracks"] == 1


def test_refractory_gate_refuses_overlapping_duplicate_clusters():
    config = UnwarpedIdentityConfig(
        epoch_duration_s=10.0,
        epoch_overlap_s=2.0,
        max_spatial_distance_um=20.0,
        max_refractory_violation_fraction=0.01,
        min_spikes_per_epoch=5,
    )

    # Two duplicate clusters firing almost simultaneously in epoch 0 and 1
    t1 = np.linspace(8.0, 9.9, 20)
    t2 = t1 + 0.0005  # 0.5 ms shift -> triggers refractory violation on union

    times = np.concatenate([t1, t2])
    clusters = np.concatenate([np.zeros(len(t1), dtype=int), np.ones(len(t2), dtype=int)])

    obs = [
        EpochObservation(
            epoch_idx=0, cluster_id=0, start_s=0.0, stop_s=10.0, num_spikes=len(t1),
            firing_rate_hz=2.0, mean_observed_depth_um=100.0, mean_tissue_depth_um=100.0,
            mean_amplitude=1.0, refractory_violation_rate=0.0
        ),
        EpochObservation(
            epoch_idx=1, cluster_id=1, start_s=8.0, stop_s=18.0, num_spikes=len(t2),
            firing_rate_hz=2.0, mean_observed_depth_um=100.0, mean_tissue_depth_um=100.0,
            mean_amplitude=1.0, refractory_violation_rate=0.0
        ),
    ]

    links = build_candidate_links(obs, times, clusters, config)
    # Should fail link due to union refractory violation
    assert len(links) == 0


def test_ambiguity_handling_filters_competing_links():
    config = UnwarpedIdentityConfig(
        epoch_duration_s=10.0,
        epoch_overlap_s=2.0,
        max_spatial_distance_um=30.0,
        ambiguity_threshold_ratio=0.85,
        min_spikes_per_epoch=5,
    )

    t0 = np.linspace(0.5, 9.5, 20)
    t1a = np.linspace(8.5, 17.5, 20)
    t1b = np.linspace(8.5, 17.5, 20) + 0.1

    times = np.concatenate([t0, t1a, t1b])
    clusters = np.concatenate([np.zeros(20, dtype=int), np.ones(20, dtype=int), np.full(20, 2, dtype=int)])

    obs = [
        EpochObservation(epoch_idx=0, cluster_id=0, start_s=0.0, stop_s=10.0, num_spikes=20, firing_rate_hz=2.0, mean_observed_depth_um=100.0, mean_tissue_depth_um=100.0, mean_amplitude=10.0, refractory_violation_rate=0.0),
        EpochObservation(epoch_idx=1, cluster_id=1, start_s=8.0, stop_s=18.0, num_spikes=20, firing_rate_hz=2.0, mean_observed_depth_um=101.0, mean_tissue_depth_um=101.0, mean_amplitude=10.0, refractory_violation_rate=0.0),
        EpochObservation(epoch_idx=1, cluster_id=2, start_s=8.0, stop_s=18.0, num_spikes=20, firing_rate_hz=2.0, mean_observed_depth_um=101.5, mean_tissue_depth_um=101.5, mean_amplitude=10.0, refractory_violation_rate=0.0),
    ]

    links = build_candidate_links(obs, times, clusters, config)
    # Both cluster 1 and cluster 2 in epoch 1 compete for cluster 0 with near-identical scores -> filtered as ambiguous
    assert len(links) == 0
