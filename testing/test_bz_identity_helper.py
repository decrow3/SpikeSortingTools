import numpy as np

from testing.bz_identity_helper import (
    aggregate_parent_votes,
    score_rest_and_episode,
    score_waveform_pair,
    temporal_block_bootstrap,
)


def _wave(seed=0, events=12, time=21, channels=6, noise=0.01):
    rng = np.random.default_rng(seed)
    t = np.linspace(-2, 2, time)
    base = np.exp(-t[:, None] ** 2) * np.linspace(1, 2, channels)[None]
    return base[None] + noise * rng.normal(size=(events, time, channels))


def test_known_equal_is_compatible_and_noise_weighted():
    a1, a2 = _wave(1), _wave(2)
    b1, b2 = _wave(3), _wave(4)
    score = score_waveform_pair(a1, b1, a2, b2, noise_std=np.arange(1, 7))
    assert score["eligible"]
    assert score["status"] == "limited_waveform_compatibility"
    assert score["heldout_cross_cosine"] > 0.99


def test_known_difference_has_stable_signed_difference():
    a1, a2 = _wave(5), _wave(6)
    b1, b2 = _wave(7), _wave(8)
    pulse = np.zeros_like(b1)
    pulse[:, 7:11, 0:2] = 2.0
    b1, b2 = b1 + pulse, b2 + pulse
    score = score_waveform_pair(a1, b1, a2, b2, noise_std=np.ones(6))
    assert score["status"] == "stable_waveform_difference"
    assert score["difference_cosine"] > 0.95


def test_half1_gain_and_lag_are_locked_for_half2():
    a1, a2 = _wave(9, noise=0.0), _wave(10, noise=0.0)
    b1 = np.zeros_like(a1)
    b2 = np.zeros_like(a2)
    b1[:, :-2] = a1[:, 2:] / 2.0
    b2[:, :-2] = a2[:, 2:] / 4.0  # held-out gain differs; it must not be refit
    score = score_waveform_pair(a1, b1, a2, b2, noise_std=np.ones(6))
    assert score["fitted_lag_samples"] == 2  # inclusive lag-search boundary
    assert np.isclose(score["fitted_b_to_a_gain"], 2.0, atol=3e-3)
    assert score["heldout_residual_rms"] > 0.1


def test_high_noise_channel_is_downweighted():
    a1, a2 = _wave(20, noise=0.0), _wave(21, noise=0.0)
    b1, b2 = a1.copy(), a2.copy()
    b1[:, :, -1] *= -3
    b2[:, :, -1] *= -3
    unweighted = score_waveform_pair(a1, b1, a2, b2, noise_std=np.ones(6))
    weighted = score_waveform_pair(
        a1, b1, a2, b2, noise_std=np.array([1, 1, 1, 1, 1, 100.0])
    )
    assert not unweighted["eligible"]
    assert weighted["eligible"]
    assert weighted["heldout_cross_cosine"] > 0.99


def test_support_and_episode_eligibility_are_explicit():
    data = _wave(11)
    bad = score_waveform_pair(
        data, data, data, data, noise_std=np.ones(6),
        support=np.array([1, 1, 1, 0, 0, 0], bool), min_supported_channels=4,
    )
    assert not bad["eligible"]
    assert bad["reason"] == "insufficient_finite_support"
    rest = dict(a_half1=data, b_half1=data, a_half2=data, b_half2=data,
                noise_std=np.ones(6))
    strata = score_rest_and_episode(rest)
    assert strata["rest"]["eligible"] and not strata["episode"]["eligible"]


def test_bootstrap_resamples_blocks_and_is_deterministic():
    values = [_wave(seed, events=8) for seed in range(12, 16)]
    blocks = np.repeat([0, 1], 4)
    kwargs = dict(
        blocks_a_half1=blocks, blocks_b_half1=blocks,
        blocks_a_half2=blocks + 2, blocks_b_half2=blocks + 2,
        noise_std=np.ones(6), n_bootstrap=8, seed=17,
    )
    first = temporal_block_bootstrap(*values, **kwargs)
    second = temporal_block_bootstrap(*values, **kwargs)
    assert first["accepted_replicates"] == 8
    assert np.array_equal(first["fitted_lag_samples"], second["fitted_lag_samples"])
    assert np.allclose(first["heldout_cross_cosine"], second["heldout_cross_cosine"])


def test_parent_aggregation_does_not_count_edges_as_independent_votes():
    rows = [
        {"parent_a": "a", "parent_b": "b", "heldout_cross_cosine": 0.2},
        {"parent_a": "a", "parent_b": "c", "heldout_cross_cosine": 0.4},
        {"parent_a": "a", "parent_b": "d", "heldout_cross_cosine": 0.6},
    ]
    result = aggregate_parent_votes(rows)
    assert result["parent_count"] == 4
    assert result["parent_values"]["a"] == 0.4
    assert result["parent_equal_weight_median"] == 0.4
