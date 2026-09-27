import numpy as np

from testing.bz_identity_helper import (
    aggregate_parent_votes, fit_training_state, percentile_intervals,
    qualify_with_intervals, score_rest_and_episode, score_waveform_pair,
    temporal_block_bootstrap,
)


def _wave(seed=0, events=20, time=21, channels=6, noise=0.01):
    rng = np.random.default_rng(seed)
    t = np.linspace(-2, 2, time)
    base = np.exp(-t[:, None] ** 2) * np.linspace(1, 2, channels)[None]
    return base[None] + noise * rng.normal(size=(events, time, channels))


def _intervals(**changes):
    result = {
        "within_a_cosine": (0.95, 1.0), "within_b_cosine": (0.95, 1.0),
        "cross_minus_within_gap": (-0.01, 0.01), "cross_deficit": (-0.01, 0.01),
        "difference_cosine": (0.0, 1.0),
    }
    result.update(changes)
    return result


def test_point_metrics_do_not_claim_final_label_without_intervals():
    values = [_wave(seed) for seed in range(4)]
    score = score_waveform_pair(*values, noise_std=np.ones(6))
    assert score["eligible"] and score["status"] == "descriptive_metrics_only"
    assert qualify_with_intervals(score, _intervals()) == "limited_waveform_compatibility"
    assert qualify_with_intervals(score, _intervals(
        cross_minus_within_gap=(-0.20, -0.05))) == "inconclusive"


def test_difference_call_requires_reliable_units_and_intervals():
    point = {"eligible": True}
    evidence = _intervals(cross_minus_within_gap=(-0.3, -0.2),
                          cross_deficit=(0.20, 0.30), difference_cosine=(0.90, 0.99))
    assert qualify_with_intervals(point, evidence) == "stable_waveform_difference"
    evidence["within_a_cosine"] = (0.85, 0.99)
    assert qualify_with_intervals(point, evidence) == "inconclusive"


def test_training_state_is_locked_for_heldout_and_episode():
    a1, a2 = _wave(5, noise=0), _wave(6, noise=0)
    b1, b2 = np.zeros_like(a1), np.zeros_like(a2)
    b1[:, :-2] = a1[:, 2:] / 2
    b2[:, :-2] = a2[:, 2:] / 4
    rest = dict(a_half1=a1, b_half1=b1, a_half2=a2, b_half2=b2,
                noise_std=np.ones(6))
    episode = dict(a_half1=a1, b_half1=a1, a_half2=a2, b_half2=a2)
    scores = score_rest_and_episode(rest, episode)
    assert scores["frozen_state"].lag_samples == 2
    assert np.isclose(scores["frozen_state"].b_to_a_gain, 2, atol=3e-3)
    assert scores["episode"]["fitted_lag_samples"] == 2
    assert scores["episode"]["fitted_b_to_a_gain"] == scores["rest"]["fitted_b_to_a_gain"]


def test_heldout_missing_training_selected_support_is_unresolved():
    values = [_wave(seed) for seed in range(7, 11)]
    values[2][:, :, 0] = np.nan
    score = score_waveform_pair(*values, noise_std=np.ones(6))
    assert not score["eligible"] and score["reason"] == "missing_frozen_support"


def test_bootstrap_freezes_lag_gain_and_uses_same_common_domain_for_point():
    values = [_wave(seed, events=22) for seed in range(11, 15)]
    # Blocks 99/199 are noncommon and contain extreme values; point must exclude them.
    blocks_a1 = np.r_[np.arange(10).repeat(2), 99, 99]
    blocks_b1 = np.r_[np.arange(10).repeat(2), 98, 98]
    blocks_a2 = np.r_[np.arange(10, 20).repeat(2), 199, 199]
    blocks_b2 = np.r_[np.arange(10, 20).repeat(2), 198, 198]
    values[0][-2:] *= 100
    result = temporal_block_bootstrap(*values, blocks_a_half1=blocks_a1,
        blocks_b_half1=blocks_b1, blocks_a_half2=blocks_a2, blocks_b_half2=blocks_b2,
        noise_std=np.ones(6), n_bootstrap=12, seed=1)
    state = result["frozen_state"]
    assert result["point"]["n_a_half1"] == 20
    assert result["point"]["common_blocks_half1"] == 10
    assert result["accepted_replicates"] == 12
    # Bootstrap exposes no fitted-state distribution because the state is frozen.
    assert "fitted_lag_samples" not in result["samples"]
    assert state.lag_samples == result["point"]["fitted_lag_samples"]
    assert set(percentile_intervals(result["samples"])) == set(result["samples"])


def test_bootstrap_requires_ten_common_blocks():
    values = [_wave(seed, events=18) for seed in range(15, 19)]
    blocks = np.arange(9).repeat(2)
    try:
        temporal_block_bootstrap(*values, blocks_a_half1=blocks, blocks_b_half1=blocks,
            blocks_a_half2=blocks, blocks_b_half2=blocks, noise_std=np.ones(6),
            n_bootstrap=2)
    except ValueError as error:
        assert str(error) == "insufficient_common_blocks"
    else:
        raise AssertionError("nine common blocks must be unresolved")


def test_noise_control_and_same_parent_vote_is_not_duplicated():
    a1, a2 = _wave(20, noise=0), _wave(21, noise=0)
    b1, b2 = a1.copy(), a2.copy()
    b1[:, :, -1] *= -3
    b2[:, :, -1] *= -3
    state = fit_training_state(a1, b1, noise_std=np.array([1, 1, 1, 1, 1, 100.]))
    assert state.support.sum() == 6
    score = score_waveform_pair(a1, b1, a2, b2, frozen_state=state)
    assert score["heldout_cross_cosine"] > 0.99
    result = aggregate_parent_votes([
        {"parent_a": "a", "parent_b": "a", "heldout_cross_cosine": 0.2},
        {"parent_a": "a", "parent_b": "b", "heldout_cross_cosine": 0.6},
    ])
    assert result["parent_count"] == 2
    assert result["parent_values"]["a"] == 0.4
    assert result["parent_values"]["b"] == 0.6
