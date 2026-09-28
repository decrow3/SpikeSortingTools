import numpy as np

from testing.luke_imec1_compact_alignment_pilot_v2 import best_lag_scores, shift_waveforms


def test_best_lag_is_retained_and_alignment_improves_score():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(49, 16)).astype("float32")
    observed = np.zeros_like(reference); observed[2:] = reference[:-2]
    score, lag = best_lag_scores(observed[None], reference / np.linalg.norm(reference))
    aligned = shift_waveforms(observed[None], lag)
    before = np.dot(observed.ravel(), reference.ravel()) / (np.linalg.norm(observed) * np.linalg.norm(reference))
    after = np.dot(aligned.ravel(), reference.ravel()) / (np.linalg.norm(aligned) * np.linalg.norm(reference))
    assert lag[0] == -2
    assert score[0] > before + .8
    assert after > .97


def test_shift_roundtrip_preserves_interior():
    wave = np.arange(49 * 2).reshape(1, 49, 2)
    left = shift_waveforms(wave, np.array([-2]))
    restored = shift_waveforms(left, np.array([2]))
    np.testing.assert_array_equal(restored[:, 2:-2], wave[:, 2:-2])
