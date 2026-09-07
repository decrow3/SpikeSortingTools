import numpy as np

from testing.luke_motion_conditioned_screen import ratio, select_pairs


def test_quiet_bins_are_local_and_never_reused():
    dose = np.array([0., 10., 11., 1., 2., 12., 13., 3., 4., 14., 15., 5.])
    pairs = select_pairs(dose, radius=2)
    assert pairs
    assert len({q for h, q in pairs}) == len(pairs)
    for h, q in pairs:
        assert abs(h-q) <= 2
        assert dose[h] >= np.quantile(dose, .75)
        assert dose[q] <= np.quantile(dose, .25)


def test_stationary_motion_does_not_create_contrasts():
    assert select_pairs(np.ones(20)) == []


def test_rate_screen_distinguishes_rescue_and_undefined_dropout():
    # Same quiet rate, half the off spikes lost during motion, rigid recovers.
    assert ratio(ratio(80, 80), ratio(40, 80)) == 2
    assert ratio(0, 80) == 0
    # A ratio to a completely silent reference is unmeasurable, not infinite gain.
    assert np.isnan(ratio(80, 0))
