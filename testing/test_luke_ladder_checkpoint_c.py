import numpy as np

from testing.luke_ladder_checkpoint_c import DONOR_PLAN, RATES_HZ, _inject

FS = 30_000.0


def test_inject_places_every_donor_and_returns_matching_truth():
    rng = np.random.default_rng(0)
    bg = rng.normal(0, 12, (int(FS * 20), 112)).astype(np.float32)
    injected, truth = _inject(bg, FS)

    assert injected.shape == bg.shape
    assert set(truth) == {f"inj{i}" for i in range(len(DONOR_PLAN))}
    # each injected unit fires near its requested rate
    for i, rate in enumerate(RATES_HZ):
        n = truth[f"inj{i}"].size
        assert 0.7 * rate < n / 18.0 < 1.3 * rate  # ~18 s usable window
    # the injection changed the voltage only where donors sit (localised)
    changed = np.abs(injected - bg).max(axis=0) > 1.0
    assert 20 < changed.sum() < 112  # not the whole strip


def test_inject_is_deterministic():
    bg = np.zeros((int(FS * 10), 112), dtype=np.float32)
    a, ta = _inject(bg.copy(), FS)
    b, tb = _inject(bg.copy(), FS)
    assert np.array_equal(a, b)
    assert all(np.array_equal(ta[k], tb[k]) for k in ta)
