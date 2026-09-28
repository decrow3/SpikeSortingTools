import numpy as np

from testing.luke_imec1_dots_sorterfree_waveform_discovery_v3 import classify_scores, score_global


RELATIVE_GEOMETRY = np.array([
    [0, -60], [32, -60], [16, -40], [48, -40],
    [0, -20], [32, -20], [16, 0], [48, 0],
    [0, 20], [32, 20], [16, 40], [48, 40],
    [0, 60], [32, 60], [16, 80], [48, 80],
], dtype=float)


def test_global_cross_phase_score_and_gain():
    rng = np.random.default_rng(3)
    template = rng.normal(size=(49, 16)).astype(np.float32)
    template /= np.linalg.norm(template)
    event = np.zeros((49, 16), np.float32)
    event_channels = [i for i, v in enumerate(RELATIVE_GEOMETRY - RELATIVE_GEOMETRY[8])
                      if tuple(v) in {tuple(x) for x in RELATIVE_GEOMETRY - RELATIVE_GEOMETRY[6]}]
    template_lookup = {tuple(v): i for i, v in enumerate(RELATIVE_GEOMETRY - RELATIVE_GEOMETRY[6])}
    for i in event_channels:
        event[:, i] = 2 * template[:, template_lookup[tuple(RELATIVE_GEOMETRY[i] - RELATIVE_GEOMETRY[8])]]
    scored = score_global(event[None], np.array([8]), template[None], np.array([6]), np.array([1.0]), RELATIVE_GEOMETRY)
    assert scored["winner_kind"][0] == 0
    assert scored["score"][0] > .99999
    assert 1.99 < scored["gain"][0] < 2.01
    assert classify_scores(scored)[0] == "strict"


def test_gain_gate_rejects_large_amplitude_mismatch():
    rng = np.random.default_rng(4)
    template = rng.normal(size=(49, 16)).astype(np.float32)
    template /= np.linalg.norm(template)
    scored = score_global((4 * template)[None], np.array([6]), template[None], np.array([6]), np.array([1.0]), RELATIVE_GEOMETRY)
    assert classify_scores(scored)[0] == "identity_ambiguous" or classify_scores(scored)[0] == "unmatched"
