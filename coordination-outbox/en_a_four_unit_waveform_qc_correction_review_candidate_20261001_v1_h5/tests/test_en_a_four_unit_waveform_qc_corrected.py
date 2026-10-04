import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).parents[1] / "source/en_a_four_unit_waveform_qc_corrected.py"
SPEC = importlib.util.spec_from_file_location("qc_corrected", SOURCE)
QC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QC)


def geometry():
    # A synthetic 384-site two-column 20-um-row lattice.
    return np.c_[np.tile([0.0, 16.0], 192), np.repeat(np.arange(192) * 20.0, 2)]


def synthetic(states):
    g = geometry(); states = np.asarray(states, dtype=float)
    original = np.zeros((384, len(states)), dtype=np.int16)
    anchor = 200
    maps = {float(q): QC.exact_mapping(g, float(q)) for q in np.unique(np.r_[states, -states])}
    for col, q in enumerate(states):
        source = maps[float(q)][anchor]
        if source >= 0:
            original[source, col] = -100
    corrected, _ = QC.remap_per_sample(original, states, maps)
    return g, anchor, original, corrected, maps


def test_interior_state_change_per_sample_remap():
    states = np.r_[np.zeros(5), np.full(6, -40.0)]
    _, anchor, original, corrected, maps = synthetic(states)
    replay, _ = QC.remap_per_sample(original, states, maps)
    assert np.array_equal(replay, corrected)
    assert np.all(corrected[anchor] == -100)


def test_exact_edge_belongs_to_later_cell():
    centers = np.array([0.0, 0.25, 0.5])
    states = np.array([0.0, -40.0, -80.0])
    sampled = QC.sample_stepwise_states(centers, states, np.array([0.124999, 0.125, 0.375]), 0.25)
    assert sampled.tolist() == [0.0, -40.0, -80.0]


def test_unchanged_state_remap_and_q0_identity():
    states = np.zeros(9)
    _, _, original, corrected, maps = synthetic(states)
    replay, _ = QC.remap_per_sample(original, states, maps)
    assert np.array_equal(replay, original)
    assert np.array_equal(corrected, original)


def test_strong_distant_distractor_does_not_select_local_domain():
    g = geometry(); anchor = 200; local = QC.local_domain(g, anchor, 80.0)
    values = np.zeros((384, 121), dtype=float)
    values[anchor, 60] = -12
    distant = 20
    assert distant not in local
    values[distant, 60] = -1000
    signal, baseline = QC.frozen_windows("ordinary_event", 0, 121)
    result = QC.local_detectability(QC.channel_metrics(values, signal, baseline), local, 6.0)
    assert result["locally_detectable"] is True
    assert result["local_peak_channel"] == anchor
    assert result["global_peak_channel_descriptive"] == distant


def test_distant_only_is_not_locally_detectable():
    g = geometry(); anchor = 200; local = QC.local_domain(g, anchor, 80.0)
    values = np.zeros((384, 121), dtype=float); values[20, 60] = -1000
    signal, baseline = QC.frozen_windows("ordinary_event", 0, 121)
    result = QC.local_detectability(QC.channel_metrics(values, signal, baseline), local, 6.0)
    assert result["locally_detectable"] is False
    assert result["global_peak_channel_descriptive"] == 20


def test_negative_controls_are_nonvacuous_or_explicitly_disabled():
    states = np.r_[np.zeros(5), np.full(6, -40.0)]
    _, _, original, corrected, maps = synthetic(states)
    expected, _ = QC.remap_per_sample(original, states, maps)
    no_shift = original
    wrong_sign, _ = QC.remap_per_sample(original, -states, maps)
    assert QC.negative_control(expected, corrected, no_shift, states, "no_shift")["nonvacuous"]
    assert QC.negative_control(expected, corrected, wrong_sign, states, "wrong_sign")["nonvacuous"]
    q0 = np.zeros(states.size)
    q0_expected, _ = QC.remap_per_sample(original, q0, maps)
    assert QC.negative_control(q0_expected, q0_expected, original, q0, "q0")["enabled"] is False
