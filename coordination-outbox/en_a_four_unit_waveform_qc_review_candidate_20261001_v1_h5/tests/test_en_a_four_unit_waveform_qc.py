from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).parents[1] / "source" / "en_a_four_unit_waveform_qc.py"
spec = spec_from_file_location("waveform_qc", SOURCE)
qc = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qc)


def geometry():
    return np.c_[np.zeros(384), np.arange(384) * 20.0]


def test_q0_identity_mapping_and_force_zero():
    g = geometry()
    q0 = qc.exact_mapping(g, 0)
    assert np.array_equal(q0, np.arange(384))
    shifted = qc.exact_mapping(g, -40)
    assert np.array_equal(shifted[2:], np.arange(382))
    assert np.array_equal(shifted[:2], [-1, -1])


def test_positive_translation_lands_at_corrected_anchor():
    g = geometry(); original = np.zeros((384, 121), dtype=np.int16)
    original[8, 60] = -100
    corrected = qc.remap_array(original, qc.exact_mapping(g, -40))
    assert corrected[10, 60] == -100
    assert np.count_nonzero(corrected) == 1


def test_wrong_sign_and_no_shift_are_negative_controls():
    g = geometry(); original = np.arange(384 * 121, dtype=np.int32).reshape(384, 121)
    corrected = qc.remap_array(original, qc.exact_mapping(g, -40))
    wrong = qc.remap_array(original, qc.exact_mapping(g, 40))
    assert np.mean(wrong == corrected) < 0.99
    assert np.mean(original == corrected) < 0.99


def test_windows_have_frozen_lengths_and_pair_endpoints():
    signal, baseline = qc.frozen_windows("ordinary_event", 0, 121)
    assert (signal[0], signal[-1], len(signal)) == (40, 100, 61)
    assert len(baseline) == 40
    signal, baseline = qc.frozen_windows("short_pair", 29, 150)
    assert (signal[0], signal[-1], len(signal)) == (40, 129, 90)
    assert len(baseline) == 40


def test_stage_and_robust_metric_known_answer():
    raw = np.zeros((384, 121), dtype=np.int16)
    raw[100, 60] = -60; raw[100, 61] = 60
    stages = qc.stage_arrays(raw)
    assert set(stages) == {"raw_counts", "temporal_centered", "car_centered"}
    signal, baseline = qc.frozen_windows("q0_identity_control", 0, 121)
    metric = qc.per_channel_metrics(stages["car_centered"], signal, baseline)
    assert metric["robust_noise_sigma"][100] == 0
    assert metric["signal_ptp"][100] == 120
    assert metric["ptp_snr"][100] == 120


def test_shape_correlation_identity_and_degenerate():
    x = np.array([0.0, 1.0, -1.0])
    assert np.isclose(qc.shape_correlation(x, x), 1.0)
    assert qc.shape_correlation(np.zeros(3), np.zeros(3)) is None
