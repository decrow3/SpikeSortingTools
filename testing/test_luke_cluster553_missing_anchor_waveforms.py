import numpy as np

from testing.luke_cluster553_missing_anchor_waveforms import evenly_bounded, waveform_metrics


def test_evenly_bounded_preserves_edges_and_limit():
    selected = evenly_bounded(np.arange(20), 5)
    assert selected.tolist() == [0, 4, 9, 14, 19]


def test_waveform_metrics_recovers_reference_shape():
    wave = np.zeros((9, 2)); wave[4, 0] = -3; wave[5, 1] = 1
    metrics, median = waveform_metrics(np.stack([wave, wave]), wave, 2)
    assert metrics["sampled_events"] == 2
    assert np.isclose(metrics["reference_cosine"], 1)
    assert metrics["baseline_noise_uv"] == 0
    assert median.shape == wave.shape
