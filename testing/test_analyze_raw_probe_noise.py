import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze_raw_probe_noise.py")
SPEC = importlib.util.spec_from_file_location("analyze_raw_probe_noise", MODULE_PATH)
noise = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(noise)


class RawProbeNoiseTests(unittest.TestCase):
    def test_robust_scale_recovers_gaussian_sigma_with_spike_outliers(self):
        rng = np.random.default_rng(4)
        traces = rng.normal(0, 7.0, size=(200_000, 3))
        traces[::1000, 0] -= 100.0
        estimate = noise.robust_scale(traces, axis=0)
        np.testing.assert_allclose(estimate, [7, 7, 7], rtol=0.025)

    def test_deterministic_windows_respect_edges(self):
        starts = noise.deterministic_window_starts(
            n_frames=30_000, sampling_frequency=1000,
            window_duration_s=2, n_windows=5, edge_margin_s=1,
        )
        np.testing.assert_array_equal(starts, [1000, 7500, 14000, 20500, 27000])

    def test_local_reference_removes_shared_signal(self):
        t = np.linspace(0, 1, 4000, endpoint=False)
        common = 20 * np.sin(2 * np.pi * 300 * t)
        traces = common[:, None] + np.array([1, -1, 2, -2])[None, :]
        positions = np.column_stack((np.zeros(4), np.arange(4) * 20.0))
        neighbors = noise.make_local_neighbor_indices(positions, 1, 100)
        referenced = noise.local_median_reference(traces, neighbors)
        self.assertLess(float(np.std(np.median(referenced, axis=1))), 1e-10)

    def test_excursion_mask_dilates_and_detects_global_event(self):
        traces = np.zeros((100, 4))
        traces[50, :2] = 20
        mask, global_mask = noise.dilated_excursion_mask(
            traces, np.ones(4), threshold_sigma=5,
            dilation_samples=2, global_participation_fraction=0.5,
        )
        self.assertTrue(np.all(global_mask[48:53]))
        self.assertTrue(np.all(mask[48:53]))
        self.assertFalse(global_mask[47])

    def test_masked_scale_reports_removed_fraction(self):
        rng = np.random.default_rng(9)
        traces = rng.normal(size=(10_000, 2))
        mask = np.zeros_like(traces, dtype=bool)
        mask[:100, 0] = True
        _, fraction = noise.masked_robust_scale(traces, mask)
        np.testing.assert_allclose(fraction, [0.01, 0.0])

    def test_contiguous_intervals_are_half_open(self):
        intervals = noise.contiguous_intervals(
            np.array([False, True, True, False, True]), start_s=10, fs=10
        )
        self.assertEqual(intervals, [(10.1, 10.3), (10.4, 10.5)])


if __name__ == "__main__":
    unittest.main()
