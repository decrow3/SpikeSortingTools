import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TESTING = Path(__file__).parent
sys.path.insert(0, str(TESTING))
spec = importlib.util.spec_from_file_location(
    "analyze_cross_probe_common", TESTING / "analyze_cross_probe_common.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class CrossProbeCommonTests(unittest.TestCase):
    def test_short_xcorr_recovers_known_lag(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=5000)
        b = np.roll(a, 4)
        corr, lag_ms = mod.normalized_short_xcorr(a, b, fs=1000, max_lag_ms=10)
        self.assertGreater(corr, 0.99)
        self.assertAlmostEqual(abs(lag_ms), 4.0)

    def test_band_mean(self):
        freq = np.arange(10.0)
        values = np.arange(10.0)
        self.assertEqual(mod.band_mean(freq, values, 2, 5), 3.0)

    def test_sync_mapping_corrects_offset_and_clock_drift(self):
        source = np.array([10.0, 110.0, 210.0, 310.0])
        target = np.array([20.0, 121.0, 222.0, 323.0])
        mapped = mod.map_sample_frames(np.array([60.0, 260.0]), source, target)
        np.testing.assert_allclose(mapped, [70.5, 272.5])

    def test_sync_mapping_extrapolates_short_end_intervals(self):
        mapped = mod.map_sample_frames(np.array([0.0, 3.0]),
                                       np.array([1.0, 2.0]),
                                       np.array([3.0, 5.0]))
        np.testing.assert_allclose(mapped, [1.0, 7.0])


if __name__ == "__main__":
    unittest.main()
