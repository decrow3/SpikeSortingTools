import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


TESTING = Path(__file__).parent
sys.path.insert(0, str(TESTING))
spec = importlib.util.spec_from_file_location(
    "analyze_reference_safety", TESTING / "analyze_reference_safety.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class ReferenceSafetyTests(unittest.TestCase):
    def test_alpha_recovers_subtracted_fraction(self):
        x = np.array([0.0, -2.0, -4.0, -2.0, 0.0])
        self.assertAlmostEqual(mod.waveform_alpha(0.2 * x, x), 0.2)

    def test_negative_alpha_means_reference_increases_waveform(self):
        x = np.array([0.0, -1.0, -3.0, -1.0, 0.0])
        self.assertAlmostEqual(mod.waveform_alpha(-0.1 * x, x), -0.1)

    def test_footprint_width_tracks_spatial_spread(self):
        depths = np.array([0.0, 20.0, 40.0])
        narrow = np.zeros((5, 3)); narrow[2, 1] = 1
        broad = np.zeros((5, 3)); broad[2, :] = 1
        self.assertEqual(mod.footprint_width_um(narrow, depths), 0.0)
        self.assertGreater(mod.footprint_width_um(broad, depths), 10.0)

    def test_baseline_waveform_removes_offset(self):
        waveform = np.array([[5.0], [5.0], [4.0], [2.0], [4.0]])
        out = mod.baseline_waveform(waveform, 2)
        np.testing.assert_allclose(out[:2].mean(axis=0), 0.0)

    def test_waveform_correlation_is_a_scalar_metric(self):
        self.assertNotIn("waveform_correlation", mod.WAVEFORM_ARRAY_KEYS)


if __name__ == "__main__":
    unittest.main()
