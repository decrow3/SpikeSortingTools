import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


try:
    from spikeinterface.core import NumpyRecording
    from spikeinterface.core.motion import Motion
    from pipeline.motion import correct_motion_lfp
    from pipeline.motion import _validate_lfp_displacement
    from pipeline.preprocess import paired_lfp_stream_id, prepare_lfp_for_motion
except ModuleNotFoundError as exc:  # Allow the lightweight development shell to collect tests.
    NumpyRecording = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f'SpikeInterface environment unavailable: {IMPORT_ERROR}')
class LfpMotionPreprocessingTests(unittest.TestCase):
    def test_paired_lfp_stream_id(self):
        self.assertEqual(paired_lfp_stream_id('imec0.ap'), 'imec0.lf')
        with self.assertRaises(ValueError):
            paired_lfp_stream_id('imec0.lf')

    def test_preprocessing_averages_duplicate_depths_and_resamples(self):
        recording = self._make_recording()

        prepared = prepare_lfp_for_motion(
            recording,
            target_sampling_frequency=250,
            freq_min=0.5,
            freq_max=100.0,
            detect_bad_lfp_channels=False,
        )

        self.assertEqual(prepared.get_num_channels(), 2)
        self.assertEqual(prepared.get_sampling_frequency(), 250)
        np.testing.assert_array_equal(prepared.get_channel_locations()[:, 1], [0.0, 20.0])
        self.assertEqual(prepared.get_traces().shape, (4 * 250, 2))

    def test_cross_band_correction_saves_separate_lfp_cache(self):
        recording = self._make_recording()
        prepared = prepare_lfp_for_motion(
            recording, target_sampling_frequency=250, freq_min=0.5, freq_max=100.0,
            detect_bad_lfp_channels=False,
        )
        times = prepared.get_times()
        fake_motion = Motion(
            displacement=np.zeros((times.size, 2)),
            temporal_bins_s=times,
            spatial_bins_um=np.array([0.0, 20.0]),
        )

        with TemporaryDirectory() as tmpdir, patch(
            'pipeline.motion.estimate_motion', return_value=fake_motion
        ) as estimate:
            corrected = correct_motion_lfp(
                prepared, recording, Path(tmpdir), recalc=True,
            )
            self.assertEqual(corrected.dtype, np.dtype('int16'))
            self.assertTrue((Path(tmpdir) / 'dredge-lfp-motion' / 'config.json').exists())
            self.assertEqual(estimate.call_args.kwargs['method'], 'dredge_lfp')

    def test_discontinuous_motion_is_rejected(self):
        displacement = np.array([[0.0], [80.0], [0.0]])
        with self.assertRaisesRegex(ValueError, 'underconstrained'):
            _validate_lfp_displacement(displacement, max_step_um=40.0)

    @staticmethod
    def _make_recording():
        fs = 1000
        duration_s = 4
        time = np.arange(fs * duration_s) / fs
        traces = np.column_stack([
            np.sin(2 * np.pi * 8 * time + phase)
            for phase in (0.0, 0.1, 0.2, 0.3)
        ]).astype('float32')
        recording = NumpyRecording(traces_list=[traces], sampling_frequency=fs)
        recording.set_channel_locations(np.array([
            [0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0],
        ]))
        return recording


if __name__ == '__main__':
    unittest.main()
