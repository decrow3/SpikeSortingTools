import numpy as np
import pytest
import torch

from testing.joint_model_reader_injection import inject_waveforms_before_centering_car


def test_pre_centering_injection_boundaries_overlap_and_fail_closed():
    source = torch.zeros((2, 8), dtype=torch.float32)
    waveforms = np.asarray([
        [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]],
        [[7, 7, 7, 7, 7], [9, 9, 9, 9, 9]],
        [[4, 3, 2, 1, 0], [8, 6, 4, 2, 0]],
    ], dtype=np.float32)
    result = inject_waveforms_before_centering_car(
        source,
        padded_start_sample=-2,
        recording_num_samples=6,
        event_samples=np.asarray([0, 2, 5], dtype=np.int64),
        event_waveforms=waveforms,
        trough_offset_samples=2,
    )
    expected = torch.tensor([
        [0, 0, 10, 11, 12, 11, 10, 2],
        [0, 0, 39, 49, 59, 17, 15, 4],
    ], dtype=torch.float32)
    assert torch.equal(result, expected)
    assert torch.count_nonzero(source) == 0

    with pytest.raises(ValueError, match="align"):
        inject_waveforms_before_centering_car(
            source,
            padded_start_sample=-2,
            recording_num_samples=6,
            event_samples=np.asarray([1], dtype=np.int64),
            event_waveforms=np.zeros((2, 2, 5), dtype=np.float32),
            trough_offset_samples=2,
        )
    with pytest.raises(ValueError, match="outside the recording"):
        inject_waveforms_before_centering_car(
            source,
            padded_start_sample=-2,
            recording_num_samples=6,
            event_samples=np.asarray([6], dtype=np.int64),
            event_waveforms=np.zeros((1, 2, 5), dtype=np.float32),
            trough_offset_samples=2,
        )
