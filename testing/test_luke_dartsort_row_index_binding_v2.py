from pathlib import Path

import h5py
import numpy as np
import pytest

from dartsort.util.data_util import DARTsortSorting
from testing.luke_dartsort_row_index_binding_v2 import validate_row_index_binding


def _fixture(root: Path, channels=(4, 5, 6)) -> tuple[Path, Path]:
    h5_path = root / "parent.h5"
    with h5py.File(h5_path, "w") as h5:
        h5["times_samples"] = np.array([100, 200, 300], dtype=np.int64)
        h5["channels"] = np.array([4, 5, 6], dtype=np.int64)
        h5["labels"] = np.array([10, 11, 12], dtype=np.int32)
        h5["sampling_frequency"] = 30_000.0
        h5["geom"] = np.array([[0., 0.], [0., 20.]], dtype=np.float64)
        h5["denoised_ptp_amplitudes"] = np.array([11., 22., 33.], dtype=np.float32)
        h5["point_source_localizations"] = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
    npz_path = root / "sorting.npz"
    np.savez(
        npz_path,
        times_samples=np.array([102, 199, 301], dtype=np.int64),
        channels=np.array(channels, dtype=np.int64),
        labels=np.array([2, 0, 2], dtype=np.int32),
        sampling_frequency=np.array(30_000.0),
        geom=np.array([[0., 0.], [0., 20.]], dtype=np.float64),
        parent_h5_path=np.array("parent.h5"),
        ephemeral_feature_names=np.array([], dtype="U1"),
        loaded_persistent_features=np.array(["denoised_ptp_amplitudes", "point_source_localizations"]),
    )
    return npz_path, h5_path


def test_actual_loader_preserves_feature_rows_with_final_core_arrays(tmp_path):
    npz_path, h5_path = _fixture(tmp_path)
    sorting = DARTsortSorting.load(npz_path)
    assert np.array_equal(sorting.times_samples, [102, 199, 301])
    assert np.array_equal(sorting.labels, [2, 0, 2])
    assert np.array_equal(sorting.denoised_ptp_amplitudes, [11., 22., 33.])
    assert np.array_equal(sorting.point_source_localizations[:, 2], [3., 6., 9.])
    assert validate_row_index_binding(npz_path, h5_path)["channels_equal"] is True


def test_deliberate_channel_misbinding_is_rejected(tmp_path):
    npz_path, h5_path = _fixture(tmp_path, channels=(5, 4, 6))
    with pytest.raises(ValueError, match="channels mismatch"):
        validate_row_index_binding(npz_path, h5_path)
