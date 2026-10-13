import h5py
import numpy as np
import pytest

from testing.joint_model_matching1_adapter import (
    KILOSORT_FIELDS_NOT_APPLICABLE,
    materialize_times_labels,
)


def test_saved_arm_round_trip_is_exact_and_fail_closed(tmp_path):
    matching = tmp_path / "matching1.h5"
    expected_times = np.asarray([217620009, 217637561, 217637561, 217649990], dtype=np.int64)
    expected_labels = np.asarray([12, -1, 8, 12], dtype=np.int64)
    with h5py.File(matching, "w") as h5:
        h5.create_dataset("times_samples", data=expected_times)
        h5.create_dataset("labels", data=expected_labels)
        h5.create_dataset("amplitudes", data=np.asarray([9.0, 8.0, 7.0, 6.0]))

    output = tmp_path / "adapted"
    times_path, labels_path = materialize_times_labels(matching, output)
    assert np.array_equal(np.load(times_path, allow_pickle=False), expected_times)
    assert np.array_equal(np.load(labels_path, allow_pickle=False), expected_labels)
    assert [path.name for path in sorted(output.iterdir())] == [
        "spike_clusters.npy",
        "spike_times.npy",
    ]
    assert set(KILOSORT_FIELDS_NOT_APPLICABLE) == {
        "amplitudes.npy", "spike_positions.npy", "cluster_KSLabel.tsv"
    }
    with pytest.raises(FileExistsError):
        materialize_times_labels(matching, output)

    malformed = tmp_path / "malformed.h5"
    with h5py.File(malformed, "w") as h5:
        h5.create_dataset("times_samples", data=np.asarray([1.0, 2.0]))
        h5.create_dataset("labels", data=np.asarray([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="integer dtype"):
        materialize_times_labels(malformed, tmp_path / "malformed-output")
