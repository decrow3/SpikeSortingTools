from pathlib import Path

import h5py
import numpy as np
import torch

from testing.joint_model_chunk_starts_entry import run_selected_chunks
from testing.joint_model_donor_selector import select_donors
from testing.joint_model_matching1_adapter import materialize_times_labels
from testing.joint_model_reader_injection import inject_waveforms_before_centering_car


def test_a4_combined_boundary_moving_fixed_overlap_fixture(tmp_path):
    units = np.arange(100, 118, dtype=np.int64)
    donor_ledger = select_donors(
        units,
        np.arange(1, 19, dtype=np.float64),
        np.tile(np.asarray([.10, .20, .30, .40, .50, .60]), 3),
        np.asarray([10, 300, 300, 300, 300, 300, 20, 300, 300, 300, 300, 300, 30, 40, 300, 300, 300, 300], dtype=float),
    )
    selected = {row["unit_id"] for row in donor_ledger}
    assert {100, 106}.issubset(selected)

    switch = 217637561
    event_samples = np.asarray([switch - 1, switch, switch], dtype=np.int64)
    event_labels = np.asarray([100, 100, 106], dtype=np.int64)
    event_kind = np.asarray(["moving-left", "moving-right", "fixed-overlap"])
    waveforms = np.zeros((3, 3, 5), dtype=np.float32)
    waveforms[0, 0, 2] = 5.0
    waveforms[1, 1, 2] = 5.0
    waveforms[2, 2, 2] = 7.0
    injected = inject_waveforms_before_centering_car(
        torch.zeros((3, 12), dtype=torch.float32),
        padded_start_sample=switch - 5,
        recording_num_samples=314204894,
        event_samples=event_samples,
        event_waveforms=waveforms,
        trough_offset_samples=2,
    )
    assert torch.equal(injected[:, 4], torch.tensor([5.0, 0.0, 0.0]))
    assert torch.equal(injected[:, 5], torch.tensor([0.0, 5.0, 7.0]))
    assert event_kind.tolist() == ["moving-left", "moving-right", "fixed-overlap"]
    assert event_samples[0] < switch <= event_samples[1]

    calls = []

    def existing_c4_runner(*, chunk_starts_samples, feature_switches, output_dir):
        calls.append((chunk_starts_samples.copy(), dict(feature_switches), Path(output_dir)))
        Path(output_dir).mkdir()
        with h5py.File(Path(output_dir) / "matching1.h5", "w") as h5:
            h5.create_dataset("times_samples", data=event_samples)
            h5.create_dataset("labels", data=event_labels)
        return Path(output_dir) / "matching1.h5"

    arm_dir = tmp_path / "n1"
    matching = run_selected_chunks(
        [217620000],
        {"segment_ownership": True, "pulse_family": False},
        arm_dir,
        c4_continuation_runner=existing_c4_runner,
    )
    assert len(calls) == 1
    assert np.array_equal(calls[0][0], [217620000])
    assert calls[0][1] == {"segment_ownership": True, "pulse_family": False}

    times_path, labels_path = materialize_times_labels(matching, tmp_path / "adapted")
    assert np.array_equal(np.load(times_path, allow_pickle=False), event_samples)
    assert np.array_equal(np.load(labels_path, allow_pickle=False), event_labels)
    assert np.count_nonzero(event_samples == switch) == 2
    assert set(event_labels[event_samples == switch]) == {100, 106}
