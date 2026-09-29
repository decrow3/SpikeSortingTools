import numpy as np
import pytest
import hashlib
import json
from pathlib import Path

from testing.em2g_slice_full_sort import slice_arrays


ROOT = Path(__file__).resolve().parents[1]


def test_slice_arrays_uses_half_open_window_and_local_clock():
    times = np.asarray([9, 10, 11, 19, 20], dtype=np.int64)
    labels = np.asarray([0, 1, -1, 2, 3], dtype=np.int32)
    channels = np.asarray([4, 5, 6, 7, 8], dtype=np.int16)
    local, selected_labels, selected_channels = slice_arrays(
        times, labels, channels, 10, 20
    )
    assert local.tolist() == [0, 1, 9]
    assert selected_labels.tolist() == [1, -1, 2]
    assert selected_channels.tolist() == [5, 6, 7]


def test_slice_arrays_rejects_unsorted_events():
    with pytest.raises(ValueError, match="not sorted"):
        slice_arrays(
            np.asarray([2, 1]),
            np.asarray([0, 0]),
            np.asarray([0, 0]),
            0,
            3,
        )


def test_validation_contract_is_frozen_to_launched_run_and_references():
    contract_path = ROOT / "configs/em2g_full_slice_validation.v1.json"
    contract = json.loads(contract_path.read_text())
    assert contract["status"] == "frozen_before_full_session_outcome"
    assert contract["scope"]["rf"] is False
    assert contract["scope"]["outer_holdout"] is False
    assert contract["scope"]["voltage_read"] is False
    run = Path(contract["source_run"]["output"])
    assert hashlib.sha256((run / "config.json").read_bytes()).hexdigest() == contract[
        "source_run"
    ]["resolved_config_sha256"]
    for specification in contract["windows"].values():
        reference = specification["exact_reference"]
        assert hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest() == reference[
            "sha256"
        ]
