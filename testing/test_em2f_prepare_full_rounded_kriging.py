import copy
import hashlib
import json
from pathlib import Path

import spikeinterface as si

from testing.em2f_prepare_full_rounded_kriging import remove_unique_frame_slice


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_remove_unique_frame_slice_removes_only_slice():
    parent = {"class": "Raw", "kwargs": {"value": 1}}
    source = {
        "class": "Outer",
        "kwargs": {
            "recording": {
                "class": "spikeinterface.core.frameslicerecording.FrameSliceRecording",
                "kwargs": {"parent_recording": parent, "start_frame": 10, "end_frame": 20},
            }
        },
    }
    original = copy.deepcopy(source)
    result, removed = remove_unique_frame_slice(source)
    assert source == original
    assert result["kwargs"]["recording"] == parent
    assert removed == {"start_frame": 10, "end_frame": 20}


def test_full_descriptor_is_bound_and_reloadable():
    receipt = json.loads((ROOT / "testing/inputs/em2f_full/PREPARATION.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["voltage_read"] is False
    assert receipt["sort_launched"] is False
    assert receipt["frame_slice_removed"] == {
        "start_frame": 26_999_783,
        "end_frame": 37_199_701,
    }
    path = ROOT / receipt["recording"]["path"]
    assert sha256(path) == receipt["recording"]["sha256"]
    recording = si.load(path)
    assert recording.get_num_samples() == 314_204_094
    assert recording.get_num_channels() == 182
    assert str(recording.get_dtype()) == "float32"


def test_full_lattice_copy_is_hash_bound():
    receipt = json.loads((ROOT / "testing/inputs/em2f_full/PREPARATION.json").read_text())
    lattice = ROOT / receipt["inputs"]["lattice_path"]
    assert sha256(lattice) == receipt["inputs"]["lattice_sha256"]
    assert receipt["rounded_states_um"] == [-280, -240, -200, -160, -120, -80, -40, 0, 40, 80]
