import hashlib
import json
from pathlib import Path

import spikeinterface as si


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_w3_descriptor_is_bound_and_reloadable():
    receipt = json.loads((ROOT / "testing/inputs/em2e_w3/PREPARATION.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["voltage_read"] is False
    assert receipt["sort_launched"] is False
    path = ROOT / receipt["recording"]["path"]
    assert sha256(path) == receipt["recording"]["sha256"]
    recording = si.load(path)
    assert recording.get_num_samples() == 10_199_918
    assert recording.get_num_channels() == 182
    assert str(recording.get_dtype()) == "float32"


def test_w3_transfer_contract_keeps_holdouts_closed():
    contract = json.loads((ROOT / "configs/em2e_w3_transfer_contract.v1.json").read_text())
    assert contract["frozen_before_sort"] is True
    assert contract["scope"]["outer_holdout_accessed"] is False
    assert contract["scope"]["rf_evaluated"] is False
    assert contract["routing"]["idw_or_nearest"] == "do not run"
