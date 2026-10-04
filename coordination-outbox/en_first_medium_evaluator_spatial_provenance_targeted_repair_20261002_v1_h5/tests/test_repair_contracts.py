import copy
import hashlib
import json
from pathlib import Path

import pytest


PACKET = Path(__file__).resolve().parents[1]
CONTRACT = PACKET / "contracts" / "physical_waveform_evidence.execution_disabled.v2.json"
SOURCE_DIGEST = "d1e5a4465d387c66a7655db442ecbf8740d2cae0032c300b0456432b0080c614"
SOURCE_HASHES = {
    "io.py": "767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd",
    "postprocessing.py": "d20da9026de10eb83a02466cdb233c85ae4215c15af7d5656c3593c3809a0a55",
}


def validate_enablement(contract):
    if contract["execution_enabled"] or contract["voltage_access_enabled"]:
        text = json.dumps(contract["input_identity"], sort_keys=True)
        if "PROSPECTIVE_BINDING_REQUIRED" in text:
            raise ValueError("prospective bindings cannot enable voltage access")
        if contract["status"] != "FROZEN_REVIEWED_ENABLED":
            raise ValueError("enabled contract lacks reviewed status")


def test_bundled_kilosort_dependency_is_content_addressed():
    root = PACKET / "dependencies" / "kilosort" / SOURCE_DIGEST
    observed = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(SOURCE_HASHES)
    }
    assert observed == SOURCE_HASHES
    aggregate = hashlib.sha256(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert aggregate == SOURCE_DIGEST


def test_waveform_contract_is_disabled_and_resource_arithmetic_is_exact():
    contract = json.loads(CONTRACT.read_text())
    assert contract["execution_enabled"] is False
    assert contract["voltage_access_enabled"] is False
    assert contract["sampling"]["maximum_records"] == 3 * 3 * 4 * 128
    extraction = contract["waveform_extraction"]
    assert extraction["padded_samples"] == 121 + 2 * 512
    logical = 4608 * extraction["padded_samples"] * extraction["raw_channels_per_read"] * 2
    assert logical == contract["resources_and_stop_conditions"]["logical_read_bytes_max"]
    assert "RF fitting" in contract["explicit_exclusions"][0]
    assert "holdout" in contract["explicit_exclusions"][1]
    validate_enablement(contract)


def test_waveform_contract_cannot_be_enabled_with_unbound_inputs():
    contract = json.loads(CONTRACT.read_text())
    altered = copy.deepcopy(contract)
    altered["execution_enabled"] = True
    altered["voltage_access_enabled"] = True
    altered["status"] = "FROZEN_REVIEWED_ENABLED"
    with pytest.raises(ValueError, match="prospective bindings"):
        validate_enablement(altered)


def test_waveform_contract_fixed_crop_and_epoch_partition():
    contract = json.loads(CONTRACT.read_text())
    origin = contract["voltage_identity"]["crop_origin_global_frame"]
    frames = contract["voltage_identity"]["crop_local_frames_half_open"][1]
    expected = [origin + (k * frames) // 3 for k in range(4)]
    assert contract["epochs"]["global_edges_frames"] == expected
    assert expected[-1] == contract["voltage_identity"]["crop_global_frames_half_open"][1]
