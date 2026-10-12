"""Independent NUL and service-unit boundary controls for frozen v3."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PACKET = Path(__file__).resolve().parents[1] / "joint_model_n0_n1_contract_validator_h1_20261012_v3"
REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = _load("frozen_n0_n1_validator_v3_boundary", PACKET / "IMPLEMENTATION.py")
fixtures = _load("frozen_n0_n1_validator_v3_fixture_boundary", PACKET / "TESTS.py")


def valid():
    return fixtures.valid_contract()


def accepts(document):
    return validator.validate_n0_n1_contract(document)


def test_service_name_exact_255_byte_boundary_passes():
    unit = "a" * 247 + ".service"
    assert len(unit.encode("ascii")) == 255
    document = valid(); document["resources"]["service_unit"] = unit
    assert accepts(document)["status"] == "STRUCTURALLY_VALID_METADATA_ONLY"


def test_service_name_256_byte_boundary_fails():
    unit = "a" * 248 + ".service"
    assert len(unit.encode("ascii")) == 256
    document = valid(); document["resources"]["service_unit"] = unit
    with pytest.raises(ValueError, match="service unit"):
        accepts(document)


@pytest.mark.parametrize(
    "unit",
    ["bad unit.service", ".service", "bad\x00unit.service", "café.service", "job.timer", "dir/job.service"],
)
def test_non_simple_service_names_fail_closed(unit):
    document = valid(); document["resources"]["service_unit"] = unit
    with pytest.raises(ValueError, match="service unit"):
        accepts(document)


@pytest.mark.parametrize("suffix", ["\x00tail", "middle\x00tail"])
@pytest.mark.parametrize("target", ["recording", "motion", "N0", "N1"])
def test_nul_is_rejected_in_every_path_binding(target, suffix):
    document = valid()
    if target == "recording": document["input"]["recording"]["path"] += suffix
    elif target == "motion": document["input"]["motion_field"]["path"] += suffix
    else: document["outputs"][target] += suffix
    with pytest.raises(ValueError, match="NUL"):
        accepts(document)


def test_non_nul_canonical_sibling_outputs_remain_valid_metadata_only():
    document = valid()
    result = accepts(document)
    assert result["does_not_verify_referenced_bytes"] is True
    assert result["does_not_release_execution"] is True

