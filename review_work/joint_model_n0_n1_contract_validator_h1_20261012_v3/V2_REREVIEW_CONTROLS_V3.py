"""Independent focused controls for frozen N0/N1 validator v2."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PACKET = Path(__file__).resolve().parent
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


validator = _load("frozen_n0_n1_validator_v3_controls", PACKET / "IMPLEMENTATION.py")
fixtures = _load("frozen_n0_n1_validator_v3_fixture_controls", PACKET / "TESTS.py")


def valid():
    return fixtures.valid_contract()


def accepts(document):
    return validator.validate_n0_n1_contract(document)


def add_second_window(document):
    second = copy.deepcopy(document["input"]["windows"][0])
    second["window_id"] = "w2"
    second["start_frame"], second["stop_frame"] = 4000, 6000
    second["state_intervals"] = [
        {"start_frame": 4000, "stop_frame": 5000, "state_um": 0},
        {"start_frame": 5000, "stop_frame": 6000, "state_um": 80},
    ]
    document["input"]["windows"].append(second)
    return document


def test_checkpoint_maps_are_key_exact_order_independent_and_hash_normalized():
    document = add_second_window(valid())
    document["arms"][0]["checkpoints_by_window"] = {"w2": "D" * 64, "w1": "B" * 64}
    document["arms"][1]["checkpoints_by_window"] = {"w1": "b" * 64, "w2": "d" * 64}
    assert accepts(document)["window_ids"] == ["w1", "w2"]
    for mutation in ("missing", "extra", "mismatch"):
        bad = copy.deepcopy(document)
        if mutation == "missing": del bad["arms"][1]["checkpoints_by_window"]["w2"]
        if mutation == "extra": bad["arms"][1]["checkpoints_by_window"]["w3"] = "e" * 64
        if mutation == "mismatch": bad["arms"][1]["checkpoints_by_window"]["w2"] = "e" * 64
        with pytest.raises(ValueError, match="checkpoint"):
            accepts(bad)


def test_exact_k10_hashes_are_normalized_but_any_content_drift_rejected():
    document = valid()
    document["evaluator"]["implementation_sha256"] = validator.K10_IMPLEMENTATION_SHA256.upper()
    document["evaluator"]["review_complete_sha256"] = validator.K10_REVIEW_COMPLETE_SHA256.upper()
    assert accepts(document)["status"] == "STRUCTURALLY_VALID_METADATA_ONLY"
    for field in ("implementation_sha256", "review_complete_sha256"):
        bad = valid(); bad["evaluator"][field] = "0" * 64
        with pytest.raises(ValueError, match="K10"):
            accepts(bad)


def test_clock_identity_links_and_half_open_semantics_are_exact():
    assert accepts(valid())["does_not_release_execution"] is True
    for target in ("motion", "window"):
        document = valid()
        if target == "motion": document["input"]["motion_field"]["clock"] = "other"
        else: document["input"]["windows"][0]["session_clock_id"] = "other"
        with pytest.raises(ValueError, match="session clock"):
            accepts(document)
    document = valid(); document["input"]["interval_bounds"] = "closed"
    with pytest.raises(ValueError, match="half_open"):
        accepts(document)
    document = valid(); document["input"]["session_clock"]["origin_frame"] = 1.5
    with pytest.raises(ValueError, match="integer"):
        accepts(document)


@pytest.mark.parametrize(
    "n0,n1",
    [
        ("/scratch/run/N0/", "/scratch/run/N1"),
        ("/scratch/run/../run/N0", "/scratch/run/N1"),
        ("/scratch/run", "/scratch/run/N1"),
        ("/data", "/scratch/run/N1"),
        ("/scratch/run/N0", "/data/field.npz"),
    ],
)
def test_canonical_ancestral_and_input_path_guards(n0, n1):
    document = valid(); document["outputs"]["N0"] = n0; document["outputs"]["N1"] = n1
    with pytest.raises(ValueError, match="canonical|output|input"):
        accepts(document)


def test_nul_containing_paths_fail_closed():
    for section, field in (("recording", "path"), ("motion_field", "path")):
        document = valid(); document["input"][section][field] += "\x00suffix"
        with pytest.raises(ValueError, match="path|canonical"):
            accepts(document)
    document = valid(); document["outputs"]["N0"] = "/scratch/run/N0\x00suffix"
    with pytest.raises(ValueError, match="path|canonical"):
        accepts(document)


def test_manager_receipt_and_service_unit_structure_are_exact():
    for field, value in (
        ("manager", "systemd_system"),
        ("launcher_disconnect_verified", 1),
        ("service_unit", "dir/job.service"),
        ("service_unit", "job.timer"),
    ):
        document = valid(); document["resources"][field] = value
        with pytest.raises(ValueError, match="service|manager|disconnection"):
            accepts(document)


@pytest.mark.parametrize("unit", ["bad unit.service", ".service", "bad\x00unit.service"])
def test_service_unit_must_be_a_valid_simple_systemd_unit_name(unit):
    document = valid(); document["resources"]["service_unit"] = unit
    with pytest.raises(ValueError, match="service|unit"):
        accepts(document)


def test_stop_condition_list_is_exact_but_order_independent():
    document = valid(); document["stop_conditions"] = list(reversed(document["stop_conditions"]))
    assert accepts(document)["status"] == "STRUCTURALLY_VALID_METADATA_ONLY"
    variants = [
        tuple(document["stop_conditions"]),
        document["stop_conditions"][:-1],
        document["stop_conditions"] + ["extra"],
        [document["stop_conditions"][0]] * len(document["stop_conditions"]),
    ]
    for conditions in variants:
        bad = valid(); bad["stop_conditions"] = conditions
        with pytest.raises(ValueError, match="stop condition"):
            accepts(bad)
