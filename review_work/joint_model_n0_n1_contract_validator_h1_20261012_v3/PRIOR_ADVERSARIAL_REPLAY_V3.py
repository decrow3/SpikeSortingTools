"""Independent metadata-only adversarial review of frozen N0/N1 validator v1."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from collections.abc import Sequence

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


validator = _load("frozen_n0_n1_validator_v1_review", PACKET / "IMPLEMENTATION.py")
fixtures = _load("frozen_n0_n1_validator_tests_v1_review", PACKET / "TESTS.py")


def valid():
    return fixtures.valid_contract()


def accepts(document):
    return validator.validate_n0_n1_contract(document)


def test_nominal_receipt_is_metadata_only_and_never_releases_execution():
    got = accepts(valid())
    assert got["status"] == "STRUCTURALLY_VALID_METADATA_ONLY"
    assert got["does_not_verify_referenced_bytes"] is True
    assert got["does_not_release_execution"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=True),
        lambda d: d["source"].update(extra=True),
        lambda d: d["input"]["recording"].update(extra=True),
        lambda d: d["input"]["motion_field"].update(extra=True),
        lambda d: d["input"]["windows"][0].update(extra=True),
        lambda d: d["input"]["windows"][0]["state_intervals"][0].update(extra=True),
        lambda d: d["evaluator"].update(extra=True),
        lambda d: d["resources"].update(extra=True),
        lambda d: d["outputs"].update(extra=True),
    ],
)
def test_unknown_mapping_fields_fail_closed_at_every_schema_level(mutation):
    document = valid(); mutation(document)
    with pytest.raises(ValueError, match="keys"):
        accepts(document)


@pytest.mark.parametrize("field", ["input_binding_sha256", "common_config_sha256"])
def test_cross_arm_binding_drift_fails_closed(field):
    document = valid(); document["arms"][1][field] = "d" * 64
    with pytest.raises(ValueError, match="must share"):
        accepts(document)


def test_cross_arm_checkpoint_binding_drift_fails_closed():
    document = valid(); document["arms"][1]["checkpoints_by_window"]["w1"] = "d" * 64
    with pytest.raises(ValueError, match="must share"):
        accepts(document)


def test_only_segment_ownership_can_differ_and_boolean_is_exact():
    document = valid(); document["arms"][1]["other_deltas"] = ["threshold"]
    with pytest.raises(ValueError, match="unlisted"):
        accepts(document)
    for n0, n1 in ((True, True), (False, False), (0, True), (False, 1)):
        document = valid()
        document["arms"][0]["segment_owned_matching"] = n0
        document["arms"][1]["segment_owned_matching"] = n1
        with pytest.raises(ValueError):
            accepts(document)


@pytest.mark.parametrize("bad", [True, 1.5, 1 + 0j, "1", object(), 2**63])
def test_integer_fields_reject_ambiguous_dtypes_and_overflow(bad):
    for path in ("bytes", "num_samples", "num_channels"):
        document = valid(); document["input"]["recording"][path] = bad
        with pytest.raises(ValueError):
            accepts(document)
    document = valid(); document["input"]["windows"][0]["start_frame"] = bad
    with pytest.raises(ValueError):
        accepts(document)


@pytest.mark.parametrize("bad", [True, 0, -1, np.nan, np.inf, 1 + 0j, "30000"])
def test_sampling_rate_and_resource_bounds_fail_closed(bad):
    document = valid(); document["input"]["recording"]["sampling_frequency_hz"] = bad
    with pytest.raises(ValueError):
        accepts(document)
    document = valid(); document["resources"]["runtime_limit_s"] = bad
    with pytest.raises(ValueError):
        accepts(document)


def test_half_open_state_partition_and_padding_boundaries():
    document = valid()
    document["input"]["windows"][0]["start_frame"] = 100
    document["input"]["windows"][0]["stop_frame"] = 200
    document["input"]["windows"][0]["padding_samples"] = 100
    document["input"]["windows"][0]["state_intervals"] = [
        {"start_frame": 100, "stop_frame": 150, "state_um": 0},
        {"start_frame": 150, "stop_frame": 200, "state_um": 80},
    ]
    assert accepts(document)["window_count"] == 1
    for change in ("gap", "overlap", "uncovered", "padding"):
        bad = copy.deepcopy(document)
        if change == "gap": bad["input"]["windows"][0]["state_intervals"][1]["start_frame"] = 151
        if change == "overlap": bad["input"]["windows"][0]["state_intervals"][1]["start_frame"] = 149
        if change == "uncovered": bad["input"]["windows"][0]["state_intervals"][1]["stop_frame"] = 199
        if change == "padding": bad["input"]["windows"][0]["padding_samples"] = 101
        with pytest.raises(ValueError):
            accepts(bad)


def test_endpoint_margin_direction_and_equivalence_boundaries():
    for margin in (True, -1, np.nan, np.inf, 1 + 0j, "0"):
        document = valid(); document["evaluator"]["endpoints"]["recovery"]["practical_loss_margin"] = margin
        with pytest.raises(ValueError):
            accepts(document)
    document = valid(); document["evaluator"]["endpoints"]["recovery"]["direction"] = "maximize"
    with pytest.raises(ValueError, match="higher or lower"):
        accepts(document)
    document = valid(); document["evaluator"]["inconclusive_is_equivalence"] = 0
    with pytest.raises(ValueError, match="equivalence"):
        accepts(document)


def test_case_insensitive_placeholders_must_fail_closed():
    for placeholder in ("todo", " tBd ", "unknown", "<PATH>", "<SHA256>"):
        document = valid(); document["resources"]["host"] = placeholder
        with pytest.raises(ValueError, match="placeholder"):
            accepts(document)


def test_other_deltas_requires_an_actual_empty_json_list():
    document = valid(); document["arms"][1]["other_deltas"] = np.asarray([], dtype=float)
    with pytest.raises(ValueError, match="deltas|sequence|list"):
        accepts(document)


def test_explicit_session_clock_binding_is_required():
    document = valid(); del document["input"]["session_clock"]
    with pytest.raises(ValueError, match="clock|session"):
        accepts(document)


def test_explicit_half_open_interval_semantics_binding_is_required():
    document = valid(); document["input"]["interval_bounds"] = "inclusive"
    with pytest.raises(ValueError, match="half.open|bounds"):
        accepts(document)


def test_common_event_identity_binding_is_required():
    document = valid(); del document["input"]["common_events_sha256"]
    with pytest.raises(ValueError, match="event"):
        accepts(document)


def test_multiwindow_contract_requires_one_shared_n0_checkpoint_per_window():
    document = valid()
    second = copy.deepcopy(document["input"]["windows"][0])
    second["window_id"] = "w2"
    second["start_frame"], second["stop_frame"] = 4000, 6000
    second["state_intervals"] = [
        {"start_frame": 4000, "stop_frame": 5000, "state_um": 0},
        {"start_frame": 5000, "stop_frame": 6000, "state_um": 80},
    ]
    document["input"]["windows"].append(second)
    with pytest.raises(ValueError, match="checkpoint.*window|window.*checkpoint"):
        accepts(document)


def test_evaluator_must_be_pinned_to_exact_reviewed_k10_v4_artifacts():
    document = valid()
    document["evaluator"]["implementation_sha256"] = "1" * 64
    document["evaluator"]["review_complete_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="K10|evaluator|review"):
        accepts(document)


@pytest.mark.parametrize("service", ["none", "interactive terminal", "nohup", "bash"])
def test_managed_service_declaration_cannot_name_unmanaged_launchers(service):
    document = valid(); document["resources"]["manager"] = service
    with pytest.raises(ValueError, match="managed|service"):
        accepts(document)


def test_stop_conditions_cannot_be_arbitrary_nonempty_text():
    document = valid(); document["stop_conditions"] = ["ignore every bound and continue"]
    with pytest.raises(ValueError, match="stop|condition"):
        accepts(document)


@pytest.mark.parametrize(
    "n0,n1",
    [
        ("/scratch/run/N0", "/scratch/run/./N0"),
        ("/scratch/run", "/scratch/run/N1"),
        ("/data/input.raw", "/scratch/run/N1"),
    ],
)
def test_output_namespaces_must_be_canonical_disjoint_and_input_safe(n0, n1):
    document = valid(); document["outputs"]["N0"] = n0; document["outputs"]["N1"] = n1
    with pytest.raises(ValueError, match="output|distinct|input"):
        accepts(document)
