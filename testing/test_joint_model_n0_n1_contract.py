import copy

import numpy as np
import pytest

from testing.joint_model_n0_n1_contract import (
    EXACT_DARTSORT_COMMIT,
    K10_IMPLEMENTATION_SHA256,
    K10_REVIEW_COMPLETE_SHA256,
    SCHEMA,
    validate_n0_n1_contract,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def valid_contract():
    arm_common = {
        "input_binding_sha256": SHA_A,
        "checkpoints_by_window": {"w1": SHA_B},
        "common_config_sha256": "c" * 64,
        "other_deltas": [],
    }
    return {
        "schema": SCHEMA,
        "source": {
            "dartsort_commit": EXACT_DARTSORT_COMMIT,
            "patch_sha256": SHA_A,
            "import_closure_sha256": SHA_B,
            "environment_sha256": "c" * 64,
        },
        "input": {
            "recording": {
                "path": "/data/input.raw", "content_sha256": SHA_A,
                "bytes": 80_000, "dtype": "int16", "num_samples": 10_000,
                "num_channels": 4, "sampling_frequency_hz": 30_000.0,
            },
            "session_clock": {"clock_id": "imec0_ap_session", "origin_frame": 0},
            "interval_bounds": "half_open",
            "channel_ids_sha256": SHA_A,
            "geometry_sha256": SHA_B,
            "motion_field": {
                "path": "/data/field.npz", "sha256": SHA_A,
                "clock": "imec0_ap_session", "sign": "corrected = observed - displacement",
                "lattice_um": 40,
            },
            "cohort_sha256": SHA_B,
            "common_events_sha256": SHA_A,
            "windows": [{
                "window_id": "w1", "session_clock_id": "imec0_ap_session",
                "start_frame": 1000, "stop_frame": 3000,
                "padding_samples": 100,
                "state_intervals": [
                    {"start_frame": 1000, "stop_frame": 2000, "state_um": 0},
                    {"start_frame": 2000, "stop_frame": 3000, "state_um": -80},
                ],
            }],
        },
        "arms": [
            {"name": "N0", "segment_owned_matching": False, **copy.deepcopy(arm_common)},
            {"name": "N1", "segment_owned_matching": True, **copy.deepcopy(arm_common)},
        ],
        "evaluator": {
            "implementation_sha256": K10_IMPLEMENTATION_SHA256,
            "review_complete_sha256": K10_REVIEW_COMPLETE_SHA256,
            "endpoints": {
                "recovery": {"direction": "higher", "practical_loss_margin": 0.05},
                "duplicates": {"direction": "lower", "practical_loss_margin": 0.01},
            },
            "inconclusive_is_equivalence": False,
        },
        "resources": {
            "host": "huklaban5", "runtime_limit_s": 7200,
            "memory_limit_bytes": 80_000_000_000,
            "disk_free_gate_bytes": 100_000_000_000,
            "gpu_required": True, "manager": "systemd_user",
            "service_unit": "joint-n0-n1.service",
            "launcher_disconnect_verified": True,
        },
        "outputs": {
            "N0": "/scratch/run/N0", "N1": "/scratch/run/N1",
            "create_only": True, "preserve_failures": True,
        },
        "stop_conditions": [
            "runtime_limit", "memory_limit", "disk_free_gate",
            "first_unreconciled_invariant", "managed_service_failure",
        ],
    }


def test_valid_contract_is_metadata_only_and_does_not_release_execution():
    result = validate_n0_n1_contract(valid_contract())
    assert result["status"] == "STRUCTURALLY_VALID_METADATA_ONLY"
    assert result["does_not_verify_referenced_bytes"] is True
    assert result["does_not_release_execution"] is True
    assert result["window_ids"] == ["w1"]


@pytest.mark.parametrize("value", ["", "TODO", "<path>", "relative/path"])
def test_unresolved_or_relative_paths_fail_closed(value):
    contract = valid_contract()
    contract["input"]["recording"]["path"] = value
    with pytest.raises(ValueError):
        validate_n0_n1_contract(contract)


@pytest.mark.parametrize("value", ["short", "g" * 64, "<sha256>"])
def test_malformed_hashes_fail_closed(value):
    contract = valid_contract()
    contract["source"]["patch_sha256"] = value
    with pytest.raises(ValueError, match="SHA-256"):
        validate_n0_n1_contract(contract)


def test_exact_dartsort_commit_is_required():
    contract = valid_contract()
    contract["source"]["dartsort_commit"] = "0" * 40
    with pytest.raises(ValueError, match="exact frozen"):
        validate_n0_n1_contract(contract)


@pytest.mark.parametrize("mutation", ["gap", "overlap", "outside", "fractional_state"])
def test_state_bounds_fail_closed(mutation):
    contract = valid_contract()
    intervals = contract["input"]["windows"][0]["state_intervals"]
    if mutation == "gap":
        intervals[1]["start_frame"] = 2100
    elif mutation == "overlap":
        intervals[1]["start_frame"] = 1900
    elif mutation == "outside":
        intervals[1]["stop_frame"] = 3100
    else:
        intervals[1]["state_um"] = -60
    with pytest.raises(ValueError):
        validate_n0_n1_contract(contract)


@pytest.mark.parametrize("field", ["input_binding_sha256", "common_config_sha256"])
def test_arms_must_share_input_checkpoint_and_config(field):
    contract = valid_contract()
    contract["arms"][1][field] = "d" * 64
    with pytest.raises(ValueError, match="must share"):
        validate_n0_n1_contract(contract)


def test_arms_must_share_checkpoints_for_every_window():
    contract = valid_contract()
    contract["arms"][1]["checkpoints_by_window"]["w1"] = "d" * 64
    with pytest.raises(ValueError, match="must share"):
        validate_n0_n1_contract(contract)


def test_only_intended_segment_ownership_delta_is_allowed():
    contract = valid_contract()
    contract["arms"][1]["other_deltas"] = ["threshold"]
    with pytest.raises(ValueError, match="unlisted"):
        validate_n0_n1_contract(contract)
    contract = valid_contract()
    contract["arms"][0]["segment_owned_matching"] = True
    with pytest.raises(ValueError, match="N0 must disable"):
        validate_n0_n1_contract(contract)


def test_endpoint_practical_bounds_and_no_equivalence_claim_are_required():
    contract = valid_contract()
    del contract["evaluator"]["endpoints"]["recovery"]["practical_loss_margin"]
    with pytest.raises(ValueError):
        validate_n0_n1_contract(contract)
    contract = valid_contract()
    contract["evaluator"]["inconclusive_is_equivalence"] = True
    with pytest.raises(ValueError, match="equivalence"):
        validate_n0_n1_contract(contract)


def test_resource_and_output_lifecycle_must_be_bounded():
    contract = valid_contract()
    contract["resources"]["runtime_limit_s"] = 0
    with pytest.raises(ValueError):
        validate_n0_n1_contract(contract)
    contract = valid_contract()
    contract["outputs"]["N1"] = contract["outputs"]["N0"]
    with pytest.raises(ValueError, match="distinct"):
        validate_n0_n1_contract(contract)
    contract = valid_contract()
    contract["outputs"]["preserve_failures"] = False
    with pytest.raises(ValueError, match="failure-preserving"):
        validate_n0_n1_contract(contract)


def test_extra_top_level_or_arm_fields_fail_closed():
    contract = valid_contract()
    contract["outcome"] = "PASS"
    with pytest.raises(ValueError, match="keys"):
        validate_n0_n1_contract(contract)


@pytest.mark.parametrize("placeholder", ["todo", " tBd ", "unknown", "<PATH>", "<SHA256>"])
def test_placeholders_are_casefolded_and_stripped(placeholder):
    contract = valid_contract()
    contract["resources"]["host"] = placeholder
    with pytest.raises(ValueError, match="placeholder"):
        validate_n0_n1_contract(contract)


def test_other_deltas_requires_actual_empty_list():
    contract = valid_contract()
    contract["arms"][1]["other_deltas"] = np.asarray([], dtype=float)
    with pytest.raises(ValueError, match="deltas"):
        validate_n0_n1_contract(contract)


def test_session_clock_half_open_and_common_event_bindings_are_exact():
    contract = valid_contract()
    contract["input"]["windows"][0]["session_clock_id"] = "other"
    with pytest.raises(ValueError, match="session clock"):
        validate_n0_n1_contract(contract)
    contract = valid_contract()
    contract["input"]["motion_field"]["clock"] = "other"
    with pytest.raises(ValueError, match="session clock"):
        validate_n0_n1_contract(contract)
    contract = valid_contract()
    contract["input"]["interval_bounds"] = "inclusive"
    with pytest.raises(ValueError, match="half_open"):
        validate_n0_n1_contract(contract)
    contract = valid_contract()
    del contract["input"]["common_events_sha256"]
    with pytest.raises(ValueError, match="keys"):
        validate_n0_n1_contract(contract)


def test_each_window_has_one_shared_checkpoint():
    contract = valid_contract()
    second = copy.deepcopy(contract["input"]["windows"][0])
    second["window_id"] = "w2"
    second["start_frame"], second["stop_frame"] = 4000, 6000
    second["state_intervals"] = [
        {"start_frame": 4000, "stop_frame": 5000, "state_um": 0},
        {"start_frame": 5000, "stop_frame": 6000, "state_um": 80},
    ]
    contract["input"]["windows"].append(second)
    with pytest.raises(ValueError, match="checkpoint keys"):
        validate_n0_n1_contract(contract)
    for arm in contract["arms"]:
        arm["checkpoints_by_window"]["w2"] = "d" * 64
    assert validate_n0_n1_contract(contract)["window_ids"] == ["w1", "w2"]


def test_evaluator_is_pinned_to_exact_reviewed_k10_v4():
    contract = valid_contract()
    contract["evaluator"]["implementation_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="K10"):
        validate_n0_n1_contract(contract)


@pytest.mark.parametrize("manager", ["none", "nohup", "bash", "interactive terminal"])
def test_only_verified_systemd_user_service_is_managed(manager):
    contract = valid_contract()
    contract["resources"]["manager"] = manager
    with pytest.raises(ValueError, match="systemd_user"):
        validate_n0_n1_contract(contract)


def test_stop_conditions_are_exact_enumerated_ids():
    contract = valid_contract()
    contract["stop_conditions"] = ["ignore every bound and continue"]
    with pytest.raises(ValueError, match="exact required"):
        validate_n0_n1_contract(contract)


def test_nul_containing_paths_fail_closed():
    for section in ("recording", "motion_field"):
        contract = valid_contract()
        contract["input"][section]["path"] += "\0suffix"
        with pytest.raises(ValueError, match="NUL"):
            validate_n0_n1_contract(contract)
    contract = valid_contract()
    contract["outputs"]["N0"] += "\0suffix"
    with pytest.raises(ValueError, match="NUL"):
        validate_n0_n1_contract(contract)


@pytest.mark.parametrize(
    "unit",
    ["bad unit.service", ".service", "bad\0unit.service", f"{'a' * 248}.service"],
)
def test_service_unit_uses_bounded_simple_systemd_name_grammar(unit):
    contract = valid_contract()
    contract["resources"]["service_unit"] = unit
    with pytest.raises(ValueError, match="service unit"):
        validate_n0_n1_contract(contract)


def test_service_unit_accepts_255_byte_simple_name_boundary():
    contract = valid_contract()
    contract["resources"]["service_unit"] = f"{'a' * 247}.service"
    assert validate_n0_n1_contract(contract)["status"] == "STRUCTURALLY_VALID_METADATA_ONLY"


@pytest.mark.parametrize(
    "n0,n1",
    [
        ("/scratch/run/N0", "/scratch/run/./N0"),
        ("/scratch/run", "/scratch/run/N1"),
        ("/data/input.raw", "/scratch/run/N1"),
    ],
)
def test_outputs_are_canonical_disjoint_and_input_safe(n0, n1):
    contract = valid_contract()
    contract["outputs"]["N0"] = n0
    contract["outputs"]["N1"] = n1
    with pytest.raises(ValueError, match="output|canonical"):
        validate_n0_n1_contract(contract)
    contract = valid_contract()
    contract["arms"][0]["hidden_delta"] = True
    with pytest.raises(ValueError, match="keys"):
        validate_n0_n1_contract(contract)
