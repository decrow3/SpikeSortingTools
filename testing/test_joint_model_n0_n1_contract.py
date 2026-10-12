import copy

import pytest

from testing.joint_model_n0_n1_contract import (
    EXACT_DARTSORT_COMMIT,
    SCHEMA,
    validate_n0_n1_contract,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def valid_contract():
    arm_common = {
        "input_binding_sha256": SHA_A,
        "checkpoint_sha256": SHA_B,
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
            "channel_ids_sha256": SHA_A,
            "geometry_sha256": SHA_B,
            "motion_field": {
                "path": "/data/field.npz", "sha256": SHA_A,
                "clock": "AP frame zero", "sign": "corrected = observed - displacement",
                "lattice_um": 40,
            },
            "cohort_sha256": SHA_B,
            "windows": [{
                "window_id": "w1", "start_frame": 1000, "stop_frame": 3000,
                "padding_samples": 100,
                "state_intervals": [
                    {"start_frame": 1000, "stop_frame": 2000, "state_um": 0},
                    {"start_frame": 2000, "stop_frame": 3000, "state_um": -80},
                ],
            }],
        },
        "arms": [
            {"name": "N0", "segment_owned_matching": False, **arm_common},
            {"name": "N1", "segment_owned_matching": True, **arm_common},
        ],
        "evaluator": {
            "implementation_sha256": SHA_A,
            "review_complete_sha256": SHA_B,
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
            "gpu_required": True, "managed_service": "joint-n0-n1.service",
        },
        "outputs": {
            "N0": "/scratch/run/N0", "N1": "/scratch/run/N1",
            "create_only": True, "preserve_failures": True,
        },
        "stop_conditions": ["runtime limit", "memory limit", "first unreconciled invariant"],
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


@pytest.mark.parametrize("field", ["input_binding_sha256", "checkpoint_sha256", "common_config_sha256"])
def test_arms_must_share_input_checkpoint_and_config(field):
    contract = valid_contract()
    contract["arms"][1][field] = "d" * 64
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
    contract = valid_contract()
    contract["arms"][0]["hidden_delta"] = True
    with pytest.raises(ValueError, match="keys"):
        validate_n0_n1_contract(contract)
