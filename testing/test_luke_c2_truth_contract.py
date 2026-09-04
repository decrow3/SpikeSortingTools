"""Regression tests for the paired-arm truth contract (C2 v4 scorer threading)."""

import numpy as np
import pytest

from testing.ladder_score import (
    TRUTH_CONTRACT_SCHEMA,
    TruthContractError,
    assert_paired_truth,
    build_truth_contract,
    ground_truth_scores,
    truth_digest,
    validate_truth_contract,
)

FS = 30_000.0
CHANNEL_IDS = np.arange(112)
GEOMETRY = np.column_stack([np.tile([0.0, 32.0, 16.0, 48.0], 28),
                            np.repeat(np.arange(56) * 20.0, 2)])


def admission_record(n_total: int, n_admitted: int, by_level: dict) -> dict:
    return {
        "schema": "luke-c2-staircase-control-v1",
        "rule": "same commensurate displacement across the template window",
        "guard_bins": 1,
        "n_total": n_total,
        "n_admitted": n_admitted,
        "counts_by_level_um": by_level,
    }


def contract_for(truth, **overrides) -> dict:
    n = int(sum(np.asarray(v).size for v in truth.values()))
    kwargs = dict(
        injected=truth,                      # the correct order, by default
        admission=admission_record(n + 3, n, {"0.0": n // 2, "40.0": n - n // 2}),
        channel_ids=CHANNEL_IDS,
        geometry=GEOMETRY,
    )
    kwargs.update(overrides)
    return build_truth_contract(truth, **kwargs)


# --------------------------------------------------------------------------- #
# only the admitted events form the denominator
# --------------------------------------------------------------------------- #
def test_only_admitted_events_enter_recall_and_miss_counts():
    """The 3 excluded events must not become misses, and must not be expected."""
    full = np.arange(10, dtype=np.int64) * 5_000 + 1_000
    admitted = np.delete(full, [3, 4, 5])          # 7 of 10, as if near a step
    # the sort found every event, admitted or not
    sort = {"st": full.copy(), "cl": np.zeros(full.size, dtype=np.int64),
            "label": {0: "good"}, "good": {0}}

    scored = ground_truth_scores(sort, {"inj0": admitted}, FS, duration_s=2.0)
    unit = scored["units"][0]
    assert unit["n_truth"] == 7          # denominator is the admitted train
    assert unit["tp"] == 7
    assert unit["fn"] == 0               # excluded events are not misses
    assert unit["fp"] == 3               # they are unmatched sorter output

    # scoring the unfiltered train instead would change the denominator
    unfiltered = ground_truth_scores(sort, {"inj0": full}, FS, duration_s=2.0)
    assert unfiltered["units"][0]["n_truth"] == 10


def test_contract_records_the_exact_admitted_array_and_its_provenance():
    admitted = np.arange(687, dtype=np.int64) * 5_000
    contract = contract_for({"inj0": admitted})
    assert contract["schema"] == TRUTH_CONTRACT_SCHEMA
    assert contract["n_expected"] == 687
    assert contract["units"]["inj0"]["n"] == 687
    assert contract["truth_sha256"] == truth_digest({"inj0": admitted})
    assert contract["filtered_before_injection"] is True
    assert contract["admission"]["schema"] == "luke-c2-staircase-control-v1"
    assert sum(contract["admission"]["counts_by_level_um"].values()) == 687
    assert contract["spatial"]["n_channels"] == 112


# --------------------------------------------------------------------------- #
# every paired arm shares one denominator
# --------------------------------------------------------------------------- #
def test_all_three_arms_share_one_denominator():
    admitted = np.arange(687, dtype=np.int64) * 5_000
    contract = contract_for({"inj0": admitted})
    shared = assert_paired_truth(
        [contract, contract, contract],
        labels=["static", "staircase", "staircase_corrected"],
    )
    assert shared["n_expected"] == 687
    assert shared["identical_denominator"] is True
    assert shared["arms"] == ["static", "staircase", "staircase_corrected"]


# --------------------------------------------------------------------------- #
# fail closed
# --------------------------------------------------------------------------- #
def test_a_truth_that_is_not_the_contracted_train_fails_closed():
    admitted = np.arange(20, dtype=np.int64) * 5_000
    contract = contract_for({"inj0": admitted})
    reconstructed = np.arange(23, dtype=np.int64) * 5_000  # the unfiltered train
    with pytest.raises(TruthContractError, match="does not match its contract"):
        validate_truth_contract({"inj0": reconstructed}, contract)


def test_one_shifted_sample_fails_closed():
    admitted = np.arange(20, dtype=np.int64) * 5_000
    contract = contract_for({"inj0": admitted})
    nudged = admitted.copy()
    nudged[7] += 1
    with pytest.raises(TruthContractError):
        validate_truth_contract({"inj0": nudged}, contract)


def test_a_spatial_support_mismatch_fails_closed():
    admitted = np.arange(20, dtype=np.int64) * 5_000
    truth = {"inj0": admitted}
    wide = contract_for(truth)
    cropped = contract_for(truth, channel_ids=CHANNEL_IDS[:100], geometry=GEOMETRY[:100])
    with pytest.raises(TruthContractError, match="channel ids differs"):
        assert_paired_truth([wide, cropped], labels=["static", "staircase"])


def test_a_geometry_mismatch_fails_closed():
    admitted = np.arange(20, dtype=np.int64) * 5_000
    truth = {"inj0": admitted}
    moved = GEOMETRY.copy()
    moved[:, 1] += 20.0
    with pytest.raises(TruthContractError, match="geometry differs"):
        assert_paired_truth(
            [contract_for(truth), contract_for(truth, geometry=moved)],
            labels=["static", "staircase"],
        )


def test_differing_denominators_fail_closed():
    a = contract_for({"inj0": np.arange(20, dtype=np.int64) * 5_000})
    b = contract_for({"inj0": np.arange(19, dtype=np.int64) * 5_000})
    with pytest.raises(TruthContractError, match="truth differs"):
        assert_paired_truth([a, b], labels=["static", "staircase"])


def test_injecting_the_unfiltered_train_is_refused():
    """The exact bug that shipped: inject 708, score 687, certify as correct.

    The attestation is derived from the array actually injected, so an
    inject-then-filter runner cannot certify itself.
    """
    full = np.arange(708, dtype=np.int64) * 5_000
    admitted = np.delete(full, np.arange(21))     # the 21 boundary straddlers
    with pytest.raises(TruthContractError, match="injected 708 events, scoring 687"):
        contract_for({"inj0": admitted}, injected={"inj0": full})


def test_the_attestation_is_derived_not_declared():
    admitted = np.arange(20, dtype=np.int64) * 5_000
    contract = contract_for({"inj0": admitted})
    assert contract["filtered_before_injection"] is True
    assert contract["injected_sha256"] == contract["truth_sha256"]
    # there is no boolean a caller can set to claim this
    with pytest.raises(TypeError):
        build_truth_contract(
            {"inj0": admitted}, injected={"inj0": admitted},
            admission=admission_record(23, 20, {"0.0": 20}),
            channel_ids=CHANNEL_IDS, geometry=GEOMETRY,
            filtered_before_injection=True,
        )


def test_one_extra_injected_event_is_refused():
    admitted = np.arange(20, dtype=np.int64) * 5_000
    extra = np.append(admitted, admitted[-1] + 5_000)
    with pytest.raises(TruthContractError, match="not the admitted train"):
        contract_for({"inj0": admitted}, injected={"inj0": extra})


def test_a_miscounted_admission_record_is_refused():
    admitted = np.arange(20, dtype=np.int64) * 5_000
    with pytest.raises(TruthContractError, match="admission claims"):
        build_truth_contract(
            {"inj0": admitted},
            injected={"inj0": admitted},
            admission=admission_record(30, 25, {"0.0": 25}),  # claims 25, holds 20
            channel_ids=CHANNEL_IDS, geometry=GEOMETRY,
        )


def test_an_incomplete_admission_record_is_refused():
    with pytest.raises(TruthContractError, match="missing"):
        build_truth_contract(
            {"inj0": np.arange(5, dtype=np.int64)},
            injected={"inj0": np.arange(5, dtype=np.int64)},
            admission={"schema": "x", "rule": "y", "n_admitted": 5},  # no n_total
            channel_ids=CHANNEL_IDS, geometry=GEOMETRY,
        )


def test_a_single_arm_is_not_a_paired_comparison():
    contract = contract_for({"inj0": np.arange(5, dtype=np.int64)})
    with pytest.raises(TruthContractError, match="at least two arms"):
        assert_paired_truth([contract], labels=["static"])
