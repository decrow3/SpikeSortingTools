"""Independent conservative-boundary controls for K7 v2."""

from __future__ import annotations

import numpy as np
import pytest

from PRIOR_ADVERSARIAL_REPLAY_V2 import prior


def test_exact_contract_floors_are_accepted_and_reported():
    got = prior.score([0, 80], prior.residuals(80, 50, -1.0))
    assert got["reason"] == "scored"
    assert got["training_min_separation_um"] == 80
    assert got["held_min_informative_um"] == 80
    assert got["minimum_events_per_state"] == 50


def test_stricter_heldout_and_count_requirements_do_not_tighten_training_rule():
    got = prior.score(
        [0, 80], prior.residuals(160, 60, -1.0),
        informative_abs_state_um=160, minimum_events_per_state=60,
    )
    assert got["reason"] == "scored"
    assert got["training_max_separation_um"] == 80
    assert got["state_event_counts"] == {160: 60}


def test_reverse_extreme_residual_ratio_is_finite_and_probe_directed():
    tiny = np.nextafter(0.0, 1.0)
    huge = np.finfo(np.float64).max
    rows = [{"state_um": -80, "residual_probe": tiny, "residual_tissue": huge}] * 50
    got = prior.score([0, -80], rows)
    assert np.isfinite(got["score"])
    assert got["reported_label"] == "probe_fixed"


@pytest.mark.parametrize(
    "row,match",
    [
        ({"state_um": 0, "residual_tissue": 1.0}, "lacks residual"),
        ({"state_um": 0, "residual_probe": -1.0, "residual_tissue": 1.0}, "strictly positive"),
    ],
)
def test_excluded_states_still_validate_all_residual_fields(row, match):
    with pytest.raises(ValueError, match=match):
        prior.score([0, 80], [row])


def test_valid_excluded_rows_abstain_without_contribution():
    got = prior.score([0, 80], prior.residuals(0, 100, -5.0))
    assert got["reason"] == "held_out_state_support"
    assert got["score"] is None
    assert got["state_event_counts"] == {}

