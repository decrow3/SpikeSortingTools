import pytest

from testing.ladder_sorter import RESCUE
from testing.luke_c2_threshold_sweep import SWEEP, check_cell, grid_configs, output_root


def manifest(nblocks=0, tu=12, tl=9):
    return {"summary": {"effective_nblocks": nblocks, "applied_do_CAR": True,
                        "applied_Th_universal": tu, "applied_Th_learned": tl},
            "sorter_params": {}}


def test_grid_is_two_dimensional_not_paired_values():
    """Th_universal and Th_learned must be separable, or the sweep cannot say
    which one matters."""
    cells = {(c.overrides["Th_universal"], c.overrides["Th_learned"])
             for c in grid_configs()}
    # the same Th_universal at several Th_learned, and vice versa
    assert len({tl for tu, tl in cells if tu == 10}) > 1
    assert len({tu for tu, tl in cells if tl == 8}) > 1
    assert all(tl <= tu for tu, tl in cells)


def test_the_decisive_missing_cell_is_present():
    """Legacy thresholds with correction OFF has never been run.

    Without it, "legacy recovers what rescue loses" cannot be separated from
    "legacy applies motion correction".
    """
    cells = {(c.overrides["Th_universal"], c.overrides["Th_learned"])
             for c in grid_configs()}
    assert (9, 8) in cells
    assert (12, 9) in cells  # the rescue reference, same arm, same recording
    assert SWEEP["reference_cells"]["legacy_thresholds_no_correction"] == [9, 8]


def test_every_cell_holds_correction_off():
    """Thresholds are the only variable; a cell with correction on is confounded."""
    assert all(c.overrides["do_correction"] is False for c in grid_configs())
    assert "only variable" in SWEEP["correction"]


def test_cells_have_distinct_cache_digests():
    configs = grid_configs()
    assert len({c.digest for c in configs}) == len(configs)
    assert all(c.digest != RESCUE.digest for c in configs)


def test_check_cell_fails_closed_if_correction_crept_back_on():
    with pytest.raises(RuntimeError, match="effective_nblocks"):
        check_cell("th_12_9", manifest(nblocks=1), 12, 9)


def test_check_cell_fails_closed_on_an_unapplied_threshold():
    with pytest.raises(RuntimeError, match="Th_learned"):
        check_cell("th_9_8", manifest(tu=9, tl=9), 9, 8)
    assert check_cell("th_9_8", manifest(tu=9, tl=8), 9, 8)["Th_learned"] == 8


def test_contamination_endpoints_are_scored_not_just_accuracy():
    """D12 raised TP to 682 while adding 603 FP; accuracy alone hides that."""
    for endpoint in ("fp", "refractory_violation_median",
                     "similar_pairs_per_good_unit", "n_output_units_capturing"):
        assert endpoint in SWEEP["endpoints"]
    assert "603 FP" in SWEEP["contamination_guard"]


def test_sweep_is_labelled_development_evidence():
    """D10 motivated the experiment and is in the cohort, so it cannot certify."""
    assert "held-out" in SWEEP["status"]
    assert "D10" in SWEEP["why_not_confirmatory"]


def test_never_writes_under_mnt():
    with pytest.raises(ValueError, match="under /mnt"):
        output_root("/mnt/NPX/somewhere")
