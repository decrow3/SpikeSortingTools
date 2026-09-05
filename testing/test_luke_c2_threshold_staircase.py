import pytest

from testing.luke_c2_threshold_staircase import CANDIDATES, COMPARISON, output_root


def test_the_three_required_configurations_are_present():
    cells = {(c.overrides["Th_universal"], c.overrides["Th_learned"]) for c in CANDIDATES}
    assert cells == {(12, 9), (8, 8), (9, 9)}


def test_correction_is_off_everywhere_so_thresholds_are_the_only_variable():
    assert all(c.overrides["do_correction"] is False for c in CANDIDATES)
    assert len({c.digest for c in CANDIDATES}) == 3


def test_arms_are_paired_static_and_staircase_on_one_denominator():
    assert COMPARISON["arms"] == ["static", "staircase"]
    assert "687" in COMPARISON["truth"] and "identical" in COMPARISON["truth"]


def test_artifacts_are_retained_for_the_follow_up_analyses():
    """Deleting recordings would make truncation and stage tracing impossible."""
    assert "curation outputs" in COMPARISON["retain"]


def test_selection_excludes_yield_and_fractional_cells():
    assert "truncation" in COMPARISON["selection"]
    assert "ks_good_yield_alone" in COMPARISON["excluded"]
    assert "fractional_threshold_cells" in COMPARISON["excluded"]


def test_never_writes_under_mnt():
    with pytest.raises(ValueError, match="under /mnt"):
        output_root("/mnt/NPX/nope")
