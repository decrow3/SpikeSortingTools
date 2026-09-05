import numpy as np
import pytest

from testing.luke_c2_threshold_stage_trace import CASES, first_divergent_stage, matched


def test_matched_finds_events_within_tolerance_only():
    reference = np.array([1000, 2000, 3000], dtype=np.int64)
    candidate = np.array([1005, 2500], dtype=np.int64)
    assert matched(reference, candidate, tol=10).tolist() == [True, False, False]
    assert matched(reference, np.array([], dtype=np.int64), tol=10).tolist() == [False] * 3


def test_detection_divergence_is_named_as_a_threshold_effect():
    """Events absent from full_st were never detected; no curation can recover them."""
    fail = {"n_truth": 708, "stage_detection_found": 609, "stage_kept_found": 609,
            "stage_final_found": 609, "stage_best_cluster_found": 467}
    ok = {"n_truth": 708, "stage_detection_found": 708, "stage_kept_found": 708,
          "stage_final_found": 708, "stage_best_cluster_found": 704}
    verdict = first_divergent_stage(fail, ok)
    assert verdict["first_divergent_stage"] == "stage_detection_found"
    assert "Th_universal" in verdict["interpretation"]
    assert verdict["delta_events"] == 99


def test_pure_contamination_reports_no_divergent_stage():
    """D14's cliffs lose no spikes; the winning cluster is polluted instead."""
    fail = {"n_truth": 708, "stage_detection_found": 708, "stage_kept_found": 708,
            "stage_final_found": 708, "stage_best_cluster_found": 697}
    ok = {"n_truth": 708, "stage_detection_found": 708, "stage_kept_found": 708,
          "stage_final_found": 708, "stage_best_cluster_found": 701}
    verdict = first_divergent_stage(fail, ok)
    assert verdict["first_divergent_stage"] == "none"
    assert "contamination" in verdict["interpretation"]


def test_the_required_cases_are_covered():
    pairs = {(c["donor"], c["fail"], c["success"]) for c in CASES}
    assert ("D10", "th_12_9", "th_9_9") in pairs
    assert ("D14", "th_9_8", "th_10_8") in pairs
    assert ("D14", "th_12_7", "th_10_8") in pairs
