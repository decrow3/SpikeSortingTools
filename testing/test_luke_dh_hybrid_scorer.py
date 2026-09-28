import numpy as np
from testing.luke_dh_hybrid_scorer import exact_permitted_matching, score_with_frozen_association_v3


def assoc(donor=1, label=7):
    return [{"donor_id": donor, "primary_label": label}]


def test_counterexample_is_maximum_cardinality_and_minimum_timing():
    matches, policy = exact_permitted_matching(
        np.array([0, 18]), np.array([1, 1]), np.array([12, 30]), np.array([7, 7]), assoc(), 12)
    assert [(t, o) for t, o, _, _ in matches] == [(0, 0), (1, 1)]
    assert len(matches) == 2
    assert policy["tolerance_inclusive"] is True


def test_deterministic_under_row_permutation():
    truth = np.array([0, 18, 50]); output = np.array([12, 30, 49])
    first, _ = exact_permitted_matching(truth, np.ones(3, int), output, np.full(3, 7), assoc(), 12)
    tp = np.array([2, 0, 1]); op = np.array([1, 2, 0])
    second, _ = exact_permitted_matching(truth[tp], np.ones(3, int), output[op], np.full(3, 7), assoc(), 12)
    paired1 = [(truth[t], output[o]) for t, o, _, _ in first]
    paired2 = [(truth[tp][t], output[op][o]) for t, o, _, _ in second]
    assert paired1 == paired2 == [(0, 12), (18, 30), (50, 49)]


def test_background_not_fp_and_noise_explicitly_excluded():
    got = score_with_frozen_association_v3(
        np.array([100]), np.array([1]), np.array([False]),
        np.array([100, 200, 210, 220, 230]), np.array([7, 7, 7, 7, 7]), np.zeros(5, bool),
        assoc(), 12,
        output_provenance=np.array(["injection_candidate", "background_supported", "novel", "unknown", "noise"]),
    )
    row = next(r for r in got["scores"] if r["region"] == "rest")
    assert row["tp"] == 1 and row["injection_fp"] == 0
    assert row["background_supported_unmatched"] == 1
    assert row["novel_unmatched"] == 1 and row["unknown_unmatched"] == 1
    assert row["noise_excluded_for_label_region"] == 1


def test_missing_donor_is_retained():
    got = score_with_frozen_association_v3(
        np.array([100]), np.array([4]), np.array([True]),
        np.array([], int), np.array([], int), np.array([], bool),
        [{"donor_id": 4, "primary_label": None}], 12, output_provenance=np.array([], dtype="U32"))
    ep = next(r for r in got["scores"] if r["region"] == "episode")
    assert ep["tp"] == 0 and ep["fn_without_candidate"] == 1 and ep["primary_label"] is None


def test_background_cannot_be_truth_match_and_ambiguous_is_unresolved():
    got = score_with_frozen_association_v3(
        np.array([100, 200]), np.array([1, 1]), np.array([False, False]),
        np.array([100, 200]), np.array([7, 7]), np.array([False, False]), assoc(), 0,
        output_provenance=np.array(["background_supported", "ambiguous"]),
    )
    row = next(r for r in got["scores"] if r["region"] == "rest")
    assert row["tp"] == 0 and row["matched_candidates"] == 1
    assert row["ambiguous_tp"] == 1 and row["fn_without_candidate"] == 1
    assert got["matches"][0]["match_certainty"] == "unresolved"


def test_fresh_output_defaults_unknown_and_negative_labels_are_noise():
    got = score_with_frozen_association_v3(
        np.array([100]), np.array([1]), np.array([False]),
        np.array([100, 100]), np.array([7, -1]), np.array([False, False]), assoc(), 0)
    row = next(r for r in got["scores"] if r["region"] == "rest")
    assert got["provenance_mode"] == "all_assigned_outputs_unknown"
    assert row["tp"] == 0 and row["unknown_tp"] == 1
    assert got["matching_policy"]["noise_outputs_excluded"] == 1


def test_cross_boundary_match_is_reported_not_discarded():
    got = score_with_frozen_association_v3(
        np.array([100]), np.array([1]), np.array([False]),
        np.array([100]), np.array([7]), np.array([True]), assoc(), 0,
        output_provenance=np.array(["injection_candidate"]))
    assert len(got["cross_boundary_matches"]) == 1
    rest = next(r for r in got["scores"] if r["region"] == "rest")
    episode = next(r for r in got["scores"] if r["region"] == "episode")
    assert rest["cross_boundary_truth_matches"] == 1
    assert episode["cross_boundary_output_consumed"] == 1
    assert rest["global_unmatched_truth_count"] == 0
    assert rest["fn_without_candidate"] == 0
    assert rest["regional_outflow_matches"] == 1
    assert rest["candidate_recall"] == 1.0
    assert rest["same_region_candidate_recall"] == 0.0


def test_two_donors_sharing_label_cannot_reuse_one_output():
    associations = [{"donor_id": 1, "primary_label": 7}, {"donor_id": 2, "primary_label": 7}]
    got = score_with_frozen_association_v3(
        np.array([100, 100]), np.array([1, 2]), np.array([False, False]),
        np.array([100]), np.array([7]), np.array([False]), associations, 0)
    assert len(got["matches"]) == 1
    assert len({row["output_index"] for row in got["matches"]}) == 1
    rest = [row for row in got["scores"] if row["region"] == "rest"]
    assert sum(row["globally_matched_truth_count"] for row in rest) == 1
    assert sum(row["global_unmatched_truth_count"] for row in rest) == 1
