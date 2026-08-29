import pandas as pd
import pytest

from testing.luke_score_multichannel_review import score_review, wilson_interval


def test_wilson_interval_contains_observed_fraction():
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high


def test_score_review_computes_primary_endpoint():
    key = pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d", "e", "f"],
            "status": ["matched"] * 3 + ["unmatched"] * 3,
        }
    )
    labels = pd.DataFrame(
        {
            "review_id": key.review_id,
            "review_label": [
                "neural",
                "neural",
                "artifact",
                "neural",
                "artifact",
                "artifact",
            ],
        }
    )
    summary, result = score_review(labels, key)
    unmatched = summary.set_index("status").loc["unmatched"]
    assert unmatched.n_neural == 1
    assert unmatched.neural_fraction_excluding_uncertain == pytest.approx(1 / 3)
    assert result["artifact_hypothesis_screen"]["passes"] is False


def test_score_review_refuses_incomplete_blind_labels():
    key = pd.DataFrame({"review_id": ["a"], "status": ["matched"]})
    labels = pd.DataFrame({"review_id": ["a"], "review_label": [""]})
    with pytest.raises(ValueError, match="unlabeled"):
        score_review(labels, key)
