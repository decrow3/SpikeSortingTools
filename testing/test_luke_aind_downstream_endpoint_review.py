import json

import pandas as pd

from testing.luke_aind_downstream_endpoint_review import (
    paired_metric_review,
    validate_score_grain,
    write_markdown_artifacts,
)


def test_validate_score_grain_accepts_complete_frozen_panel():
    rows = []
    for probe in ("imec0", "imec1"):
        for window in ("T1_high_motion", "T2_combined", "T3_combined"):
            for condition in (
                "rescue_ks_car_on",
                "pinned_aind_ks_car_on",
                "pinned_aind_ks_car_off",
            ):
                rows.append(
                    {
                        "probe": probe,
                        "window": window,
                        "condition": condition,
                        "sealed_event_count": 72,
                    }
                )
    audit = validate_score_grain(pd.DataFrame(rows))
    assert audit["score_rows"] == 18
    assert audit["duplicate_grain_rows"] == 0


def test_paired_review_counts_favorable_lower_values():
    rows = []
    for probe in ("imec0", "imec1"):
        for window in ("T1_high_motion", "T2_combined", "T3_combined"):
            for condition, value in (
                ("rescue_ks_car_on", 0.2),
                ("pinned_aind_ks_car_on", 0.1),
                ("pinned_aind_ks_car_off", 0.3),
            ):
                rows.append(
                    {
                        "probe": probe,
                        "window": window,
                        "condition": condition,
                        "median_good_refractory_fraction_1p5ms": value,
                    }
                )
    scores = pd.DataFrame(rows)
    for metric in (
        "sealed_event_recovery",
        "stable_good_fraction_30s",
        "coincidence_excess",
        "similar_pairs_per_100_good_units",
        "residual_pairs_supporting_redundancy",
        "kilosort_good_count",
        "median_good_contamination_pct",
    ):
        scores[metric] = 1.0
    review = paired_metric_review(scores)
    on = review[
        review.challenger.eq("pinned_aind_ks_car_on")
        & review.metric.eq("median_good_refractory_fraction_1p5ms")
    ].iloc[0]
    off = review[
        review.challenger.eq("pinned_aind_ks_car_off")
        & review.metric.eq("median_good_refractory_fraction_1p5ms")
    ].iloc[0]
    assert on.favorable_cells == 6
    assert off.unfavorable_cells == 6


def test_write_markdown_artifacts_mirrors_csv_and_json(tmp_path):
    pd.DataFrame({"metric": ["recovery"], "value": [0.75]}).to_csv(
        tmp_path / "results.csv", index=False
    )
    (tmp_path / "audit.json").write_text(json.dumps({"complete": True}) + "\n")

    written = write_markdown_artifacts(tmp_path)

    assert {path.name for path in written} == {
        "README.md",
        "audit.md",
        "results.md",
    }
    assert "| recovery | 0.75 |" in (tmp_path / "results.md").read_text()
    assert '"complete": true' in (tmp_path / "audit.md").read_text()
    assert "[Results](results.md)" in (tmp_path / "README.md").read_text()
