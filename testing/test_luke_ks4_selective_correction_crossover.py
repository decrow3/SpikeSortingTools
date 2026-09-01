import pandas as pd

from testing.luke_ks4_selective_correction_crossover import (
    crossover_result,
    paired_deltas,
    summarize,
)


def _rows(arm: str, residuals, amplitudes, cosines):
    rows = []
    for displacement, residual, amplitude, cosine in zip(
        (-1.0, 1.0, -20.0, 20.0), residuals, amplitudes, cosines
    ):
        rows.append(
            {
                "arm": arm,
                "template_id": "T01",
                "background": "B01",
                "motion_class": "quiet",
                "generator": "G01",
                "displacement_um": displacement,
                "displacement_sign": "negative" if displacement < 0 else "positive",
                "edge_status": "interior",
                "residual_fraction": residual,
                "amplitude_retention": amplitude,
                "template_cosine": cosine,
            }
        )
    return rows


def test_crossover_requires_all_three_metrics_and_both_signs():
    metrics = pd.DataFrame(
        _rows(
            "moved_no_correction",
            residuals=(0.10, 0.10, 0.30, 0.30),
            amplitudes=(0.90, 0.90, 0.70, 0.70),
            cosines=(0.90, 0.90, 0.70, 0.70),
        )
        + _rows(
            "moved_ks4_native_inverse",
            residuals=(0.11, 0.11, 0.20, 0.20),
            amplitudes=(0.89, 0.89, 0.698, 0.698),
            cosines=(0.89, 0.89, 0.80, 0.80),
        )
    )
    paired = paired_deltas(metrics)
    signed, worst = summarize(paired)
    result = crossover_result(worst)

    assert len(signed) == 4
    assert result["complete_crossover_um"] == 20.0
    assert result["residual_and_cosine_crossover_um"] == 20.0


def test_amplitude_failure_blocks_complete_crossover():
    metrics = pd.DataFrame(
        _rows(
            "moved_no_correction",
            residuals=(0.30, 0.30, 0.30, 0.30),
            amplitudes=(0.70, 0.70, 0.70, 0.70),
            cosines=(0.70, 0.70, 0.70, 0.70),
        )
        + _rows(
            "moved_ks4_native_inverse",
            residuals=(0.20, 0.20, 0.20, 0.20),
            amplitudes=(0.69, 0.69, 0.69, 0.69),
            cosines=(0.80, 0.80, 0.80, 0.80),
        )
    )
    _, worst = summarize(paired_deltas(metrics))
    result = crossover_result(worst)

    assert result["residual_and_cosine_crossover_um"] == 1.0
    assert result["complete_crossover_um"] is None
