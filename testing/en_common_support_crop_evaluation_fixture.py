#!/usr/bin/env python3
"""Known-answer checks for crop evaluation alignment and q0 support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from testing.en_common_support_crop_evaluation import (
    factorial_state_contrasts, fixed_q0_bootstrap_support, load_crop_sort,
    mass_summary, presence_60s, unit_population,
)
from testing.en_common_support_crop_panel import crop_field_cells_from_global_midpoints
from testing.en_tier1_panel import arm_state_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        np.save(root / "spike_times.npy", np.array([100, 150, 199], dtype=np.int64))
        np.save(root / "spike_clusters.npy", np.array([0, 1, 0], dtype=np.int64))
        np.save(root / "spike_positions.npy", np.c_[np.zeros(3), [10, 20, 30]])
        (root / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n0\tgood\n1\tmua\n")
        sort, receipt = load_crop_sort(root, 100, 200)
        if not np.array_equal(sort["times"], [0, 50, 99]):
            raise RuntimeError("half-open crop localization failed")
        if not np.array_equal(sort["clusters"], [0, 1, 0]) or not np.array_equal(sort["depths"], [10, 20, 30]):
            raise RuntimeError("one shared mask was not applied to all aligned arrays")
        if receipt["outside_crop_events_rejected"] != 0:
            raise RuntimeError("inside-crop audit count failed")
        np.save(root / "spike_times.npy", np.array([99, 150, 199], dtype=np.int64))
        try:
            load_crop_sort(root, 100, 200)
        except RuntimeError as error:
            outside_rejected = "outside the frozen crop" in str(error)
        else:
            outside_rejected = False
        if not outside_rejected:
            raise RuntimeError("outside-crop saved event was silently filtered")

    cells = crop_field_cells_from_global_midpoints(
        np.arange(0.5, 4.0, 1.0), np.zeros(4), 4.0, 0.0, 4.0, 2.0)
    # Unit 0 has 100 q0 events across two blocks; unit 1 has 99.
    times = np.r_[np.arange(50), np.arange(200, 250), np.arange(50), np.arange(200, 249)]
    clusters = np.r_[np.zeros(100, int), np.ones(99, int)]
    order = np.argsort(times, kind="stable")
    synthetic = {"times": times[order], "clusters": clusters[order],
                 "depths": np.full(len(times), 1000.0), "labels": {0: "good", 1: "mua"}}
    synthetic_calculated = arm_state_metrics(
        synthetic, cells, sampling_frequency_hz=100.0, num_samples=400,
        multiplicities=np.array([[1, 1], [2, 0], [0, 2]], dtype=np.int16),
        minimum_reference_events=100, processing_depth_um=(200.0, 3640.0),
        scoring_depth_um=(300.0, 3540.0), seed=1)
    support = fixed_q0_bootstrap_support(synthetic_calculated, 100)
    if support["fixed_eligible_unit_count"] != 1 or support["bootstrap_draws"] != 3:
        raise RuntimeError("q0 eligibility support failed")

    # Unit 0's median is inside the common domain even though one event is
    # outside it. Both events must be retained by the unit-level population.
    population_sort = {"clusters": np.array([0, 0, 1, 1]),
                       "depths": np.array([100.0, 200.0, 50.0, 60.0])}
    masses = mass_summary(unit_population(population_sort, (120.0, 3780.0)), 1.0, (120.0, 3780.0))
    common = masses["common_unit_median_depth_domain"]
    if common["units"] != 1 or common["events"] != 2:
        raise RuntimeError("common population used event-depth rather than unit-median selection")

    # A short adjacent interval crosses the 2-s bootstrap boundary but neither
    # a unit nor a state-segment boundary. It remains eligible and is assigned
    # to the second event's block by the frozen panel implementation.
    boundary_sort = {
        "times": np.array([199, 201], dtype=np.int64),
        "clusters": np.array([0, 0], dtype=np.int64),
        "depths": np.array([1000.0, 1000.0]),
        "labels": {0: "good"},
    }
    boundary = arm_state_metrics(
        boundary_sort, cells, sampling_frequency_hz=100.0, num_samples=400,
        multiplicities=np.array([[1, 0], [0, 1]], dtype=np.int16),
        minimum_reference_events=100, processing_depth_um=(200.0, 3640.0),
        scoring_depth_um=(300.0, 3540.0), seed=1,
    )
    state_row = boundary["state_metrics"].iloc[0]
    draw_fraction = boundary["bootstrap"].sort_values("draw")["short_interval_fraction_1_29"].to_numpy()
    if state_row.short_interval_count_1_29 != 1 or not np.isnan(draw_fraction[0]) or draw_fraction[1] != 1.0:
        raise RuntimeError("cross-block interval was dropped or assigned to the wrong block")
    presence, _ = presence_60s(boundary_sort, 100.0, 60000)
    if presence["block_count"] != 10 or presence["block_seconds"] != 60.0:
        raise RuntimeError("crop presence did not use the frozen 60-s blocks")

    arms = ("REF384", "B384", "REF368", "B368")
    point_values = {"REF384": 1.0, "B384": 3.0, "REF368": 4.0, "B368": 10.0}
    points, boots = {}, {}
    for arm in arms:
        points[arm] = {"all_output": pd.DataFrame({
            "state_um": [0.0], "short_interval_fraction_1_29": [point_values[arm]],
            "state_rate_rho_vs_q0": [point_values[arm]]})}
        boots[arm] = {"all_output": pd.DataFrame({
            "draw": [0, 1], "state_um": [0.0, 0.0],
            "short_interval_fraction_1_29": [point_values[arm], point_values[arm]],
            "state_rate_rho_vs_q0": [point_values[arm], np.nan]})}
    contrasts = factorial_state_contrasts(points, boots)
    short = contrasts[contrasts.metric == "short_interval_fraction_1_29"].iloc[0]
    rho = contrasts[contrasts.metric == "state_rate_rho_vs_q0"].iloc[0]
    if short.ref_crop != 3.0 or short.b_crop != 7.0 or short.interaction != 4.0:
        raise RuntimeError("factorial point contrasts failed")
    if short.interaction_finite_draws != 2 or rho.interaction_finite_draws != 1:
        raise RuntimeError("factorial finite-draw accounting failed")
    result = {
        "schema": "en-common-support-crop-evaluation-fixture-v1", "status": "pass",
        "shared_aligned_half_open_mask": True, "single_origin_subtraction": True,
        "outside_crop_saved_event_fails": True,
        "q0_threshold_not_relaxed": support["threshold_q0_events"] == 100,
        "q0_cohort_fixed_from_point_counts": True,
        "bootstrap_draw_support_reported": True, "no_voltage_reads": True,
        "presence_60s_separate_from_300s_bootstrap": True,
        "aligned_factorial_contrasts_and_finite_draw_counts": True,
        "common_domain_uses_unit_median_and_retains_all_unit_events": True,
        "cross_block_interval_preserved_and_assigned_to_second_event": True,
        "no_sorts_launched": True,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
