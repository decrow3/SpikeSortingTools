#!/usr/bin/env python3
"""Evaluate the frozen EN common-support crop screen from completed sorts.

This never reads voltage or launches a sort. Kilosort crop exports use global
sample indices, so one shared half-open mask is applied to every aligned spike
array before the crop origin is subtracted exactly once.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from testing.en_common_support_crop_panel import crop_field_cells_from_global_midpoints
from testing.en_rounded_field import rounded_rigid_field
from testing.en_tier1_panel import (
    RAW_FILES, arm_state_metrics, chance_aware_coincidence,
    common_block_multiplicities, correspondence, correspondence_summary,
    json_clean, load_labels, sha256,
)


SCHEMA = "en-common-support-crop-evaluation-v1"
DEFAULT_SORT_ROOT = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1")
ARM_PAIRS = (("REF384", "B384"), ("REF368", "B368"))
COMMON_SUPPORT_DEPTH_UM = (120.0, 3780.0)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_crop_sort(root: Path, start: int, stop: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load aligned arrays, apply one common global-time mask, then localize."""
    missing = [name for name in RAW_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing sort files in {root}: {missing}")
    times_global = np.asarray(np.load(root / "spike_times.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    clusters = np.asarray(np.load(root / "spike_clusters.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    positions = np.asarray(np.load(root / "spike_positions.npy", mmap_mode="r"), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"invalid spike_positions.npy in {root}")
    if not (len(times_global) == len(clusters) == len(positions)):
        raise ValueError(f"unaligned saved spike arrays in {root}")
    if len(times_global) and np.any(np.diff(times_global) < 0):
        raise ValueError(f"unsorted spike times in {root}")
    shared = (times_global >= start) & (times_global < stop)
    rejected = int(len(times_global) - np.count_nonzero(shared))
    if rejected:
        raise RuntimeError(f"{root}: {rejected} saved events lie outside the frozen crop")
    times = times_global[shared] - np.int64(start)
    clusters = clusters[shared]
    depths = positions[shared, 1]
    if not (len(times) == len(clusters) == len(depths)):
        raise RuntimeError("shared crop mask did not preserve alignment")
    if len(times) and (times[0] < 0 or times[-1] >= stop - start):
        raise RuntimeError("localized sample lies outside effective crop")
    if not np.isfinite(depths).all():
        raise ValueError("nonfinite saved spike depth")
    receipt = {
        "root": str(root.resolve()), "saved_events": int(len(times_global)),
        "crop_events": int(len(times)), "outside_crop_events_rejected": rejected,
        "all_saved_events_inside_crop": rejected == 0, "shared_aligned_mask_applied": True,
        "global_bounds": [int(times_global[0]), int(times_global[-1])] if len(times_global) else None,
        "files": {name: {"bytes": (root / name).stat().st_size, "sha256": sha256(root / name)}
                  for name in RAW_FILES},
    }
    return {"times": times, "clusters": clusters, "depths": depths,
            "labels": load_labels(root / "cluster_KSLabel.tsv"), "root": str(root.resolve())}, receipt


def unit_population(sort: dict[str, Any], common_depth_um: tuple[float, float]) -> pd.DataFrame:
    """Classify units by median saved y; never filter individual events."""
    clusters = np.asarray(sort["clusters"], dtype=np.int64)
    depths = np.asarray(sort["depths"], dtype=float)
    units, inverse, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    medians = np.asarray([np.median(depths[inverse == index]) for index in range(len(units))], dtype=float)
    low, high = common_depth_um
    domain = np.where(medians < low, "lower_excluded", np.where(medians > high, "upper_excluded", "common"))
    return pd.DataFrame({"unit_id": units, "spike_count": counts, "median_saved_y_um": medians,
                         "population": domain, "common_unit": domain == "common"})


def mass_summary(membership: pd.DataFrame, duration_s: float,
                 common_depth_um: tuple[float, float]) -> dict[str, Any]:
    """Report all, common, lower-excluded, and upper-excluded masses."""
    low, high = common_depth_um
    def group(name: str) -> dict[str, Any]:
        selected = membership[membership.population == name]
        events = int(selected.spike_count.sum())
        return {"events": events, "event_rate_hz": float(events / duration_s), "units": int(len(selected))}
    all_events = int(membership.spike_count.sum())
    return {
        "all_output": {"events": all_events, "event_rate_hz": float(all_events / duration_s),
                       "units": int(len(membership))},
        "common_unit_median_depth_domain": {
            "bounds_um_inclusive": [low, high],
            **group("common"),
            "selection": "unit median saved y; all original events retained",
        },
        "lower_excluded": group("lower_excluded"),
        "upper_excluded": group("upper_excluded"),
    }


def fixed_q0_bootstrap_support(calculated: dict[str, Any], threshold: int) -> dict[str, Any]:
    """Report finite rho draws for the point-frozen q0-eligible cohort."""
    units = calculated["units"]
    eligible = units.loc[units.state_rate_eligible, "unit_id"].astype(int).tolist()
    bootstrap = calculated["bootstrap"]
    finite = bootstrap.assign(finite=np.isfinite(bootstrap.state_rate_rho_vs_q0)).groupby("state_um").finite.sum()
    return {
        "threshold_q0_events": threshold, "cohort_rule": "fixed from point q0 counts; never recomputed per draw",
        "fixed_eligible_unit_count": int(len(eligible)), "fixed_eligible_unit_ids": eligible,
        "bootstrap_draws": int(bootstrap.draw.max() + 1) if len(bootstrap) else 0,
        "finite_rho_draws_by_state_um": {str(float(state)): int(count) for state, count in finite.items()},
    }


def presence_60s(sort: dict[str, Any], fs: float, num_samples: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Frozen crop-screen presence on complete/partial 60-s local blocks."""
    times, clusters = np.asarray(sort["times"], dtype=np.int64), np.asarray(sort["clusters"], dtype=np.int64)
    units, inverse = np.unique(clusters, return_inverse=True)
    duration_s = num_samples / fs
    edges = np.unique(np.r_[np.arange(0.0, duration_s, 60.0), duration_s])
    block = np.searchsorted(edges, times / fs, side="right") - 1
    block = np.clip(block, 0, len(edges) - 2)
    occupied = np.zeros((len(edges) - 1, len(units)), dtype=bool)
    occupied[block, inverse] = True
    ratios = occupied.mean(axis=0)
    table = pd.DataFrame({"unit_id": units, "presence_ratio_60s": ratios})
    summary = {"block_seconds": 60.0, "block_count": int(len(edges) - 1),
               "partial_last_block_seconds": float(np.diff(edges)[-1]),
               "median": float(np.median(ratios)) if len(ratios) else None,
               "p10": float(np.quantile(ratios, 0.1)) if len(ratios) else None,
               "p90": float(np.quantile(ratios, 0.9)) if len(ratios) else None}
    return summary, table


def factorial_state_contrasts(points: dict[str, dict[str, pd.DataFrame]],
                              bootstraps: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Aligned four-arm contrasts using the shared draw IDs and multiplicities."""
    arms = ("REF384", "B384", "REF368", "B368")
    metrics = ("short_interval_fraction_1_29", "state_rate_rho_vs_q0")
    rows = []
    for scope in points["REF384"]:
        states = sorted(set.intersection(*(set(points[arm][scope].state_um) for arm in arms)))
        for state in states:
            for metric in metrics:
                point = {arm: float(points[arm][scope].set_index("state_um").loc[state, metric]) for arm in arms}
                draws = {}
                for arm in arms:
                    selected = bootstraps[arm][scope].loc[
                        lambda frame: frame.state_um == state, ["draw", metric]
                    ].sort_values("draw")
                    draws[arm] = selected[metric].to_numpy(float)
                if len({len(values) for values in draws.values()}) != 1:
                    raise RuntimeError("bootstrap draw grids differ across arms")
                contrasts = {
                    "ref_crop": (point["REF368"] - point["REF384"], draws["REF368"] - draws["REF384"]),
                    "b_crop": (point["B368"] - point["B384"], draws["B368"] - draws["B384"]),
                    "correction_full": (point["B384"] - point["REF384"], draws["B384"] - draws["REF384"]),
                    "correction_common": (point["B368"] - point["REF368"], draws["B368"] - draws["REF368"]),
                }
                contrasts["interaction"] = (
                    contrasts["b_crop"][0] - contrasts["ref_crop"][0],
                    contrasts["b_crop"][1] - contrasts["ref_crop"][1],
                )
                row = {"scope": scope, "state_um": float(state), "metric": metric,
                       **{f"point_{arm}": value for arm, value in point.items()}}
                for name, (value, draw_values) in contrasts.items():
                    finite = draw_values[np.isfinite(draw_values)]
                    low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
                    row.update({name: float(value), f"{name}_ci95_low": float(low),
                                f"{name}_ci95_high": float(high),
                                f"{name}_finite_draws": int(len(finite))})
                rows.append(row)
    return pd.DataFrame(rows)


def audit(sort_root: Path) -> dict[str, Any]:
    arms = {}
    for arm in ("REF384", "B384", "REF368", "B368"):
        root, receipt = sort_root / arm / "sorter_output", sort_root / arm / "RECEIPT.json"
        arms[arm] = {"root": str(root), "missing": [name for name in RAW_FILES if not (root / name).is_file()],
                     "receipt_complete": receipt.is_file() and json.loads(receipt.read_text()).get("status") == "complete"}
        arms[arm]["ready"] = not arms[arm]["missing"] and arms[arm]["receipt_complete"]
    return {"schema": SCHEMA, "sort_root": str(sort_root), "arms": arms,
            "all_ready": all(row["ready"] for row in arms.values())}


def run(screen_path: Path, panel_path: Path, sort_root: Path, output: Path) -> dict[str, Any]:
    screen, panel = json.loads(screen_path.read_text()), json.loads(panel_path.read_text())
    readiness = audit(sort_root)
    if not readiness["all_ready"]:
        raise RuntimeError("all four completed arm receipts and raw sorter outputs are required")
    if "[120,3780]" not in screen["common_support"]["rule"].replace(" ", ""):
        raise RuntimeError("frozen common-support depth bounds differ")
    fs = float(screen["crop"]["sampling_frequency_hz"])
    start, stop = map(int, screen["crop"]["effective_frames_half_open"])
    n_samples, duration_s = stop - start, (stop - start) / fs
    if n_samples != int(screen["crop"]["effective_samples"]):
        raise RuntimeError("effective frame bounds disagree with frozen sample count")
    if not np.isclose(duration_s, screen["crop"]["effective_duration_s"], rtol=0, atol=1e-12):
        raise RuntimeError("effective frame integration disagrees with frozen duration")

    field_spec, field_path = panel["fields"]["B"], Path(panel["fields"]["B"]["path"])
    if sha256(field_path) != field_spec["sha256"] or sha256(field_path) != screen["window_selection"]["field_sha256"]:
        raise RuntimeError("motion field hash differs")
    with np.load(field_path, allow_pickle=False) as saved:
        centers = np.asarray(saved[field_spec["time_key"]], dtype=float)
        displacement = np.asarray(saved[field_spec["displacement_key"]], dtype=float)
    if displacement.ndim == 2:
        displacement = np.median(displacement, axis=1)
    rounded, reference = rounded_rigid_field(displacement)
    block_seconds = float(panel["uncertainty"]["time_block_seconds"])
    cells = crop_field_cells_from_global_midpoints(
        centers, rounded, float(panel["scope"]["duration_s"]), start / fs, stop / fs, block_seconds)
    if not np.isclose(cells.state_exposure_s.sum(), duration_s, rtol=0, atol=1e-8):
        raise RuntimeError("effective field integration does not close crop")
    draws, seed = int(panel["uncertainty"]["bootstrap_draws"]), int(panel["uncertainty"]["seed"])
    multiplicities = common_block_multiplicities(len(cells.block_edges_s) - 1, draws, seed)
    sorts, receipts = {}, {}
    for arm in readiness["arms"]:
        sorts[arm], receipts[arm] = load_crop_sort(sort_root / arm / "sorter_output", start, stop)

    processing = tuple(map(float, panel["metrics"]["edge"]["processing_depth_um"]))
    scoring = tuple(map(float, panel["metrics"]["edge"]["scoring_depth_um"]))
    minimum = int(panel["state_assignment"]["minimum_reference_state_events_per_unit"])
    arm_results, masses, q0_support, coincidence_results = {}, {}, {}, {}
    point_tables, bootstrap_tables = {}, {}
    for index, (arm, sort) in enumerate(sorts.items()):
        membership = unit_population(sort, COMMON_SUPPORT_DEPTH_UM)
        membership.to_csv(output / f"{arm}_POPULATION_MEMBERSHIP.csv", index=False)
        masses[arm] = mass_summary(membership, duration_s, COMMON_SUPPORT_DEPTH_UM)
        common_ids = membership.loc[membership.common_unit, "unit_id"].to_numpy(dtype=np.int64)
        common_mask = np.isin(sort["clusters"], common_ids)
        scoped_sorts = {
            "all_output": sort,
            "common_unit_median_120_3780": {
                **sort,
                "times": sort["times"][common_mask], "clusters": sort["clusters"][common_mask],
                "depths": sort["depths"][common_mask],
            },
        }
        arm_results[arm], q0_support[arm] = {}, {}
        point_tables[arm], bootstrap_tables[arm] = {}, {}
        all_output_calculated = None
        for scope_index, (scope_name, scoped_sort) in enumerate(scoped_sorts.items()):
            calculated = arm_state_metrics(
                scoped_sort, cells, sampling_frequency_hz=fs, num_samples=n_samples,
                multiplicities=multiplicities, minimum_reference_events=minimum,
                processing_depth_um=processing, scoring_depth_um=scoring,
                seed=seed + 10 * index + scope_index)
            if scope_name == "all_output":
                all_output_calculated = calculated
            presence_summary, presence_units = presence_60s(scoped_sort, fs, n_samples)
            calculated["units"].merge(presence_units, on="unit_id", how="left", validate="one_to_one").to_csv(
                output / f"{arm}_{scope_name}_UNITS.csv", index=False)
            arm_results[arm][scope_name] = {
                "summary": calculated["summary"],
                "state_metrics": calculated["state_metrics"].to_dict("records"),
                "screen_presence_60s": presence_summary,
                "chance_aware_coincidence_reference": f"arm_chance_aware_coincidence.{arm}",
            }
            q0_support[arm][scope_name] = fixed_q0_bootstrap_support(calculated, minimum)
            point_tables[arm][scope_name] = calculated["state_metrics"]
            bootstrap_tables[arm][scope_name] = calculated["bootstrap"]
            calculated["bootstrap"].to_csv(output / f"{arm}_{scope_name}_BOOTSTRAP.csv", index=False)
        if all_output_calculated is None:
            raise RuntimeError("all-output panel was not computed")
        coincidence_results[arm] = chance_aware_coincidence(
            sort, all_output_calculated["units"], num_samples=n_samples, sampling_frequency_hz=fs,
            block_edges_s=cells.block_edges_s, multiplicities=multiplicities,
            tolerance_ms=float(panel["metrics"]["chance_aware_coincidence"]["tolerance_ms"]),
            depth_tolerance_um=float(panel["metrics"]["chance_aware_coincidence"]["depth_tolerance_um"]),
            processing_depth_um=processing, seed=seed + 100 + index)

    contrast_table = factorial_state_contrasts(point_tables, bootstrap_tables)
    contrast_table.to_csv(output / "FACTORIAL_STATE_CONTRASTS.csv", index=False)

    pair_results, cm = {}, panel["metrics"]["correspondence"]
    tolerance = int(round(float(cm["tolerance_ms"]) * 1e-3 * fs))
    for pair_index, (reference_arm, candidate) in enumerate(ARM_PAIRS):
        edges = correspondence(sorts[reference_arm], sorts[candidate], tolerance_frames=tolerance,
                               minimum_overlap=float(cm["minimum_overlap"]),
                               primary_retention=float(cm["reciprocal_primary_minimum_retention"]))
        pair_results[f"{reference_arm}_vs_{candidate}"] = correspondence_summary(
            edges, np.unique(sorts[reference_arm]["clusters"]), np.unique(sorts[candidate]["clusters"]),
            draws=draws, seed=seed + 500 + pair_index, reference_seed=seed + 400)
        edges.to_csv(output / f"CORRESPONDENCE_EDGES_{reference_arm}_vs_{candidate}.csv", index=False)

    result = {
        "schema": SCHEMA, "status": "complete",
        "contracts": {"screen": {"path": str(screen_path.resolve()), "sha256": sha256(screen_path)},
                      "panel": {"path": str(panel_path.resolve()), "sha256": sha256(panel_path)},
                      "executed_evaluator": {"path": str(Path(__file__).resolve()),
                                             "sha256": sha256(Path(__file__).resolve())},
                      "crop_helper": {"path": str(Path(__file__).with_name("en_common_support_crop_panel.py").resolve()),
                                      "sha256": sha256(Path(__file__).with_name("en_common_support_crop_panel.py").resolve())},
                      "tier1_helper": {"path": str(Path(__file__).with_name("en_tier1_panel.py").resolve()),
                                       "sha256": sha256(Path(__file__).with_name("en_tier1_panel.py").resolve())}},
        "effective_boundary_check": {"start_frame": start, "stop_frame": stop, "samples": n_samples,
                                     "duration_s": duration_s, "pass": True},
        "field": {"path": str(field_path), "sha256": sha256(field_path), "reference_median_um": reference,
                  "states_um": cells.states_um.tolist(), "exposure_s": cells.state_exposure_s.tolist()},
        "input_receipts": receipts, "mass_summaries": masses,
        "q0_eligibility_and_bootstrap_support": q0_support, "arm_results": arm_results,
        "arm_chance_aware_coincidence": coincidence_results,
        "same_support_correspondence": pair_results,
        "reporting_guards": {"observed_null_excess_separate": True,
            "state_intervals_cross_bootstrap_blocks_but_not_unit_or_state_segment_boundaries": True,
            "common_domain_and_all_output_masses_separate": True,
            "common_state_panels_retain_all_events_from_unit_median_eligible_units": True,
            "presence_uses_60s_blocks": True,
            "state_and_uncertainty_bootstrap_uses_300s_blocks": True,
            "bootstrap_300s_block_count": int(len(cells.block_edges_s) - 1),
            "bootstrap_warning": "Only two 300-s crop blocks; percentile intervals are weak descriptive support.",
            "factorial_contrasts": "All four point values plus REF crop, B crop, correction/full, correction/common, and interaction; uncertainty uses aligned shared draw IDs.",
            "rf_accesses": 0, "raw_voltage_bytes_read": 0, "sorts_launched": 0},
        "interpretation": "Engineering-screen proxies only; short intervals do not prove contamination, rate correlation and time-only correspondence do not prove identity or purity.",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "RESULT.json", result)
    pd.DataFrame([{"arm": arm, "scope": scope, **row}
                  for arm, scopes in arm_results.items() for scope, payload in scopes.items()
                  for row in payload["state_metrics"]]).to_csv(output / "STATE_METRICS.csv", index=False)
    products = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(output.iterdir()) if path.is_file()}
    atomic_json(output / "MANIFEST.json", {"schema": SCHEMA, "products": products})
    atomic_json(output / "COMPLETE.json", {"schema": SCHEMA, "status": "complete",
                                            "manifest_sha256": sha256(output / "MANIFEST.json"),
                                            "written_last_utc": datetime.now(timezone.utc).isoformat()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-config", type=Path, default=Path("configs/en_common_support_crop_screen.v1.json"))
    parser.add_argument("--panel-config", type=Path, default=Path("configs/en_tier1_panel.v1.json"))
    parser.add_argument("--sort-root", type=Path, default=DEFAULT_SORT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/en_common_support_crop_evaluation_v1"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(audit(args.sort_root), indent=2, sort_keys=True))
        return
    partial = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or partial.exists():
        raise FileExistsError(args.output)
    partial.mkdir(parents=True)
    result = run(args.screen_config, args.panel_config, args.sort_root, partial)
    os.replace(partial, args.output)
    print(json.dumps(json_clean(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
