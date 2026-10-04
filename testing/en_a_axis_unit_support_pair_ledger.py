#!/usr/bin/env python3
"""One frozen A-axis per-unit/state/original-pair support ledger.

This reads saved sorter events and reviewed metadata only.  It preserves the
original adjacent same-unit sequence, applies the reviewed contiguous-state
segment gate, and only then categorizes each pair by endpoint support.  It does
not read voltage, run a sorter/matcher, or access RF/holdout results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from numba import njit

from testing.en_state_support_diagnosis import (
    PAIR_NAMES,
    assign_positions,
    field_cells,
    support_tables,
)


SCHEMA = "en-a-axis-unit-support-pair-ledger-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)


@njit(cache=True)
def accumulate_unit_support_pairs(
    times, units, states, segments, support_status,
    last_time, last_segment, last_support,
    denominator, short, exact,
):
    """Accumulate original adjacency; never filter events before pairing."""
    for i in range(len(times)):
        unit = units[i]
        if last_time[unit] >= 0 and last_segment[unit] == segments[i]:
            delta = times[i] - last_time[unit]
            previous = last_support[unit]
            current = support_status[i]
            if previous >= 2 or current >= 2:
                pair_class = 3
            elif previous == 0 and current == 0:
                pair_class = 0
            elif previous == 1 and current == 1:
                pair_class = 2
            else:
                pair_class = 1
            state = states[i]
            if delta == 0:
                exact[unit, state, pair_class] += 1
            elif delta > 0:
                denominator[unit, state, pair_class] += 1
                if delta < 30:
                    short[unit, state, pair_class] += 1
        last_time[unit] = times[i]
        last_segment[unit] = segments[i]
        last_support[unit] = support_status[i]


def known_answer_fixture() -> dict:
    # Unit 0 creates two mixed short pairs. Unit 1 creates one supported
    # positive pair at lag 30 (denominator only). Unit 2 crosses a segment and
    # must create no pair. All unobserved unit/state/category cells remain zero.
    times = np.array([0, 0, 5, 10, 10, 20, 30], dtype=np.int64)
    units = np.array([0, 1, 2, 0, 2, 0, 1], dtype=np.int64)
    states = np.zeros(len(times), dtype=np.int16)
    segments = np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.int32)
    status = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.uint8)
    shape = (4, 2, len(PAIR_NAMES))
    denominator = np.zeros(shape, dtype=np.int64)
    short = np.zeros(shape, dtype=np.int64)
    exact = np.zeros(shape, dtype=np.int64)
    accumulate_unit_support_pairs(
        times, units, states, segments, status,
        np.full(4, -1, dtype=np.int64), np.full(4, -1, dtype=np.int32),
        np.full(4, 3, dtype=np.uint8), denominator, short, exact,
    )
    observed = {
        "unit0_mixed_denominator": int(denominator[0, 0, 1]),
        "unit0_mixed_short": int(short[0, 0, 1]),
        "unit1_supported_denominator": int(denominator[1, 0, 0]),
        "unit1_supported_short": int(short[1, 0, 0]),
        "unit2_total_denominator": int(denominator[2].sum()),
        "unused_state_total": int(denominator[:, 1].sum() + short[:, 1].sum() + exact[:, 1].sum()),
    }
    expected = {
        "unit0_mixed_denominator": 2,
        "unit0_mixed_short": 2,
        "unit1_supported_denominator": 1,
        "unit1_supported_short": 0,
        "unit2_total_denominator": 0,
        "unused_state_total": 0,
    }
    if observed != expected:
        raise AssertionError((observed, expected))

    metrics = pd.DataFrame({
        "spike_count": [2001, 2000, 3000, 3000],
        "reference_state_events": [199, 0, 299, 300],
        "median_depth_um": [401.0, 500.0, 400.0, 500.0],
    })
    displaced = metrics.reference_state_events < 0.1 * metrics.spike_count
    interior = (
        (metrics.spike_count > 2000)
        & (metrics.median_depth_um > 400)
        & (metrics.median_depth_um < 3600)
    )
    strata = np.select(
        [displaced & interior, (~displaced) & interior],
        ["displaced_dominant_interior", "other_interior"],
        default="remaining_boundary_or_ambiguous",
    ).tolist()
    expected_strata = [
        "displaced_dominant_interior", "remaining_boundary_or_ambiguous",
        "remaining_boundary_or_ambiguous", "other_interior",
    ]
    if strata != expected_strata:
        raise AssertionError((strata, expected_strata))
    return {"pass": True, "observed": observed, "expected": expected,
            "strict_threshold_strata": strata}


def verify_hash(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"hash mismatch: {path}: {observed}")


def analyze_arm(
    arm: str, root: Path, metrics_path: Path, cells: dict, support: dict,
    fs: float, chunk_size: int,
) -> tuple[pd.DataFrame, dict]:
    times = np.load(root / "spike_times.npy", mmap_mode="r")
    units = np.load(root / "spike_clusters.npy", mmap_mode="r")
    positions = np.load(root / "spike_positions.npy", mmap_mode="r")
    metrics = pd.read_csv(metrics_path).sort_values("unit_id").reset_index(drop=True)
    unit_ids = np.unique(units)
    if not np.array_equal(unit_ids, np.arange(len(unit_ids))):
        raise RuntimeError(f"{arm}: noncontiguous unit ids")
    if not np.array_equal(metrics.unit_id.to_numpy(), unit_ids):
        raise RuntimeError(f"{arm}: metrics unit ids differ from events")
    observed_spikes = np.bincount(np.asarray(units), minlength=len(unit_ids))
    if not np.array_equal(observed_spikes, metrics.spike_count.to_numpy(dtype=np.int64)):
        raise RuntimeError(f"{arm}: metrics spike counts differ from events")
    if len(times) != len(units) or len(times) != len(positions):
        raise RuntimeError(f"{arm}: event arrays are misaligned")
    if np.any(np.diff(np.asarray(times)) < 0):
        raise RuntimeError(f"{arm}: spike times are not globally sorted")

    n_units, n_states = len(unit_ids), len(cells["states"])
    shape = (n_units, n_states, len(PAIR_NAMES))
    denominator = np.zeros(shape, dtype=np.int64)
    short = np.zeros(shape, dtype=np.int64)
    exact = np.zeros(shape, dtype=np.int64)
    state_events = np.zeros((n_units, n_states), dtype=np.int64)
    last_time = np.full(n_units, -1, dtype=np.int64)
    last_segment = np.full(n_units, -1, dtype=np.int32)
    last_support = np.full(n_units, 3, dtype=np.uint8)
    edges = cells["edges"]
    for start in range(0, len(times), chunk_size):
        stop = min(len(times), start + chunk_size)
        t = np.asarray(times[start:stop], dtype=np.int64)
        u = np.asarray(units[start:stop], dtype=np.int64)
        y = np.asarray(positions[start:stop, 1], dtype=np.float64)
        seconds = t / fs
        knot = np.searchsorted(edges, seconds, side="right") - 1
        knot = np.clip(knot, 0, len(cells["knot_state"]) - 1)
        state = cells["knot_state"][knot]
        segment = cells["knot_segment"][knot]
        status, _, _ = assign_positions(y, state, support)
        state_events += np.bincount(
            u * n_states + state, minlength=n_units * n_states
        ).reshape(n_units, n_states)
        accumulate_unit_support_pairs(
            t, u, state, segment, status, last_time, last_segment, last_support,
            denominator, short, exact,
        )

    q0 = int(np.flatnonzero(cells["states"] == 0)[0])
    if not np.array_equal(state_events[:, q0], metrics.reference_state_events.to_numpy(dtype=np.int64)):
        raise RuntimeError(f"{arm}: reference_state_events differ from regenerated q0 counts")
    displaced = metrics.reference_state_events < 0.1 * metrics.spike_count
    interior = (
        (metrics.spike_count > 2000)
        & (metrics.median_depth_um > 400)
        & (metrics.median_depth_um < 3600)
    )
    strata = np.select(
        [displaced & interior, (~displaced) & interior],
        ["displaced_dominant_interior", "other_interior"],
        default="remaining_boundary_or_ambiguous",
    )
    rows = []
    for unit in range(n_units):
        for si, state_um in enumerate(cells["states"]):
            for pair_class, pair_name in enumerate(PAIR_NAMES):
                den = int(denominator[unit, si, pair_class])
                num = int(short[unit, si, pair_class])
                rows.append({
                    "arm": arm, "unit_id": unit, "state_um": float(state_um),
                    "support_pair": pair_name,
                    "positive_adjacent_intervals": den, "short_1_29": num,
                    "zero_lag": int(exact[unit, si, pair_class]),
                    "short_fraction": num / den if den else np.nan,
                    "spike_count": int(metrics.spike_count.iloc[unit]),
                    "reference_state_events": int(metrics.reference_state_events.iloc[unit]),
                    "median_depth_um": float(metrics.median_depth_um.iloc[unit]),
                    "displaced_dominant": bool(displaced.iloc[unit]),
                    "interior": bool(interior.iloc[unit]),
                    "stratum": str(strata[unit]),
                })
    return pd.DataFrame(rows), {
        "events": int(len(times)), "units": n_units,
        "state_events": state_events, "denominator": denominator,
        "short": short, "exact": exact,
    }


def closure(ledger: pd.DataFrame, published_path: Path) -> list[dict]:
    published = pd.read_csv(published_path)
    published = published[published.field.eq("A")]
    expected = published.groupby(["arm", "state_um", "support_pair"], as_index=False)[
        ["positive_adjacent_intervals", "short_1_29", "exact_duplicates"]
    ].sum()
    observed = ledger.groupby(["arm", "state_um", "support_pair"], as_index=False)[
        ["positive_adjacent_intervals", "short_1_29", "zero_lag"]
    ].sum().rename(columns={"zero_lag": "exact_duplicates"})
    keys = pd.MultiIndex.from_product([
        sorted(ledger.arm.unique()), sorted(ledger.state_um.unique()), PAIR_NAMES
    ], names=["arm", "state_um", "support_pair"])
    expected = expected.set_index(["arm", "state_um", "support_pair"]).reindex(keys, fill_value=0)
    observed = observed.set_index(["arm", "state_um", "support_pair"]).reindex(keys, fill_value=0)
    rows = []
    for key in keys:
        e = expected.loc[key]
        o = observed.loc[key]
        match = all(int(o[name]) == int(e[name]) for name in expected.columns)
        rows.append({"arm": key[0], "state_um": float(key[1]), "support_pair": key[2],
                     "exact_match": bool(match)})
    if not all(row["exact_match"] for row in rows):
        raise RuntimeError("per-unit ledger does not close to reviewed support totals")
    return rows


def summarize(ledger: pd.DataFrame, rule: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    nonzero_supported = ledger[
        ledger.state_um.ne(0) & ledger.support_pair.eq("both_supported")
    ]
    strata = nonzero_supported.groupby(["arm", "stratum"], as_index=False)[
        ["positive_adjacent_intervals", "short_1_29", "zero_lag"]
    ].sum()
    unit_meta = ledger[["arm", "unit_id", "stratum", "spike_count", "reference_state_events"]].drop_duplicates()
    meta = unit_meta.assign(
        nonreference_state_events=lambda x: x.spike_count - x.reference_state_events
    ).groupby(["arm", "stratum"], as_index=False).agg(
        units=("unit_id", "count"), spike_count=("spike_count", "sum"),
        reference_state_events=("reference_state_events", "sum"),
        nonreference_state_events=("nonreference_state_events", "sum"),
    )
    strata = strata.merge(meta, on=["arm", "stratum"], how="left")
    strata["short_fraction"] = strata.short_1_29 / strata.positive_adjacent_intervals

    wide = strata.pivot(index="stratum", columns="arm", values="short_1_29").fillna(0)
    excess = wide.get("A", 0) - wide.get("REF", 0)
    positive_total = float(np.maximum(excess, 0).sum())
    target = "displaced_dominant_interior"
    target_excess = float(excess.get(target, 0))
    share = target_excess / positive_total if positive_total else 0.0
    concentrated = target_excess > 0 and share >= float(rule["minimum_positive_excess_share"])
    verdict = (
        "primary_biological_validation_target"
        if concentrated else
        "retire_claim_that_displaced_dominant_interior_population_explains_supported_short_pair_increase"
    )
    targets = nonzero_supported[
        (nonzero_supported.arm == "A")
        & (nonzero_supported.stratum == target)
    ].groupby("unit_id", as_index=False)[["positive_adjacent_intervals", "short_1_29"]].sum()
    targets = targets.sort_values(["short_1_29", "positive_adjacent_intervals", "unit_id"], ascending=[False, False, True])
    targets["short_fraction"] = targets.short_1_29 / targets.positive_adjacent_intervals
    targets["rank_by_supported_short"] = np.arange(1, len(targets) + 1)
    return strata, targets, {
        "verdict": verdict, "concentrated": bool(concentrated),
        "target_positive_excess_share": share,
        "target_short_excess_A_minus_REF": int(target_excess),
        "positive_stratum_excess_total": int(positive_total),
        "rule": rule,
        "interpretation_limit": "A and REF flags are arm-local; this is not cross-arm same-neuron causality.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema") != SCHEMA:
        raise RuntimeError("wrong frozen config schema")
    source = Path(__file__).resolve()
    verify_hash(source, config["source_sha256"])
    for item in config["inputs"]:
        verify_hash(Path(item["path"]), item["sha256"])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    fixture = known_answer_fixture()

    fs = float(config["sampling_frequency_hz"])
    duration = int(config["num_samples"]) / fs
    cells = field_cells(config["field"], duration)
    roots = {name: Path(spec["root"]) for name, spec in config["arms"].items()}
    geometry = np.load(roots["A"] / "channel_positions.npy")
    if not np.array_equal(geometry, np.load(roots["REF"] / "channel_positions.npy")):
        raise RuntimeError("A and REF geometry differ")
    support = support_tables(geometry, cells["states"])
    ledgers, arm_results = [], {}
    for arm in ("A", "REF"):
        spec = config["arms"][arm]
        ledger, arm_result = analyze_arm(
            arm, roots[arm], Path(spec["unit_metrics_path"]), cells, support,
            fs, int(config["chunk_size"]),
        )
        ledgers.append(ledger); arm_results[arm] = arm_result
    ledger = pd.concat(ledgers, ignore_index=True)
    closure_rows = closure(ledger, Path(config["published_pair_support_counts_path"]))

    nonzero = ledger.state_um.ne(0)
    observed_totals = ledger[nonzero].groupby(["arm", "support_pair"])[
        ["positive_adjacent_intervals", "short_1_29", "zero_lag"]
    ].sum()
    for arm, categories in config["expected_nonzero_totals"].items():
        for category, expected in categories.items():
            observed = observed_totals.loc[(arm, category)].to_dict()
            if any(int(observed[key]) != int(value) for key, value in expected.items()):
                raise RuntimeError(f"frozen nonzero total differs: {arm}/{category}: {observed}")

    strata, targets, decision = summarize(ledger, config["decision_rule"])
    ledger.to_csv(args.output / "UNIT_STATE_SUPPORT_PAIR_LEDGER.csv", index=False)
    strata.to_csv(args.output / "STRATUM_SUPPORTED_SHORT_COUNTS.csv", index=False)
    targets.to_csv(args.output / "A_DISPLACED_DOMINANT_INTERIOR_TARGET_UNITS.csv", index=False)
    result = {
        "schema": SCHEMA, "status": "complete",
        "source": {"path": str(source), "sha256": sha256(source)},
        "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
        "fixture": fixture, "field_sha256": cells["sha256"],
        "field_axis_correction": config["field_axis_correction"],
        "arms": {arm: {"events": value["events"], "units": value["units"]}
                 for arm, value in arm_results.items()},
        "ledger_rows": len(ledger), "closure_rows": len(closure_rows),
        "closure_all_exact": all(row["exact_match"] for row in closure_rows),
        "nonzero_totals": {
            f"{arm}/{category}": {key: int(value) for key, value in row.items()}
            for (arm, category), row in observed_totals.iterrows()
        },
        "decision": decision,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "raw_voltage_bytes_read": 0, "sorts_launched": 0,
            "matcher_replays": 0, "rf_accesses": 0, "holdout_accesses": 0,
        },
        "limitations": [
            "Saved Kilosort positions are feature-mass-weighted proxies, not observed waveform origins.",
            "A and REF strata are arm-local and do not identify the same biological neurons across arms.",
            "Short intervals are a contamination proxy and do not prove duplicate spikes or impurity.",
        ],
    }
    atomic_json(args.output / "RESULT.json", result)
    products = []
    for path in sorted(args.output.iterdir()):
        if path.name in {"MANIFEST.json", "COMPLETE.json"}:
            continue
        products.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(args.output / "MANIFEST.json", {"schema": SCHEMA + "-manifest", "products": products})
    atomic_json(args.output / "COMPLETE.json", {
        "schema": SCHEMA + "-complete", "status": "complete",
        "manifest_sha256": sha256(args.output / "MANIFEST.json"),
    })


if __name__ == "__main__":
    main()
