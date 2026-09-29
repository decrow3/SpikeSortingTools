#!/usr/bin/env python3
"""Frozen EM.2b W2 scorecard for completed, saved DARTsort arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from testing.de_common_outcome_scorecard import assign_domain, intervals, segment_id


DOMAINS = ("negative_excursion", "outside_mask_flat", "catalogue_outside_remainder")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return math.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def block_domain_exposure(
    domain_intervals: list[tuple[float, float, str]], block_edges: np.ndarray
) -> np.ndarray:
    """Return exact seconds in each block/domain intersection."""
    exposure = np.zeros((len(block_edges) - 1, len(DOMAINS)), dtype=np.float64)
    domain_index = {name: index for index, name in enumerate(DOMAINS)}
    for block in range(len(block_edges) - 1):
        left, right = block_edges[block : block + 2]
        for interval_left, interval_right, domain in domain_intervals:
            overlap = max(0.0, min(right, interval_right) - max(left, interval_left))
            exposure[block, domain_index[domain]] += overlap
    return exposure


def arm_inputs(
    times_samples: np.ndarray,
    labels: np.ndarray,
    *,
    start_frame: int,
    sampling_frequency: float,
    field_times: np.ndarray,
    field_deviation: np.ndarray,
    mask: np.ndarray,
    domain_intervals: list[tuple[float, float, str]],
    block_edges: np.ndarray,
    minimum_flat_events: int,
) -> dict[str, object]:
    times_samples = np.asarray(times_samples, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    assigned = labels >= 0
    times_s = (start_frame + times_samples) / sampling_frequency
    domain = assign_domain(times_s, field_times, field_deviation, mask)
    block = np.searchsorted(block_edges, times_s, side="right") - 1
    block = np.clip(block, 0, len(block_edges) - 2)
    units = np.unique(labels[assigned])
    total_exposure = block_domain_exposure(domain_intervals, block_edges).sum(axis=0)
    counts = np.zeros((len(units), len(DOMAINS)), dtype=np.int64)
    block_counts = np.zeros((len(block_edges) - 1, len(units), len(DOMAINS)), dtype=np.int64)
    for unit_index, unit in enumerate(units):
        unit_mask = assigned & (labels == unit)
        for domain_index, name in enumerate(DOMAINS):
            selected = unit_mask & (domain == name)
            counts[unit_index, domain_index] = int(selected.sum())
            block_counts[:, unit_index, domain_index] = np.bincount(
                block[selected], minlength=len(block_edges) - 1
            )
    flat_index = DOMAINS.index("outside_mask_flat")
    eligible = counts[:, flat_index] >= minimum_flat_events
    log_rates = np.log((counts + 0.5) / total_exposure[None, :])
    point_rho = safe_spearman(
        log_rates[eligible, flat_index],
        log_rates[eligible, DOMAINS.index("negative_excursion")],
    )
    isi = {}
    for name in DOMAINS:
        inside = domain == name
        segments = segment_id(times_s, domain_intervals, name)
        selected = np.flatnonzero(inside & assigned)
        order = selected[np.lexsort((times_samples[selected], labels[selected]))]
        same = (
            (labels[order[1:]] == labels[order[:-1]])
            & (segments[order[1:]] == segments[order[:-1]])
            if order.size > 1
            else np.zeros(0, dtype=bool)
        )
        deltas = np.diff(times_samples[order])[same]
        denominator = int(deltas.size)
        isi[name] = {
            "denominator": denominator,
            "lag8": int(np.count_nonzero(deltas == 8)),
            "lag9_29": int(np.count_nonzero((deltas >= 9) & (deltas <= 29))),
            "lag30": int(np.count_nonzero(deltas == 30)),
            "fraction9_29": (
                float(np.mean((deltas >= 9) & (deltas <= 29))) if denominator else math.nan
            ),
        }
    return {
        "units": units,
        "eligible": eligible,
        "counts": counts,
        "block_counts": block_counts,
        "point_rho": point_rho,
        "yield": int(len(units)),
        "isi": isi,
    }


def bootstrap_rhos(
    inputs: dict[str, dict[str, object]],
    block_exposure: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Use one saved block-multiplicity matrix for every arm."""
    rng = np.random.default_rng(seed)
    n_blocks = block_exposure.shape[0]
    sampled = rng.integers(0, n_blocks, size=(draws, n_blocks))
    multiplicities = np.zeros((draws, n_blocks), dtype=np.int16)
    rows = np.repeat(np.arange(draws), n_blocks)
    np.add.at(multiplicities, (rows, sampled.ravel()), 1)
    exposure = multiplicities @ block_exposure
    output = {}
    flat_index = DOMAINS.index("outside_mask_flat")
    negative_index = DOMAINS.index("negative_excursion")
    for arm, values in inputs.items():
        eligible = np.asarray(values["eligible"], dtype=bool)
        counts = np.einsum(
            "qb,buk->quk", multiplicities, np.asarray(values["block_counts"]), optimize=True
        )
        arm_rhos = np.full(draws, np.nan, dtype=np.float64)
        for draw in range(draws):
            if exposure[draw, flat_index] <= 0 or exposure[draw, negative_index] <= 0:
                continue
            flat = np.log((counts[draw, eligible, flat_index] + 0.5) / exposure[draw, flat_index])
            negative = np.log(
                (counts[draw, eligible, negative_index] + 0.5) / exposure[draw, negative_index]
            )
            arm_rhos[draw] = safe_spearman(flat, negative)
        output[arm] = arm_rhos
    return output, multiplicities


def comparison_rows(
    contract: dict[str, object],
    arms: dict[str, dict[str, object]],
    bootstraps: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pairs = [("primary", contract["comparisons"]["primary"])]
    pairs.extend(("secondary", pair) for pair in contract["comparisons"]["secondary"])
    pairs.extend(("bridge", pair) for pair in contract["comparisons"]["bridge"])
    summaries, draws = [], []
    minimum_valid = int(contract["bootstrap"]["minimum_valid_draws"])
    yield_cfg = contract["guardrails"]["yield"]
    isi_cfg = contract["guardrails"]["segment_safe_short_isi"]
    for role, pair in pairs:
        first, second = pair
        if first not in arms or second not in arms:
            continue
        delta = bootstraps[first] - bootstraps[second]
        valid = np.isfinite(delta)
        valid_values = delta[valid]
        low, high = (
            np.percentile(valid_values, [2.5, 97.5]) if valid_values.size else (math.nan, math.nan)
        )
        first_yield, second_yield = arms[first]["yield"], arms[second]["yield"]
        denominator = max(1, int(second_yield))
        directional_yield = (second_yield - first_yield) / denominator <= float(
            yield_cfg["directional_max_first_arm_relative_loss"]
        )
        equivalent_yield = abs(first_yield - second_yield) / denominator <= float(
            yield_cfg["equivalence_max_absolute_relative_difference"]
        )
        directional_isi = True
        equivalent_isi = True
        for domain in DOMAINS:
            a = arms[first]["isi"][domain]["fraction9_29"]
            b = arms[second]["isi"][domain]["fraction9_29"]
            if not np.isfinite(a) or not np.isfinite(b):
                directional_isi = equivalent_isi = False
                break
            directional_isi &= a - b <= float(isi_cfg["directional_max_first_arm_increase"])
            equivalent_isi &= abs(a - b) <= float(isi_cfg["equivalence_max_absolute_difference"])
        point_delta = float(arms[first]["point_rho"] - arms[second]["point_rho"])
        resolved = (
            np.isfinite(point_delta)
            and valid_values.size >= minimum_valid
            and np.isfinite(low)
            and np.isfinite(high)
        )
        if not resolved:
            decision = "unresolved"
        elif (
            point_delta >= 0.05
            and low > 0
            and directional_yield
            and directional_isi
        ):
            decision = "meaningful_first_arm_advantage"
        elif (
            low >= -0.05
            and high <= 0.05
            and equivalent_yield
            and equivalent_isi
        ):
            decision = "practical_equivalence"
        else:
            decision = "mixed_or_inconclusive"
        summaries.append(
            {
                "role": role,
                "first_arm": first,
                "second_arm": second,
                "delta_rho": point_delta,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "valid_draws": int(valid_values.size),
                "directional_yield_pass": bool(directional_yield),
                "equivalence_yield_pass": bool(equivalent_yield),
                "directional_isi_pass": bool(directional_isi),
                "equivalence_isi_pass": bool(equivalent_isi),
                "decision": decision,
            }
        )
        draws.extend(
            {"first_arm": first, "second_arm": second, "draw": index, "delta_rho": float(value)}
            for index, value in enumerate(delta)
            if np.isfinite(value)
        )
    return summaries, draws


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--arms-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    contract = json.loads(args.contract.read_text())
    manifest = json.loads(args.arms_manifest.read_text())
    if manifest["contract_sha256"] != sha256(args.contract):
        raise RuntimeError("scorecard contract hash differs")
    for key in ("field", "mask", "catalogue"):
        receipt = manifest[key]
        expected = contract["inputs"][f"{key}_sha256"]
        if receipt["sha256"] != expected or sha256(Path(receipt["path"])) != expected:
            raise RuntimeError(f"{key} hash differs")
    with np.load(manifest["field"]["path"], allow_pickle=False) as saved:
        field_times = np.asarray(saved["time_s"], dtype=np.float64)
        displacement = np.asarray(saved["displacement_um"], dtype=np.float64)[:, 0]
    baseline = pd.Series(displacement).rolling(480, center=True, min_periods=1).median().to_numpy()
    deviation = displacement - baseline
    mask = pd.read_csv(manifest["mask"]["path"])[["start_s", "end_s"]].to_numpy(float)
    fs = float(contract["scope"]["sampling_frequency_hz"])
    start, end = map(int, contract["scope"]["frames_half_open"])
    left, right = start / fs, end / fs
    domain_intervals = intervals(field_times, deviation, mask, left, right)
    block_s = float(contract["bootstrap"]["block_seconds"])
    block_edges = np.r_[np.arange(left, right, block_s), right]
    block_edges = np.unique(block_edges)
    exposure = block_domain_exposure(domain_intervals, block_edges)
    if not np.isclose(exposure.sum(), right - left, atol=1e-9):
        raise RuntimeError("block-domain exposure does not close W2")
    minimum = int(contract["eligibility"]["minimum_flat_events"])
    arms = {}
    point_rows, isi_rows = [], []
    for arm, receipt in manifest["arms"].items():
        path = Path(receipt["path"])
        if sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"{arm} sorting hash differs")
        with np.load(path, allow_pickle=False) as saved:
            local = np.asarray(saved["times_samples"], dtype=np.int64)
            labels = np.asarray(saved["labels"], dtype=np.int64)
            if not np.isclose(float(saved["sampling_frequency"]), fs, rtol=0, atol=1e-9):
                raise RuntimeError(f"{arm} sampling frequency differs")
        if np.any((local < 0) | (local >= end - start)):
            raise RuntimeError(f"{arm} event outside W2")
        arms[arm] = arm_inputs(
            local,
            labels,
            start_frame=start,
            sampling_frequency=fs,
            field_times=field_times,
            field_deviation=deviation,
            mask=mask,
            domain_intervals=domain_intervals,
            block_edges=block_edges,
            minimum_flat_events=minimum,
        )
        point_rows.append(
            {
                "arm": arm,
                "yield": arms[arm]["yield"],
                "eligible_units": int(np.count_nonzero(arms[arm]["eligible"])),
                "rho_negative_vs_flat": arms[arm]["point_rho"],
            }
        )
        for domain in DOMAINS:
            isi_rows.append({"arm": arm, "domain": domain, **arms[arm]["isi"][domain]})
    bootstraps, multiplicities = bootstrap_rhos(
        arms,
        exposure,
        draws=int(contract["bootstrap"]["draws"]),
        seed=int(contract["bootstrap"]["seed"]),
    )
    summaries, delta_rows = comparison_rows(contract, arms, bootstraps)
    pd.DataFrame(point_rows).to_csv(args.output / "POINT_ESTIMATES.csv", index=False)
    pd.DataFrame(isi_rows).to_csv(args.output / "SEGMENT_SAFE_ISI.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.output / "COMPARISONS.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(args.output / "BOOTSTRAP_DELTAS.csv", index=False)
    pd.DataFrame(exposure, columns=DOMAINS).assign(block=np.arange(len(exposure))).to_csv(
        args.output / "BLOCK_DOMAIN_EXPOSURE.csv", index=False
    )
    result = {
        "schema": "em2b-w2-scorecard-result-v1",
        "status": "complete",
        "contract_sha256": sha256(args.contract),
        "arms_manifest_sha256": sha256(args.arms_manifest),
        "common_bootstrap_multiplicities_sha256": hashlib.sha256(multiplicities.tobytes()).hexdigest(),
        "point_estimates": point_rows,
        "comparisons": summaries,
        "interpretation": "Arm-local rate-rank continuity proxy; no cross-arm identity or purity claim.",
    }
    atomic_json(args.output / "RESULT.json", result)
    products = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(args.output.iterdir())
        if path.name not in {"MANIFEST.json", "COMPLETE.json"}
    ]
    atomic_json(args.output / "MANIFEST.json", {"products": products})
    atomic_json(
        args.output / "COMPLETE.json",
        {"status": "complete", "manifest_sha256": sha256(args.output / "MANIFEST.json")},
    )


if __name__ == "__main__":
    main()
