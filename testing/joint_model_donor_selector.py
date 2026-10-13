#!/usr/bin/env python3
"""Deterministically select the frozen 12-unit I1/I2 donor panel.

All inputs are compact per-template metadata computed beside the retained bank.
This module never loads or transfers template waveforms.
"""

from __future__ import annotations

from typing import Any

import numpy as np


AMPLITUDE_TIERS = ("low", "medium", "high")


def select_donors(
    unit_ids: Any,
    amplitudes: Any,
    nearest_template_cosines: Any,
    risk_distance_um: Any,
    *,
    isolation_cosine_max: float = 0.9,
    risk_radius_um: float = 200.0,
) -> list[dict[str, int | float | str | bool]]:
    """Return four donors per amplitude tier, including four near risk sites."""
    unit_ids = np.asarray(unit_ids)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    cosines = np.asarray(nearest_template_cosines, dtype=np.float64)
    distances = np.asarray(risk_distance_um, dtype=np.float64)
    n = unit_ids.size
    if unit_ids.ndim != 1 or any(x.shape != (n,) for x in (amplitudes, cosines, distances)):
        raise ValueError("donor metadata arrays must be aligned and one-dimensional")
    if np.issubdtype(unit_ids.dtype, np.bool_) or not np.issubdtype(unit_ids.dtype, np.integer):
        raise ValueError("unit_ids must have integer dtype")
    if np.unique(unit_ids).size != n:
        raise ValueError("unit_ids must be unique")
    if not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
        raise ValueError("amplitudes must be finite and positive")
    if not np.isfinite(cosines).all() or np.any((cosines < -1) | (cosines > 1)):
        raise ValueError("nearest-template cosines must be finite and within [-1, 1]")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("risk distances must be finite and nonnegative")
    if not np.isfinite(isolation_cosine_max) or not -1 <= isolation_cosine_max <= 1:
        raise ValueError("isolation_cosine_max must be within [-1, 1]")
    if not np.isfinite(risk_radius_um) or risk_radius_um < 0:
        raise ValueError("risk_radius_um must be finite and nonnegative")

    eligible = np.flatnonzero(cosines < float(isolation_cosine_max))
    if eligible.size < 12:
        raise ValueError("fewer than 12 isolated templates satisfy the cosine bound")
    amplitude_order = eligible[
        np.lexsort((unit_ids[eligible].astype(np.int64), amplitudes[eligible]))
    ]
    tier_rows = np.array_split(amplitude_order, 3)
    if any(rows.size < 4 for rows in tier_rows):
        raise ValueError("each amplitude third must contain at least four isolated templates")
    tier_of = {
        int(row): tier
        for tier, rows in enumerate(tier_rows)
        for row in rows
    }
    near = [
        int(row) for row in eligible
        if distances[row] <= float(risk_radius_um)
    ]
    if len(near) < 4:
        raise ValueError("fewer than four isolated templates lie within the risk radius")

    def priority(row: int) -> tuple[float, float, int]:
        return (float(cosines[row]), float(distances[row]), int(unit_ids[row]))

    selected: set[int] = set()
    tier_counts = [0, 0, 0]
    for tier, rows in enumerate(tier_rows):
        candidates = sorted((int(row) for row in rows if int(row) in near), key=priority)
        if not candidates:
            raise ValueError(f"amplitude tier {AMPLITUDE_TIERS[tier]} lacks a near-risk donor")
        selected.add(candidates[0])
        tier_counts[tier] += 1
    fourth_near = sorted(
        (row for row in near if row not in selected and tier_counts[tier_of[row]] < 4),
        key=priority,
    )
    if not fourth_near:
        raise ValueError("a fourth distinct near-risk donor is unavailable")
    selected.add(fourth_near[0])
    tier_counts[tier_of[fourth_near[0]]] += 1

    for tier, rows in enumerate(tier_rows):
        for row in sorted(map(int, rows), key=priority):
            if tier_counts[tier] == 4:
                break
            if row not in selected:
                selected.add(row)
                tier_counts[tier] += 1
    if tier_counts != [4, 4, 4] or len(selected) != 12:
        raise RuntimeError("donor panel did not close at four per amplitude tier")

    ledger = []
    for tier, rows in enumerate(tier_rows):
        tier_selected = sorted((int(row) for row in rows if int(row) in selected), key=priority)
        for within_tier_rank, row in enumerate(tier_selected):
            ledger.append({
                "unit_id": int(unit_ids[row]),
                "amplitude_tier": AMPLITUDE_TIERS[tier],
                "within_tier_selection_rank": within_tier_rank,
                "amplitude": float(amplitudes[row]),
                "nearest_template_cosine": float(cosines[row]),
                "risk_distance_um": float(distances[row]),
                "within_risk_radius": bool(distances[row] <= float(risk_radius_um)),
            })
    if sum(bool(row["within_risk_radius"]) for row in ledger) < 4:
        raise RuntimeError("selected donor panel violates the four-near-risk requirement")
    return ledger
