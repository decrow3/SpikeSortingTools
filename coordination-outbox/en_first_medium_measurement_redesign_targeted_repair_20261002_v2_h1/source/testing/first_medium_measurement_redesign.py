"""Outcome-blind first-medium measurement redesign.

This module defines two deliberately separate measurements:

* fitted amplitude completeness remains a conditional supported-window
  diagnostic; missing or poor-fit support is disclosed, never imputed;
* reference-event retention is an equal-REF-unit estimand built from exclusive
  event matches to one reciprocal primary partner.  It is continuity evidence,
  not biological identity or novel-event recovery.

No loader, recording access, sorting, training, detection, RF, or holdout path
is provided here.  Callers must supply already loaded arrays and tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from testing.sort_comparison import correspondence


SCHEMA = "first-medium-measurement-redesign-v1"
FINITE_FIT_STATUS = "finite_interior"


@dataclass(frozen=True)
class MatchRule:
    tolerance_frames: int
    minimum_overlap: float = 0.1
    primary_retention: float = 0.5

    def __post_init__(self) -> None:
        if self.tolerance_frames < 0:
            raise ValueError("tolerance_frames must be nonnegative")
        if not 0 <= self.minimum_overlap <= 1:
            raise ValueError("minimum_overlap must lie in [0, 1]")
        if not 0 <= self.primary_retention <= 1:
            raise ValueError("primary_retention must lie in [0, 1]")


def _event_arrays(sort: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(sort["st"]).reshape(-1)
    clusters = np.asarray(sort["cl"]).reshape(-1)
    if len(times) != len(clusters):
        raise ValueError(f"{name}: one cluster label is required per event")
    if not np.issubdtype(times.dtype, np.integer) or not np.issubdtype(clusters.dtype, np.integer):
        raise ValueError(f"{name}: event times and cluster labels must be integer arrays")
    if len(times) and np.any(np.diff(times) < 0):
        raise ValueError(f"{name}: events must be globally time ordered")
    return times.astype(np.int64, copy=False), clusters.astype(np.int64, copy=False)


def _cohort_ids(reference_clusters: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    cohort = np.asarray(ids)
    if cohort.ndim != 1 or not np.issubdtype(cohort.dtype, np.integer):
        raise ValueError("reference cohort must be a one-dimensional integer sequence")
    cohort = cohort.astype(np.int64, copy=False)
    if not len(cohort) or len(np.unique(cohort)) != len(cohort):
        raise ValueError("reference cohort must be nonempty and unique")
    missing = sorted(set(cohort.tolist()) - set(np.unique(reference_clusters).tolist()))
    if missing:
        raise ValueError(f"reference cohort contains absent units: {missing}")
    return cohort


def _exclusive_pairs(a: np.ndarray, b: np.ndarray, tolerance: int) -> list[tuple[int, int]]:
    """Greedy chronological one-to-one pairs, identical to the reviewed count rule."""
    i = j = 0
    pairs: list[tuple[int, int]] = []
    while i < len(a) and j < len(b):
        if a[i] < b[j] - tolerance:
            i += 1
        elif b[j] < a[i] - tolerance:
            j += 1
        else:
            pairs.append((i, j))
            i += 1
            j += 1
    return pairs


def validate_half_open_cells(cell_edges_frames: Sequence[int], *, recording_stop_frame: int) -> np.ndarray:
    edges = np.asarray(cell_edges_frames)
    if edges.ndim != 1 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("cell edges must be a one-dimensional integer sequence")
    edges = edges.astype(np.int64, copy=False)
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != int(recording_stop_frame):
        raise ValueError("cell edges must start at zero and end at recording_stop_frame")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("cell edges must be strictly increasing")
    return edges


def half_open_cell_indices(times: Sequence[int], cell_edges_frames: Sequence[int]) -> np.ndarray:
    edges = np.asarray(cell_edges_frames, dtype=np.int64)
    values = np.asarray(times)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("cell-index times must be a one-dimensional integer array")
    values = values.astype(np.int64, copy=False)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("invalid half-open cell edges")
    if len(values) and (values.min() < edges[0] or values.max() >= edges[-1]):
        raise ValueError("event lies outside the half-open cell domain")
    return np.searchsorted(edges, values, side="right") - 1


def reference_event_retention(
    reference_sort: Mapping[str, Any],
    candidate_sort: Mapping[str, Any],
    reference_cohort: Sequence[int],
    match_rule: MatchRule,
    *,
    cell_edges_frames: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute R(X), the equal-unit mean retained fraction on a fixed REF cohort.

    Correspondence is constructed over the complete supplied sorts.  Each REF
    cohort unit receives credit only from its single reciprocal primary partner.
    Exclusive matching prevents candidate events from receiving double credit.
    A cohort unit without such a partner contributes exactly zero.
    """
    rt, rc = _event_arrays(reference_sort, "reference")
    ct, cc = _event_arrays(candidate_sort, "candidate")
    cohort = _cohort_ids(rc, reference_cohort)
    cells = None
    if cell_edges_frames is not None:
        cells = validate_half_open_cells(
            cell_edges_frames, recording_stop_frame=int(cell_edges_frames[-1])
        )
        # Validate both complete sorts before correspondence.  Out-of-domain
        # non-cohort REF events and candidate events can otherwise change
        # reciprocal-primary competition even though they never appear in the
        # per-cell diagnostic rows.
        half_open_cell_indices(rt, cells)
        half_open_cell_indices(ct, cells)
    edges = correspondence(
        {"st": rt, "cl": rc}, {"st": ct, "cl": cc},
        tolerance=match_rule.tolerance_frames,
        minimum_overlap=match_rule.minimum_overlap,
        primary_retention=match_rule.primary_retention,
    )
    primary = edges[edges.primary_match.astype(bool)].copy()
    if primary.baseline_cluster.duplicated().any() or primary.candidate_cluster.duplicated().any():
        raise RuntimeError("reciprocal primary matching is not one-to-one")
    by_reference = primary.set_index("baseline_cluster") if len(primary) else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    credited_candidate_events: set[tuple[int, int]] = set()
    for unit in cohort:
        ref_times = rt[rc == unit]
        denominator = int(len(ref_times))
        if unit not in by_reference.index:
            partner = None
            matched = 0
            match_pairs: list[tuple[int, int]] = []
            candidate_times = np.empty(0, dtype=np.int64)
        else:
            edge = by_reference.loc[unit]
            partner = int(edge.candidate_cluster)
            candidate_times = ct[cc == partner]
            match_pairs = _exclusive_pairs(ref_times, candidate_times, match_rule.tolerance_frames)
            matched = len(match_pairs)
            if matched != int(edge.matched_events):
                raise RuntimeError("exclusive pair reconstruction differs from reviewed matcher")
            for _, candidate_index in match_pairs:
                key = (partner, int(candidate_index))
                if key in credited_candidate_events:
                    raise RuntimeError("candidate event received double credit")
                credited_candidate_events.add(key)
        rows.append({
            "reference_unit": int(unit),
            "candidate_partner": partner,
            "reference_events": denominator,
            "matched_reference_events": int(matched),
            "retention": float(matched / denominator),
            "status": "primary_partner" if partner is not None else "unmatched_zero",
        })
        if cells is not None:
            ref_cell = half_open_cell_indices(ref_times, cells)
            matched_ref_indices = {i for i, _ in match_pairs}
            for cell_index in range(len(cells) - 1):
                indices = np.flatnonzero(ref_cell == cell_index)
                cell_denominator = int(len(indices))
                cell_matched = int(sum(int(index) in matched_ref_indices for index in indices))
                cell_rows.append({
                    "reference_unit": int(unit),
                    "cell_index": cell_index,
                    "start_frame": int(cells[cell_index]),
                    "stop_frame": int(cells[cell_index + 1]),
                    "reference_events": cell_denominator,
                    "matched_reference_events": cell_matched,
                    "retention": float(cell_matched / cell_denominator) if cell_denominator else None,
                    "scope": "state_or_longitudinal_diagnostic_only",
                })
    unit_table = pd.DataFrame(rows)
    return {
        "schema": SCHEMA,
        "estimand": "equal-unit mean exclusive REF-event retention to one reciprocal primary partner",
        "scope": "continuity proxy; not biological identity or novel-event recovery",
        "R": float(unit_table.retention.mean()),
        "reference_units": int(len(unit_table)),
        "matched_units": int((unit_table.status == "primary_partner").sum()),
        "unit_retention": unit_table,
        "cell_diagnostics": pd.DataFrame(cell_rows),
        "correspondence_edges": edges,
        "primary_matches": primary,
        "credited_candidate_events": int(len(credited_candidate_events)),
    }


def retention_difference(
    reference_sort: Mapping[str, Any],
    existing_sort: Mapping[str, Any],
    repaired_sort: Mapping[str, Any],
    reference_cohort: Sequence[int],
    match_rule: MatchRule,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_effect: float = 0.05,
) -> dict[str, Any]:
    """Compute paired unit deltas and both the proposed and replacement rules."""
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    existing = reference_event_retention(reference_sort, existing_sort, reference_cohort, match_rule)
    repaired = reference_event_retention(reference_sort, repaired_sort, reference_cohort, match_rule)
    a = existing["unit_retention"].set_index("reference_unit").retention
    b = repaired["unit_retention"].set_index("reference_unit").retention
    if not a.index.equals(b.index):
        raise RuntimeError("paired REF cohort changed between arms")
    delta_by_unit = (b - a).to_numpy(dtype=float)
    delta = float(delta_by_unit.mean())
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0, len(delta_by_unit), size=(int(bootstrap_replicates), len(delta_by_unit)))
    bootstrap = delta_by_unit[indices].mean(axis=1)
    lower = float(np.quantile(bootstrap, 0.05, method="linear"))
    margin = assess_margin_rules(delta, lower, minimum_effect=minimum_effect)
    return {
        "schema": SCHEMA,
        "R_existing": existing["R"],
        "R_repaired": repaired["R"],
        "DeltaR": delta,
        "unit_deltas": pd.DataFrame({"reference_unit": a.index.astype(int), "delta_retention": delta_by_unit}),
        "bootstrap": {
            "kind": "paired REF-unit composition bootstrap",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "one_sided_lower_95": lower,
            "scope": "sensitivity to fixed-cohort unit composition; not repeated-recording or biological-population uncertainty",
        },
        "proposal_rule": {
            "definition": "DeltaR >= minimum_effect and lower95 > 0",
            "minimum_effect": float(minimum_effect),
            "passes": margin["proposal_passes"],
            "suitability": "rejected_as_minimum_effect_evidence",
            "issue": "a lower bound above zero can still include effects below the stated minimum meaningful effect",
        },
        "frozen_replacement_rule": {
            "definition": "one-sided paired-unit bootstrap lower95 > minimum_effect",
            "minimum_effect": float(minimum_effect),
            "passes": margin["replacement_passes"],
        },
        "estimand_dependencies": [
            "fixed REF cohort",
            "complete supplied sorts for reciprocal-primary competition",
            "exclusive tolerance-bound event matching",
            "minimum-overlap and bilateral primary-retention thresholds",
            "unmatched REF units contribute zero",
        ],
    }


def assess_margin_rules(delta: float, lower95: float, *, minimum_effect: float = 0.05) -> dict[str, Any]:
    """Expose why the proposed and replacement uncertainty rules can disagree."""
    values = np.asarray([delta, lower95, minimum_effect], dtype=float)
    if not np.isfinite(values).all() or minimum_effect < 0:
        raise ValueError("margin inputs must be finite and minimum_effect nonnegative")
    return {
        "proposal_passes": bool(delta >= minimum_effect and lower95 > 0.0),
        "replacement_passes": bool(lower95 > minimum_effect),
        "proposal_supports_positive_effect_only": bool(lower95 > 0.0),
        "replacement_supports_minimum_effect": bool(lower95 > minimum_effect),
    }


def _validate_finite_window_table(
    windows: pd.DataFrame, *, duration_s: float, name: str,
) -> None:
    required = {"cluster_id", "status", "start_s", "end_s", "missing_pct"}
    if not required <= set(windows.columns):
        raise ValueError(f"amplitude windows missing {sorted(required - set(windows.columns))}")
    selected = windows[windows.status == FINITE_FIT_STATUS]
    try:
        values = selected[["start_s", "end_s", "missing_pct"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: finite-interior windows must be numeric") from error
    if len(values) and (
        not np.isfinite(values).all()
        or np.any(values[:, 1] <= values[:, 0])
        or np.any(values[:, 0] < 0.0)
        or np.any(values[:, 1] > duration_s)
    ):
        raise ValueError(
            f"{name}: finite-interior windows must be finite, positive, and wholly inside [0, duration_s]"
        )
    for _, group in selected.groupby("cluster_id", dropna=False):
        ordered = group.sort_values("start_s")[["start_s", "end_s"]].to_numpy(dtype=float)
        if len(ordered) and np.any(ordered[1:, 0] < ordered[:-1, 1]):
            raise ValueError(f"{name}: finite-interior windows must be nonoverlapping within each unit")


def _finite_windows(windows: pd.DataFrame, unit: int) -> np.ndarray:
    selected = windows[(windows.cluster_id == unit) & (windows.status == FINITE_FIT_STATUS)]
    return selected.sort_values("start_s")[["start_s", "end_s", "missing_pct"]].to_numpy(dtype=float)


def _bounded_fraction(numerator: float, denominator: float, *, name: str) -> float:
    """Construct a finite coverage fraction and enforce its closed unit range."""
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite([numerator, denominator]).all() or denominator <= 0:
        raise RuntimeError(f"{name}: nonfinite numerator/denominator or nonpositive denominator")
    tolerance = np.finfo(float).eps * max(1.0, abs(denominator)) * 8
    if numerator < -tolerance or numerator > denominator + tolerance:
        raise RuntimeError(f"{name}: coverage lies outside [0, 1]")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _intersect_windows(a: np.ndarray, b: np.ndarray) -> tuple[float, float | None]:
    i = j = 0
    seconds = weighted = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            seconds += hi - lo
            weighted += (hi - lo) * (a[i, 2] - b[j, 2])
        if a[i, 1] <= b[j, 1]:
            i += 1
        else:
            j += 1
    return float(seconds), float(weighted / seconds) if seconds else None


def conditional_amplitude_diagnostic(
    reference_windows: pd.DataFrame,
    candidate_windows: pd.DataFrame,
    primary_matches: pd.DataFrame,
    reference_cohort: Sequence[int],
    *,
    duration_s: float,
    minimum_valid_windows: int = 2,
    minimum_common_time_fraction: float = 0.5,
) -> dict[str, Any]:
    """Retain existing fit-quality rules and expose conditional support fully."""
    duration_s = float(duration_s)
    if (
        not np.isfinite(duration_s)
        or duration_s <= 0
        or minimum_valid_windows <= 0
        or not 0 <= minimum_common_time_fraction <= 1
    ):
        raise ValueError("invalid amplitude diagnostic configuration")
    _validate_finite_window_table(reference_windows, duration_s=duration_s, name="reference")
    _validate_finite_window_table(candidate_windows, duration_s=duration_s, name="candidate")
    cohort = np.asarray(reference_cohort, dtype=np.int64)
    if not len(cohort) or len(np.unique(cohort)) != len(cohort):
        raise ValueError("reference cohort must be nonempty and unique")
    if len(primary_matches) and (
        primary_matches.baseline_cluster.duplicated().any()
        or primary_matches.candidate_cluster.duplicated().any()
    ):
        raise ValueError("primary matches must be one-to-one")
    partners = primary_matches.set_index("baseline_cluster") if len(primary_matches) else pd.DataFrame()
    rows = []
    for unit in cohort:
        rw = _finite_windows(reference_windows, int(unit))
        reference_union = float(np.sum(rw[:, 1] - rw[:, 0])) if len(rw) else 0.0
        if unit not in partners.index:
            partner = None
            cw = np.empty((0, 3), dtype=float)
            common_s, effect = 0.0, None
        else:
            partner = int(partners.loc[unit].candidate_cluster)
            cw = _finite_windows(candidate_windows, partner)
            common_s, effect = _intersect_windows(rw, cw)
        candidate_union = float(np.sum(cw[:, 1] - cw[:, 0])) if len(cw) else 0.0
        supported = bool(
            partner is not None
            and len(rw) >= minimum_valid_windows
            and len(cw) >= minimum_valid_windows
            and _bounded_fraction(common_s, duration_s, name="per-unit common-time")
            >= minimum_common_time_fraction
        )
        rows.append({
            "reference_unit": int(unit), "candidate_partner": partner,
            "reference_valid_windows": int(len(rw)), "candidate_valid_windows": int(len(cw)),
            "reference_support_union_s": reference_union,
            "candidate_support_union_s": candidate_union,
            "common_time_s": common_s,
            "common_time_fraction": _bounded_fraction(
                common_s, duration_s, name="per-unit common-time"
            ),
            "supported": supported,
            "reference_minus_candidate_missingness_pp": effect if supported else None,
            "status": "supported" if supported else "unmatched" if partner is None else "insufficient_supported_windows",
        })
    table = pd.DataFrame(rows)
    supported = table[table.supported]
    cohort_time = len(table) * duration_s
    return {
        "schema": SCHEMA,
        "scope": "conditional supported-window diagnostic only; not an advancement endpoint",
        "fit_quality_rule": "status == finite_interior; cached fits unchanged; no refit or threshold relaxation",
        "reference_cohort_units": int(len(table)),
        "matched_units": int(table.candidate_partner.notna().sum()),
        "supported_units": int(table.supported.sum()),
        "supported_unit_fraction_full_cohort": _bounded_fraction(
            float(table.supported.sum()), float(len(table)), name="supported-unit"
        ),
        "reference_support_union_fraction_full_cohort_time": _bounded_fraction(
            float(table.reference_support_union_s.sum()), cohort_time,
            name="reference full-cohort-time support",
        ),
        "candidate_support_union_fraction_full_cohort_time": _bounded_fraction(
            float(table.candidate_support_union_s.sum()), cohort_time,
            name="candidate full-cohort-time support",
        ),
        "common_time_fraction_full_cohort_time": _bounded_fraction(
            float(table.common_time_s.sum()), cohort_time,
            name="common full-cohort-time support",
        ),
        "conditional_median_missingness_difference_pp": (
            float(supported.reference_minus_candidate_missingness_pp.median()) if len(supported) else None
        ),
        "unit_support": table,
    }
