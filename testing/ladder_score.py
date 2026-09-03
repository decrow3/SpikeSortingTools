"""`score_sort` — the one scoring function the evaluation ladder runs at every tier.

`docs/pipeline_improvement_plan.md` §3 rule 1: *L1 computes the identical score
dictionary as L4.* This module is that dictionary. It is deliberately a pure
function of already-written sorter output plus (optionally) an injected-truth
manifest and a comparator sort — it never runs a sorter, never touches `/mnt`,
and never reads raw voltage.

Three layers, matching §5:

* **primary** — hybrid ground truth. Per injected unit: accuracy, split, merge,
  identity continuity. Headline: *units recovered at accuracy ≥ 0.8 with no
  split and no merge* — one integer, `headline_units_recovered`.
* **secondary** — real-data symmetric agreement against a reference sort:
  `gained_good` / `lost_good`, never a bare net. Identity uses the corrected
  one-to-one coincidence machinery. Detection-loss classification is withheld
  until spatial, null-controlled evidence is available.
* **guardrails** — similar good–good pairs per good unit, refractory-violation
  distribution, edge-spike fraction, runtime per unit data. Any breach blocks
  promotion.

`context` holds KS-good count, spike totals and the like: recorded for
provenance, **never** a promotion endpoint (§5 "Explicitly not endpoints").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from testing.luke_rescue_unique_units_audit import (
    REFRACTORY_MS,
    SIMILARITY_THRESHOLD,
    DEPTH_WINDOW_UM,
    load_sort,
    mutual_best_matches,
    nearby_similar_good_pairs,
)

SCORE_SCHEMA = "luke-ladder-score-sort-v2"

# §5 "Explicitly not endpoints". Present in `context` for provenance only; a
# promotion decision that cites any of these is out of contract.
NOT_ENDPOINTS = (
    "ks_good_count",
    "total_spikes",
    "n_clusters",
    "stable_bin_occupancy",
)

DEFAULT_TOL_MS = 0.5
DEFAULT_BIN_S = 30.0
CAPTURE_FRAC = 0.05          # §5: "> 5% of the injected train"
ACCURACY_GATE = 0.8          # §5 headline
EDGE_UM = 40.0               # §5 guardrail: "Edge-spike fraction (40 µm)"


# --------------------------------------------------------------------------- #
# coincidence
# --------------------------------------------------------------------------- #
def coincident_mask(a: np.ndarray, b: np.ndarray, tol: int) -> np.ndarray:
    """Boolean mask over `a`: True where some `b` sample is within `tol`."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.size == 0 or b.size == 0:
        return np.zeros(a.shape, dtype=bool)
    b = np.sort(b)
    idx = np.searchsorted(b, a)
    left = b[np.clip(idx - 1, 0, b.size - 1)]
    right = b[np.clip(idx, 0, b.size - 1)]
    return np.minimum(np.abs(a - left), np.abs(a - right)) <= tol


# --------------------------------------------------------------------------- #
# primary — hybrid ground truth
# --------------------------------------------------------------------------- #
def _exclusive_event_matches(
    truth: Mapping[str, np.ndarray], st: np.ndarray, tol: int
):
    """Maximum-cardinality one-to-one matches between all truth and output events.

    Matching all injected trains together prevents one output spike from being
    credited to two nearby truth events.  Inputs are interval ordered, so the
    standard two-pointer matcher is maximum-cardinality for a fixed tolerance.
    """
    truth_times = np.concatenate(list(truth.values())) if truth else np.array([], dtype=np.int64)
    truth_ids = np.concatenate([
        np.full(v.size, k, dtype=object) for k, v in truth.items()
    ]) if truth else np.array([], dtype=object)
    truth_order = np.argsort(truth_times, kind="stable")
    truth_times = truth_times[truth_order]
    truth_ids = truth_ids[truth_order]

    output_order = np.argsort(st, kind="stable")
    output_times = np.asarray(st, dtype=np.int64)[output_order]
    matched_truth: list[int] = []
    matched_output: list[int] = []
    i = j = 0
    while i < truth_times.size and j < output_times.size:
        delta = int(output_times[j]) - int(truth_times[i])
        if delta < -tol:
            j += 1
        elif delta > tol:
            i += 1
        else:
            matched_truth.append(i)
            matched_output.append(int(output_order[j]))
            i += 1
            j += 1
    return truth_times, truth_ids, np.asarray(matched_truth), np.asarray(matched_output)


def _identity_continuity(truth_st, trains, matched_times_by_out, edges):
    labels: list[int | None] = []
    bin_acc: list[float] = []
    n_bins_scored = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = truth_st[(truth_st >= lo) & (truth_st < hi)]
        if seg.size < 5:  # too few injected spikes in this bin to adjudicate
            continue
        n_bins_scored += 1
        best, best_tp = None, 0
        for oid in trains:
            mt = np.asarray(matched_times_by_out.get(oid, []), dtype=np.int64)
            tp = int(((mt >= lo) & (mt < hi)).sum())
            if tp > best_tp:
                best, best_tp = oid, tp
        labels.append(best if best_tp else None)
        if best is None:
            bin_acc.append(0.0)
            continue
        o_seg = trains[best][(trains[best] >= lo) & (trains[best] < hi)]
        fn = seg.size - best_tp
        fp = o_seg.size - best_tp
        denom = best_tp + fp + fn
        bin_acc.append(best_tp / denom if denom else 0.0)
    seq = [x for x in labels if x is not None]
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return {
        "label_switches": int(switches),
        "n_bins_scored": n_bins_scored,
        "min_bin_accuracy": float(min(bin_acc)) if bin_acc else float("nan"),
    }


def ground_truth_scores(
    sort: Mapping,
    truth: Mapping,
    fs: float,
    *,
    duration_s: float,
    tol_ms: float = DEFAULT_TOL_MS,
    bin_s: float = DEFAULT_BIN_S,
) -> dict:
    """Score a sort against a known injected spike-train manifest (§5 primary)."""
    tol = int(round(tol_ms / 1000.0 * fs))
    st, cl = np.asarray(sort["st"]), np.asarray(sort["cl"])
    trains = {int(c): np.sort(st[cl == c]) for c in np.unique(cl)}
    truth = {
        str(k): np.sort(np.asarray(v, dtype=np.int64)) for k, v in truth.items()
    }
    edges = np.arange(0.0, duration_s * fs + 1.0, bin_s * fs)
    if edges.size < 2:
        edges = np.array([0.0, duration_s * fs])

    truth_times, truth_ids, mt, mo = _exclusive_event_matches(truth, st, tol)
    caught: dict[tuple[str, int], int] = {}
    matched_times: dict[str, dict[int, list[int]]] = {tid: {} for tid in truth}
    for ti, oi in zip(mt, mo):
        tid = str(truth_ids[ti])
        oid = int(cl[oi])
        caught[(tid, oid)] = caught.get((tid, oid), 0) + 1
        matched_times[tid].setdefault(oid, []).append(int(truth_times[ti]))

    units = []
    for tid, tst in truth.items():
        n_truth = int(tst.size)
        frac_by_out = {
            oid: caught.get((tid, oid), 0) / n_truth for oid in trains
        }
        best = max(frac_by_out, key=frac_by_out.get) if frac_by_out else None
        tp = caught.get((tid, best), 0)
        fn = n_truth - tp
        fp = int(trains[best].size - tp) if best is not None else 0
        denom = tp + fp + fn
        accuracy = tp / denom if denom else 0.0
        n_capturing = sum(1 for f in frac_by_out.values() if f > CAPTURE_FRAC)
        merged_with = [
            otid
            for otid in truth
            if otid != tid
            and best is not None
            and caught.get((otid, best), 0) / truth[otid].size > CAPTURE_FRAC
        ]
        merge = len(merged_with) > 0
        cont = _identity_continuity(tst, trains, matched_times[tid], edges)
        recovered = (
            accuracy >= ACCURACY_GATE and n_capturing <= 1 and not merge
        )
        units.append({
            "truth_unit": tid,
            "n_truth": n_truth,
            "best_output_unit": best,
            "best_output_label": sort["label"].get(best, "?"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "accuracy": float(accuracy),
            "n_output_units_capturing": int(n_capturing),
            "split": bool(n_capturing > 1),
            "merge": bool(merge),
            "merged_with": merged_with,
            "label_switches": cont["label_switches"],
            "min_bin_accuracy": cont["min_bin_accuracy"],
            "n_bins_scored": cont["n_bins_scored"],
            "recovered": bool(recovered),
        })

    return {
        "units": units,
        "headline_units_recovered": int(sum(u["recovered"] for u in units)),
        "n_injected": len(units),
        "accuracy_gate": ACCURACY_GATE,
        "capture_fraction": CAPTURE_FRAC,
        "tolerance_ms": tol_ms,
        "bin_s": bin_s,
    }


# --------------------------------------------------------------------------- #
# secondary — real-data symmetric agreement
# --------------------------------------------------------------------------- #
def window_reference_sort(
    reference,
    *,
    start_frame: int,
    end_frame: int,
    depth_range: tuple[float, float] | None = None,
) -> dict:
    """Restrict a full reference sort to a snippet's window and depth strip.

    A snippet sort's spike times are 0-based within the window; a full-session
    reference is on the session clock and spans the whole probe. Comparing them
    (`symmetric_agreement`) requires the reference re-based to the same frames
    and cut to the same depth band. A reference good unit is kept only if it
    still has spikes in the window; with `depth_range`, only if its median spike
    depth falls in the band.
    """
    ref_dir = None if isinstance(reference, Mapping) else Path(reference)
    ref = reference if isinstance(reference, Mapping) else load_sort(ref_dir)
    st, cl = np.asarray(ref["st"]), np.asarray(ref["cl"])
    in_window = (st >= start_frame) & (st < end_frame)

    if depth_range is not None and ref_dir is not None:
        pos_path, clu_path = ref_dir / "spike_positions.npy", ref_dir / "spike_clusters.npy"
        if pos_path.exists() and clu_path.exists():
            native_depth = np.load(pos_path)[:, 1]
            native_clu = np.load(clu_path).reshape(-1)
            lo, hi = depth_range
            keep_unit = {
                int(c)
                for c in np.unique(native_clu)
                if lo <= float(np.median(native_depth[native_clu == c])) <= hi
            }
            in_window &= np.isin(cl, list(keep_unit))

    w_st = (st[in_window] - start_frame).astype(np.int64)
    w_cl = cl[in_window]
    present = set(np.unique(w_cl).tolist())
    return {
        "st": w_st,
        "cl": w_cl,
        "label": dict(ref["label"]),
        "good": {c for c in ref["good"] if c in present},
    }


def symmetric_agreement(sort: Mapping, reference) -> dict:
    """`+N / -M` KS-good agreement against a reference sort (§5 secondary).

    Never returns a bare net. Detection-loss status is deliberately unresolved
    here: this generic interface has no validated spatial identity/null model,
    and whole-probe temporal coincidence cannot establish presence or absence.
    """
    ref = reference if isinstance(reference, Mapping) else load_sort(Path(reference))
    matches = mutual_best_matches(sort, ref)
    matched_cand = set(matches["rescue_cluster"]) if len(matches) else set()
    matched_ref = set(matches["legacy_cluster"]) if len(matches) else set()
    gained = sorted(set(sort["good"]) - matched_cand)
    lost = sorted(set(ref["good"]) - matched_ref)

    return {
        "matched_good_pairs": int(len(matches)),
        "gained_good": len(gained),
        "lost_good": len(lost),
        "net_good": len(gained) - len(lost),
        "lost_absent_at_detection": None,
        "lost_detection_status": "unresolved_requires_spatial_null_audit",
        "reference_good": len(ref["good"]),
        "candidate_good": len(sort["good"]),
        "gained_ids": [int(c) for c in gained],
        "lost_ids": [int(c) for c in lost],
    }


# --------------------------------------------------------------------------- #
# guardrails
# --------------------------------------------------------------------------- #
def _edge_spike_fraction(path: Path, good: set) -> float:
    """Fraction of good-unit spikes localised within EDGE_UM of a probe end.

    Reads the native `spike_positions.npy` / `spike_clusters.npy` order — these
    are row-aligned as written by Kilosort, independent of any re-sort applied
    by `load_sort`.
    """
    pos_path = path / "spike_positions.npy"
    geom_path = path / "channel_positions.npy"
    clu_path = path / "spike_clusters.npy"
    if not (pos_path.exists() and geom_path.exists() and clu_path.exists()):
        return float("nan")
    depth = np.load(pos_path)[:, 1]
    clu = np.load(clu_path).reshape(-1)
    chan_y = np.load(geom_path)[:, 1]
    lo, hi = float(chan_y.min()), float(chan_y.max())
    good_mask = np.isin(clu, list(good))
    if not good_mask.any():
        return float("nan")
    d = depth[good_mask]
    near_edge = (d <= lo + EDGE_UM) | (d >= hi - EDGE_UM)
    return float(near_edge.mean())


def guardrails(path: Path, sort: Mapping, fs: float) -> dict:
    path = Path(path)
    first, _ = nearby_similar_good_pairs(path)
    n_good = max(len(sort["good"]), 1)

    rv = []
    for c in sorted(sort["good"]):
        a = np.sort(sort["st"][sort["cl"] == c])
        if a.size > 1:
            isi_ms = np.diff(a) / fs * 1000.0
            rv.append(float((isi_ms < REFRACTORY_MS).mean()))
    rv = np.asarray(rv) if rv else np.array([np.nan])

    return {
        "similar_good_good_pairs": int(len(first)),
        "similar_pairs_per_good_unit": float(len(first) / n_good),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "depth_window_um": DEPTH_WINDOW_UM,
        "refractory_violation_median": float(np.nanmedian(rv)),
        "refractory_violation_p90": float(np.nanpercentile(rv, 90)),
        "refractory_violation_frac_over_1pct": float(np.nanmean(rv > 0.01)),
        "refractory_period_ms": REFRACTORY_MS,
        "edge_spike_fraction_40um": _edge_spike_fraction(path, set(sort["good"])),
    }


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def _params_sample_rate(path: Path) -> float | None:
    params = path / "params.py"
    if not params.exists():
        return None
    ns: dict = {}
    exec(params.read_text(), {}, ns)  # KS params.py is trusted local output
    return float(ns["sample_rate"]) if "sample_rate" in ns else None


def score_sort(
    sorter_output,
    *,
    snippet: Mapping | None = None,
    truth: Mapping | None = None,
    reference=None,
    reference_window: tuple[int, int] | None = None,
    reference_depth_range: tuple[float, float] | None = None,
    runtime_s: float | None = None,
    fs: float | None = None,
    duration_s: float | None = None,
) -> dict:
    """The ladder's single scoring endpoint. Identical at L1 and L4.

    Parameters
    ----------
    sorter_output
        Path to a curated Kilosort-style output directory.
    snippet
        Optional frozen-panel snippet record; when given, supplies `fs`,
        `duration_s`, `truth` and `reference` if those are not passed directly.
    truth
        `{injected_unit_id: array of sample indices}` for the primary metric.
    reference
        A comparator sort (path or loaded dict) for the secondary metric.
    runtime_s
        Wall clock for the run that produced `sorter_output`, for the runtime
        guardrail.
    """
    path = Path(sorter_output)
    snippet = dict(snippet) if snippet else {}
    truth = truth if truth is not None else snippet.get("truth")
    reference = reference if reference is not None else snippet.get("reference")
    reference_window = reference_window or snippet.get("reference_window")
    reference_depth_range = reference_depth_range or snippet.get("reference_depth_range")
    fs = fs or snippet.get("fs") or _params_sample_rate(path)
    if fs is None:
        raise ValueError("sampling rate unknown: pass fs= or a snippet with fs")

    sort = load_sort(path)
    if duration_s is None:
        duration_s = snippet.get("duration_s")
    if duration_s is None:
        duration_s = (float(sort["st"].max()) + 1.0) / fs if sort["st"].size else 0.0

    result = {
        "schema": SCORE_SCHEMA,
        "sorter_output": str(path),
        "fs": float(fs),
        "duration_s": float(duration_s),
        "primary": None,
        "secondary": None,
        "guardrails": guardrails(path, sort, fs),
        "runtime": {
            "runtime_s": runtime_s,
            "runtime_s_per_recording_s": (
                runtime_s / duration_s if runtime_s and duration_s else None
            ),
            "runtime_s_per_good_unit": (
                runtime_s / len(sort["good"]) if runtime_s and sort["good"] else None
            ),
        },
        "context": {
            "_warning": "provenance only; NOT promotion endpoints (plan §5)",
            "not_endpoints": list(NOT_ENDPOINTS),
            "ks_good_count": len(sort["good"]),
            "total_spikes": int(sort["st"].size),
            "n_clusters": int(np.unique(sort["cl"]).size),
        },
        "headline": None,
    }

    if truth:
        result["primary"] = ground_truth_scores(
            sort, truth, fs, duration_s=duration_s
        )
        result["headline"] = result["primary"]["headline_units_recovered"]
    if reference is not None:
        if reference_window is not None:
            reference = window_reference_sort(
                reference,
                start_frame=int(reference_window[0]),
                end_frame=int(reference_window[1]),
                depth_range=reference_depth_range,
            )
            result["context"]["reference_windowed"] = {
                "start_frame": int(reference_window[0]),
                "end_frame": int(reference_window[1]),
                "depth_range_um": list(reference_depth_range)
                if reference_depth_range
                else None,
                "reference_good_in_window": len(reference["good"]),
            }
        result["secondary"] = symmetric_agreement(sort, reference)

    return result


def _load_truth_manifest(path: Path) -> dict:
    """A JSON `{unit_id: [sample, ...]}` or an .npz of per-unit int arrays."""
    if path.suffix == ".npz":
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    raw = json.loads(path.read_text())
    return {str(k): np.asarray(v, dtype=np.int64) for k, v in raw.items()}


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("sorter_output", type=Path)
    ap.add_argument("--reference", type=Path, help="comparator sort directory")
    ap.add_argument("--truth", type=Path, help="injected-truth JSON or .npz")
    ap.add_argument("--runtime-s", type=float)
    ap.add_argument("--fs", type=float)
    ap.add_argument("--duration-s", type=float)
    ap.add_argument("--out", type=Path, help="write the score dict here as JSON")
    args = ap.parse_args(argv)

    result = score_sort(
        args.sorter_output,
        truth=_load_truth_manifest(args.truth) if args.truth else None,
        reference=args.reference,
        runtime_s=args.runtime_s,
        fs=args.fs,
        duration_s=args.duration_s,
    )
    encoded = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
