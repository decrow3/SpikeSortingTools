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

SCORE_SCHEMA = "luke-ladder-score-sort-v3"
TRUTH_CONTRACT_SCHEMA = "luke-ladder-truth-contract-v1"

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
CHANCE_MARGIN = 3.0          # a split/merge participant must beat chance coincidence by this factor
ACCURACY_GATE = 0.8          # §5 headline
EDGE_UM = 40.0               # §5 guardrail: "Edge-spike fraction (40 µm)"


# --------------------------------------------------------------------------- #
# truth contract — what the scorer is allowed to score against
# --------------------------------------------------------------------------- #
class TruthContractError(ValueError):
    """A paired-arm comparison whose truth or spatial support is not identical."""


def truth_digest(truth: Mapping) -> str:
    """Content hash of an injected truth train set.

    Canonical over unit id and sample values, so any difference -- a spike
    added, an admission filter not applied, a train reconstructed from a prespec
    rather than passed through -- changes the digest.
    """
    import hashlib

    digest = hashlib.sha256(TRUTH_CONTRACT_SCHEMA.encode())
    for unit_id in sorted(map(str, truth)):
        events = np.sort(np.asarray(truth[unit_id], dtype=np.int64))
        digest.update(b"\x00" + str(unit_id).encode() + b"\x00")
        digest.update(np.asarray(events.size, dtype=np.int64).tobytes())
        digest.update(events.tobytes())
    return digest.hexdigest()


def array_digest(values) -> str:
    """Content hash of a channel-id or geometry array."""
    import hashlib

    array = np.ascontiguousarray(values)
    if array.dtype.kind in "SU O".replace(" ", ""):
        payload = "\x00".join(map(str, array.ravel().tolist())).encode()
    else:
        payload = array.astype(np.float64).tobytes()
    return hashlib.sha256(
        f"{array.shape}|".encode() + payload
    ).hexdigest()


def build_truth_contract(
    truth: Mapping,
    *,
    admission: Mapping,
    channel_ids,
    geometry,
    filtered_before_injection: bool,
    crop: tuple[int, int] | None = None,
) -> dict:
    """Bind an admitted train, its provenance and its spatial support together.

    `admission` must carry the schema and parameters that produced the filter,
    the totals, and the per-level counts, so a reader can tell *which* events
    were admitted and why without re-deriving them.
    """
    if not filtered_before_injection:
        raise TruthContractError(
            "truth must be filtered before injection, not merely before scoring: "
            "excluded events otherwise still shape detection and templates"
        )
    for key in ("schema", "rule", "n_total", "n_admitted", "counts_by_level_um"):
        if key not in admission:
            raise TruthContractError(f"admission record is missing {key!r}")
    n_expected = int(sum(np.asarray(v).size for v in truth.values()))
    if n_expected != int(admission["n_admitted"]):
        raise TruthContractError(
            f"truth holds {n_expected} events but admission claims "
            f"{admission['n_admitted']}"
        )
    return {
        "schema": TRUTH_CONTRACT_SCHEMA,
        "truth_sha256": truth_digest(truth),
        "n_expected": n_expected,
        "units": {
            str(k): {
                "n": int(np.asarray(v).size),
                "sha256": truth_digest({str(k): v}),
            }
            for k, v in truth.items()
        },
        "admission": dict(admission),
        "filtered_before_injection": True,
        "spatial": {
            "n_channels": int(len(channel_ids)),
            "channel_ids_sha256": array_digest(np.asarray(channel_ids)),
            "geometry_sha256": array_digest(np.asarray(geometry, dtype=np.float64)),
            "crop": list(crop) if crop else None,
        },
    }


def validate_truth_contract(truth: Mapping, contract: Mapping) -> dict:
    """Fail closed unless `truth` is exactly the train the contract describes."""
    if contract.get("schema") != TRUTH_CONTRACT_SCHEMA:
        raise TruthContractError(f"unknown truth contract schema {contract.get('schema')!r}")
    if not contract.get("filtered_before_injection"):
        raise TruthContractError("contract does not attest filtering before injection")
    observed = truth_digest(truth)
    if observed != contract["truth_sha256"]:
        raise TruthContractError(
            "truth does not match its contract — the scorer was handed a "
            f"different train (expected {contract['truth_sha256'][:12]}, "
            f"got {observed[:12]})"
        )
    n_observed = int(sum(np.asarray(v).size for v in truth.values()))
    if n_observed != int(contract["n_expected"]):
        raise TruthContractError(
            f"expected {contract['n_expected']} admitted events, got {n_observed}"
        )
    return dict(contract)


def assert_paired_truth(contracts, *, labels=None) -> dict:
    """Every paired arm must share one truth and one spatial support.

    A drift penalty is a within-subject Δ; if two arms score different trains,
    or sort different channels, the Δ is not attributable to the arm.
    """
    contracts = list(contracts)
    labels = list(labels) if labels else [f"arm{i}" for i in range(len(contracts))]
    if len(contracts) < 2:
        raise TruthContractError("a paired comparison needs at least two arms")
    reference = contracts[0]
    for key, path in (
        ("truth", ("truth_sha256",)),
        ("n_expected", ("n_expected",)),
        ("channel ids", ("spatial", "channel_ids_sha256")),
        ("geometry", ("spatial", "geometry_sha256")),
    ):
        def dig(contract, path=path):
            value = contract
            for step in path:
                value = value[step]
            return value

        expected = dig(reference)
        for label, contract in zip(labels[1:], contracts[1:]):
            if dig(contract) != expected:
                raise TruthContractError(
                    f"{key} differs between {labels[0]!r} and {label!r}: "
                    f"{expected!r} vs {dig(contract)!r}"
                )
    return {
        "arms": labels,
        "truth_sha256": reference["truth_sha256"],
        "n_expected": int(reference["n_expected"]),
        "channel_ids_sha256": reference["spatial"]["channel_ids_sha256"],
        "geometry_sha256": reference["spatial"]["geometry_sha256"],
        "identical_denominator": True,
    }


# --------------------------------------------------------------------------- #
# coincidence
# --------------------------------------------------------------------------- #
def coincident_mask(
    a: np.ndarray, b: np.ndarray, tol: int, *, b_sorted: bool = False
) -> np.ndarray:
    """Boolean mask over `a`: True where some `b` sample is within `tol`.

    Pass ``b_sorted=True`` to skip the internal sort when `b` is already
    ascending (the hot path in `ground_truth_scores`, where `b` is a
    normalised truth train).
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.size == 0 or b.size == 0:
        return np.zeros(a.shape, dtype=bool)
    if not b_sorted:
        b = np.sort(b)
    idx = np.searchsorted(b, a)
    left = b[np.clip(idx - 1, 0, b.size - 1)]
    right = b[np.clip(idx, 0, b.size - 1)]
    return np.minimum(np.abs(a - left), np.abs(a - right)) <= tol


# --------------------------------------------------------------------------- #
# primary — hybrid ground truth
# --------------------------------------------------------------------------- #
def _exclusive_pairs(a: np.ndarray, b: np.ndarray, tol: int):
    """Greedy interval-order 1:1 matches between two *sorted* int arrays.

    Each event is used at most once. For a fixed tolerance and interval-ordered
    inputs the greedy two-pointer is maximum-cardinality. Returns the indices
    into `a` and `b` of the matched pairs.

    Exclusivity is enforced **only between the two arrays passed in** — a single
    truth train and a single output cluster. It is deliberately not enforced
    across output clusters: on a dense recording that lets any cluster with a
    chance coincidence steal ownership of an injected event, which produced a
    constant ~10% false-negative/false-positive floor regardless of donor
    (see docs/decisions/0014). Cross-cluster competition is resolved afterwards
    by picking the best cluster, and duplicated capture across clusters is the
    signal the split/merge diagnostics are meant to detect.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    ai: list[int] = []
    bi: list[int] = []
    i = j = 0
    while i < a.size and j < b.size:
        delta = int(b[j]) - int(a[i])
        if delta < -tol:
            j += 1
        elif delta > tol:
            i += 1
        else:
            ai.append(i)
            bi.append(j)
            i += 1
            j += 1
    return np.asarray(ai, dtype=np.int64), np.asarray(bi, dtype=np.int64)


def _identity_continuity(truth_st, per_cluster, edges):
    """Per-bin owner and accuracy of an injected train over time.

    `per_cluster[oid]` carries `matched_truth_times` and `unmatched_out_times`
    from the single global exclusive match against that cluster. A bin's TP/FN
    are counted on the **truth clock** (a matched pair belongs to the bin of its
    truth event) and its FP on unmatched output spikes only — so a match that
    straddles a bin edge can never make FP negative or accuracy exceed 1.
    """
    labels: list[int | None] = []
    bin_acc: list[float] = []
    n_bins_scored = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = truth_st[(truth_st >= lo) & (truth_st < hi)]
        if seg.size < 5:  # too few injected spikes in this bin to adjudicate
            continue
        n_bins_scored += 1
        best, best_tp = None, 0
        for oid, v in per_cluster.items():
            mt = v["matched_truth_times"]
            tp = int(((mt >= lo) & (mt < hi)).sum())
            if tp > best_tp:
                best, best_tp = oid, tp
        labels.append(best if best_tp else None)
        if best is None:
            bin_acc.append(0.0)
            continue
        um = per_cluster[best]["unmatched_out_times"]
        fp = int(((um >= lo) & (um < hi)).sum())
        fn = seg.size - best_tp
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
    """Score a sort against a known injected spike-train manifest (§5 primary).

    Recovery is scored **per candidate output cluster**: each injected train is
    matched exclusively 1:1 against one cluster's spikes at a time, and the best
    cluster (by accuracy, then TP, then fewest FP) is the primary result. The
    split diagnostic counts every cluster that captures > 5% of the train *and*
    whose overlap beats chance coincidence by CHANCE_MARGIN — the chance test
    rejects the dense-background floor without suppressing genuine (even
    contaminated) fragments. The merge check uses the same chance gate. See
    docs/decisions/0014.
    """
    tol = int(round(tol_ms / 1000.0 * fs))
    st = np.asarray(sort["st"], dtype=np.int64)
    cl = np.asarray(sort["cl"])
    st_order = np.argsort(st, kind="stable")
    st, cl = st[st_order], cl[st_order]
    # Group into per-cluster, time-ordered trains in one pass: a stable argsort
    # by cluster id keeps each cluster's spikes in their existing time order.
    # (Avoids one full-length `cl == c` scan per cluster — ~700x on a big sort.)
    cl_order = np.argsort(cl, kind="stable")
    cl_grouped = cl[cl_order]
    st_grouped = st[cl_order]
    uniq, starts = np.unique(cl_grouped, return_index=True)
    ends = np.append(starts[1:], cl_grouped.size)
    trains = {
        int(c): st_grouped[s:e] for c, s, e in zip(uniq, starts, ends)
    }
    truth = {
        str(k): np.sort(np.asarray(v, dtype=np.int64)) for k, v in truth.items()
    }
    edges = np.arange(0.0, duration_s * fs + 1.0, bin_s * fs)
    if edges.size < 2:
        edges = np.array([0.0, duration_s * fs])
    total_samples = max(
        float(duration_s * fs),
        float(st[-1] + 1) if st.size else 0.0,
        *(float(v[-1] + 1) for v in truth.values() if v.size),
    ) or 1.0
    window = 2 * tol + 1

    def _chance_tp(n_a: int, n_b: int) -> float:
        """Expected coincidences within ±tol between two uniform-random trains."""
        return n_a * n_b * window / total_samples

    def _candidate_clusters(tst: np.ndarray) -> list[int]:
        """Clusters with at least one spike within tol of this train."""
        if st.size == 0 or tst.size == 0:
            return []
        near = coincident_mask(st, tst, tol, b_sorted=True)  # tst is normalised
        return [int(c) for c in np.unique(cl[near])]

    units = []
    for tid, tst in truth.items():
        n_truth = int(tst.size)
        # exclusive 1:1 match against each candidate cluster independently
        per_cluster: dict[int, dict] = {}
        for oid in _candidate_clusters(tst):
            ost = trains[oid]
            ta, tb = _exclusive_pairs(tst, ost, tol)
            tp_c = int(ta.size)
            fp_c = int(ost.size - tp_c)
            fn_c = n_truth - tp_c
            denom_c = tp_c + fp_c + fn_c
            unmatched_out = np.delete(ost, tb) if tb.size else ost
            per_cluster[oid] = {
                "tp": tp_c,
                "fp": fp_c,
                "fn": fn_c,
                "accuracy": tp_c / denom_c if denom_c else 0.0,
                "capture": tp_c / n_truth if n_truth else 0.0,
                # informative only: fraction of the cluster that is injected
                # train (~1.0 for a real fragment, ~0.005 for a chance clip).
                "precision": tp_c / ost.size if ost.size else 0.0,
                "matched_truth_times": tst[ta],
                "unmatched_out_times": unmatched_out,
            }

        if per_cluster:
            best = max(
                per_cluster,
                key=lambda o: (
                    per_cluster[o]["accuracy"],
                    per_cluster[o]["tp"],
                    -per_cluster[o]["fp"],
                ),
            )
            tp = per_cluster[best]["tp"]
            fp = per_cluster[best]["fp"]
            fn = per_cluster[best]["fn"]
            accuracy = per_cluster[best]["accuracy"]
        else:
            best, tp, fp, fn, accuracy = None, 0, 0, n_truth, 0.0

        # split burden: clusters that capture > 5% of the train AND whose TP
        # beats chance coincidence by CHANCE_MARGIN. The chance test rejects the
        # dense-background floor (a high-rate cluster clips >5% of a sparse train
        # by luck) without suppressing a genuine, even contaminated, fragment
        # whose overlap is many times chance (docs/decisions/0014).
        capturing = [
            o for o, v in per_cluster.items()
            if v["capture"] > CAPTURE_FRAC
            and v["tp"] > CHANCE_MARGIN * _chance_tp(n_truth, trains[o].size)
        ]
        n_capturing = len(capturing)

        # merge burden: does the best cluster capture > 5% of another train, by
        # more than chance? (Chance coincidence between the best cluster and a
        # sparse injected train is tiny, so an imbalanced real merge — a small
        # train wholly swallowed by a large cluster — is still caught.)
        merged_with = []
        if best is not None:
            b_train = trains[best]
            for otid, otst in truth.items():
                if otid == tid or otst.size == 0:
                    continue
                oa, _ = _exclusive_pairs(otst, b_train, tol)
                if (
                    oa.size / otst.size > CAPTURE_FRAC
                    and oa.size > CHANCE_MARGIN * _chance_tp(otst.size, b_train.size)
                ):
                    merged_with.append(otid)
        merge = len(merged_with) > 0

        cont = _identity_continuity(tst, per_cluster, edges)
        recovered = (
            accuracy >= ACCURACY_GATE and n_capturing <= 1 and not merge
        )
        units.append({
            "truth_unit": tid,
            "n_truth": n_truth,
            "best_output_unit": best,
            "best_output_label": sort["label"].get(best, "?") if best is not None else "none",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "accuracy": float(accuracy),
            "n_output_units_capturing": int(n_capturing),
            "capturing_output_units": sorted(int(o) for o in capturing),
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
    truth_contract: Mapping | None = None,
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
        This is the *admitted* train and the scorer scores exactly it: it never
        reconstructs a train from a prespec.
    truth_contract
        Optional attestation binding that admitted train, the admission
        parameters that produced it, and the cropped spatial support. When
        given it is validated against `truth` and fails closed on any mismatch,
        and is recorded in the result so paired arms can be checked with
        `assert_paired_truth`.
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

    validated_contract = (
        validate_truth_contract(truth, truth_contract)
        if truth_contract is not None else None
    )
    if truth_contract is not None and not truth:
        raise TruthContractError("a truth contract was given but no truth train")

    result = {
        "schema": SCORE_SCHEMA,
        "sorter_output": str(path),
        "truth_contract": validated_contract,
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
