"""Apply CCG, rival, and independent-identity vetoes to state-pass edges.

This stage never silently mutates Kilosort output. It emits a bounded label map
and a complete reason record. The v1 rival margin was not preregistered before
the motion scores were opened, so it is explicitly exploratory and cannot
authorize a merge even if all substantive gates pass.
"""

from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path

import numpy as np

from pipeline.motion_merge_gates import (
    dominant_final_cluster_mapping,
    endpoint_rival_score,
    family_support,
)
from testing.luke_medicine_motion_aware_merge import DEFAULT_OUTPUT, DEFAULT_SORT


OUT = DEFAULT_OUTPUT / "final_gates"
STATE_OUT = DEFAULT_OUTPUT / "state_template_replay"
FAMILY_ROOT = Path("testing/outputs/luke_kilosort_depthblind_families_v2")
PLAUSIBILITY_ROOT = Path("testing/outputs/luke_kilosort_family_plausibility_v3")
MAPPING_PURITY_THRESHOLD = 0.80
RIVAL_MARGIN = 0.05


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ccg(times_a: np.ndarray, times_b: np.ndarray, ops: dict) -> dict[str, object]:
    # Import from the pinned environment used to produce the bounded replay.
    from kilosort import CCG

    kwargs = {
        "acg_threshold": float(ops["acg_threshold"]),
        "ccg_threshold": float(ops["ccg_threshold"]),
    }
    acg_a, _, q_a = CCG.check_CCG(times_a, **kwargs)
    acg_b, _, q_b = CCG.check_CCG(times_b, **kwargs)
    _, cross, q_cross = CCG.check_CCG(times_a, times_b, **kwargs)
    union_acg, _, q_union = CCG.check_CCG(np.sort(np.r_[times_a, times_b]), **kwargs)
    return {
        "cluster_a_refractory": bool(acg_a), "cluster_b_refractory": bool(acg_b),
        "cross_refractory": bool(cross), "union_refractory": bool(union_acg),
        "q12_a": float(q_a), "q12_b": float(q_b),
        "q12_cross": float(q_cross), "q12_union": float(q_union),
        "passed": bool(acg_a and acg_b and cross and union_acg),
    }


def main() -> None:
    partial = OUT.with_name(OUT.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Partial final gates require inspection: {partial}")
    if OUT.exists():
        print((OUT / "receipt.json").read_text())
        return
    partial.mkdir(parents=True)
    settings = {
        "mapping_purity_threshold": MAPPING_PURITY_THRESHOLD,
        "rival_margin": RIVAL_MARGIN,
        "rival_margin_status": "exploratory_not_preregistered_before_motion_scores_opened",
        "independent_identity_requirement": "same depth_time_coherent waveform-only family",
        "merge_authorization": "disabled_in_v1_due_to_unpreregistered_rival_margin",
    }
    (partial / "frozen_settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    pre = np.load(DEFAULT_OUTPUT / "bounded_premerge/clu.npy")
    st = np.load(DEFAULT_OUTPUT / "bounded_premerge/st.npy", mmap_mode="r")
    source_indices = np.load(DEFAULT_OUTPUT / "bounded_premerge/source_spike_indices.npy")
    full_clu = np.load(DEFAULT_SORT / "full_clu.npy", mmap_mode="r")
    if len(pre) != len(st) or len(source_indices) != len(pre):
        raise RuntimeError("bounded arrays do not align")
    mapping = dominant_final_cluster_mapping(pre, np.asarray(full_clu[source_indices]))
    identity_root = DEFAULT_OUTPUT / "identity_controls"
    identity_receipt = json.loads((identity_root / "receipt.json").read_text())
    if not identity_receipt.get("complete") or not all(
        identity_receipt.get("zero_field_exact_identity", {}).values()
    ):
        raise RuntimeError("static/zero-field identity controls have not passed")
    static_clu = np.load(identity_root / "static_clu.npy")
    if static_clu.shape != pre.shape:
        raise RuntimeError("static replay labels do not align with bounded events")

    family_rows = _csv_rows(FAMILY_ROOT / "family_members_waveform_only.csv")
    family_by_cluster = {int(r["cid"]): r["family_id"] for r in family_rows}
    plausibility_rows = _csv_rows(PLAUSIBILITY_ROOT / "family_plausibility_audit.csv")
    family_class = {r["family_id"]: r["depth_time_class"] for r in plausibility_rows}
    pair_rows = _csv_rows(STATE_OUT / "state_pair_scores.csv")
    aggregate_rows = _csv_rows(STATE_OUT / "cluster_pair_aggregates.csv")
    candidates = [r for r in aggregate_rows if r["accepted"].lower() == "true"]
    ops = np.load(DEFAULT_SORT / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    decisions = []
    for candidate in candidates:
        a, b = int(candidate["cluster_a"]), int(candidate["cluster_b"])
        ma, mb = mapping[a], mapping[b]
        mapping_pass = ma.purity >= MAPPING_PURITY_THRESHOLD and mb.purity >= MAPPING_PURITY_THRESHOLD
        identity_pass, identity_reason = family_support(
            ma.final_cluster, mb.final_cluster, family_by_cluster, family_class
        )
        reciprocal = ast.literal_eval(candidate["reciprocal_pairs"])
        rival_links, rival_pass = [], True
        for sa, sb in reciprocal:
            intended = next(
                float(r["score"]) for r in pair_rows
                if int(r["cluster_a"]) == a and int(r["cluster_b"]) == b
                and int(r["state_a"]) == sa and int(r["state_b"]) == sb
            )
            ra = endpoint_rival_score(pair_rows, a, sa, b, sb)
            rb = endpoint_rival_score(pair_rows, b, sb, a, sa)
            margin_a = intended - ra[0] if np.isfinite(ra[0]) else np.inf
            margin_b = intended - rb[0] if np.isfinite(rb[0]) else np.inf
            passed = min(margin_a, margin_b) >= RIVAL_MARGIN
            rival_pass &= passed
            rival_links.append({
                "state_a": sa, "state_b": sb, "intended_score": intended,
                "rival_a_score": ra[0], "rival_a_cluster": ra[1], "rival_a_state": ra[2],
                "rival_b_score": rb[0], "rival_b_cluster": rb[1], "rival_b_state": rb[2],
                "margin_a": margin_a, "margin_b": margin_b, "passed": bool(passed),
            })
        ccg = _ccg(st[pre == a, 0] / fs, st[pre == b, 0] / fs, ops)
        substantive_pass = bool(mapping_pass and identity_pass and rival_pass and ccg["passed"])
        decisions.append({
            "cluster_a": a, "cluster_b": b,
            "final_cluster_a": ma.final_cluster, "final_cluster_b": mb.final_cluster,
            "mapping_purity_a": ma.purity, "mapping_purity_b": mb.purity,
            "mapping_pass": bool(mapping_pass),
            "independent_identity_pass": bool(identity_pass),
            "independent_identity_reason": identity_reason,
            "rival_pass": bool(rival_pass), "rival_links": rival_links,
            "ccg": ccg, "substantive_gates_pass": substantive_pass,
            "accepted_merge": False,
            "decision": "rejected" if not substantive_pass else "held_protocol_not_preregistered",
        })

    # No accepted motion-specific v1 edge. Preserve all ordinary Kilosort
    # merges from the passed static baseline rather than reverting to premerge.
    labels = np.unique(pre)
    np.save(partial / "motion_aware_clu.npy", static_clu)
    with (partial / "label_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["original_cluster", "final_cluster", "changed"])
        writer.writeheader()
        rows = []
        for c in labels:
            mapped = np.unique(static_clu[pre == c])
            if mapped.size != 1:
                raise RuntimeError("static replay split an original cluster")
            rows.append({
                "original_cluster": int(c), "final_cluster": int(mapped[0]),
                "changed": bool(mapped[0] != c),
            })
        writer.writerows(rows)
    (partial / "edge_decisions.json").write_text(json.dumps(decisions, indent=2, allow_nan=False) + "\n")
    receipt = {
        "schema": "medicine-aware-final-gates-v1", "complete": True,
        "state_pass_candidates": len(candidates),
        "substantive_gate_passes": sum(d["substantive_gates_pass"] for d in decisions),
        "accepted_motion_specific_merges": 0,
        "static_baseline_merges": int(identity_receipt["static_merge_count"]),
        "output_clusters": int(np.unique(static_clu).size),
        "labels_changed_vs_static": False,
        "labels_changed_vs_premerge": bool(not np.array_equal(static_clu, pre)),
        "scientific_status": "completed_bounded_v1_no_merge_authorized",
        "settings": settings,
    }
    (partial / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(partial, OUT)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
