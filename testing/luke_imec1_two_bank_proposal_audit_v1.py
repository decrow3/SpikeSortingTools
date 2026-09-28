#!/usr/bin/env python
"""Cached proposal-quality audit for the two frozen imec1 seed banks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from testing.luke_imec1_sorterfree_phase_audit_v1 import lagged_overlap_cosine

ROOT = Path(__file__).resolve().parents[1]
BANKS = {
    "s036": ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3",
    "s300": ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300",
}
OUTPUT = ROOT / "testing/outputs/luke_imec1_two_bank_proposal_audit_v1"
CORE_COSINE = 0.86
MIN_CORE = 20
MIN_HALF_CORE = 5
OVERLAP_ENERGY = 0.50
SAMPLE_TOLERANCE = 1 / 29999.835983263598 + 1e-9


def p90_span(values: pd.Series) -> float:
    return float(values.quantile(.95) - values.quantile(.05)) if len(values) else np.nan


def load_bank(tag: str, path: Path):
    families = pd.read_csv(path / "families_after_depth_reveal.csv")
    seed = pd.read_csv(path / "seed_cluster_events.csv")
    heldout = pd.read_csv(path / "heldout_events.csv")
    with np.load(path / "family_templates.npz") as saved:
        templates = np.asarray(saved["waveforms"])
        geometry = np.asarray(saved["relative_geometry"])
    return {"tag": tag, "path": path, "families": families, "seed": seed,
            "heldout": heldout, "templates": templates, "geometry": geometry}


def core_profile(bank) -> pd.DataFrame:
    rows = []
    family_table = bank["families"].set_index("family_id")
    for family_id, events in bank["seed"].sort_values("event_index").groupby("family_id"):
        events = events.reset_index(drop=True)
        core = events[events.member_cosine >= CORE_COSINE]
        first = core.iloc[::2]; second = core.iloc[1::2]
        family = family_table.loc[family_id]
        rows.append({
            "bank": bank["tag"], "family_id": family_id, "phase_id": int(family.phase_id),
            "seed_events": len(events), "core_events": len(core), "core_fraction": len(core) / len(events),
            "core_first_half_events": len(first), "core_second_half_events": len(second),
            "all_member_depth_p90_span_um": p90_span(events.waveform_centroid_um),
            "core_depth_p90_span_um": p90_span(core.waveform_centroid_um),
            "core_first_depth_p90_span_um": p90_span(first.waveform_centroid_um),
            "core_second_depth_p90_span_um": p90_span(second.waveform_centroid_um),
            "median_member_cosine": float(family.median_member_cosine),
            "split_half_cosine": float(family.split_half_cosine),
            "strict_seed_recovered": int(family.strict_seed_recovered),
            "strict_seed_recovery_fraction": float(family.strict_seed_recovery_fraction),
            "global_rival_cosine_within_bank": float(family.global_rival_cosine),
            "seed_qualified_v3": bool(family.seed_qualified_v3),
        })
    out = pd.DataFrame(rows)
    out["core_has_independent_half_support"] = (out.core_events >= MIN_CORE) & (out.core_first_half_events >= MIN_HALF_CORE) & (out.core_second_half_events >= MIN_HALF_CORE)
    out["broad_proposal_compact_local_core"] = out.core_has_independent_half_support & (out.all_member_depth_p90_span_um > 300) & (out.core_depth_p90_span_um <= 120)
    out["broad_proposal_motion_scale_core"] = out.core_has_independent_half_support & (out.all_member_depth_p90_span_um > 300) & (out.core_depth_p90_span_um > 120) & (out.core_depth_p90_span_um <= 300)
    return out


def combined_similarity(banks):
    records = []
    templates = []
    for bank in banks.values():
        for i, row in bank["families"].iterrows():
            records.append({"bank": bank["tag"], "family_id": row.family_id, "phase_id": int(row.phase_id),
                            "selected": bool(row.selected_depth_blind), "seed_qualified_v3": bool(row.seed_qualified_v3)})
            templates.append(bank["templates"][i])
    inventory = pd.DataFrame(records)
    templates = np.asarray(templates)
    geometry = next(iter(banks.values()))["geometry"]
    n = len(inventory); similarity = np.eye(n); coverage = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score, support, _ = lagged_overlap_cosine(templates[i], inventory.phase_id[i], templates[j], inventory.phase_id[j], geometry)
            similarity[i, j] = similarity[j, i] = score
            coverage[i, j] = coverage[j, i] = support
    supported = coverage >= OVERLAP_ENERGY
    np.fill_diagonal(supported, False)
    scores = np.where(supported, similarity, -np.inf)
    nearest = np.argmax(scores, axis=1)
    audit = inventory.copy()
    audit["combined_rival_bank"] = inventory.bank.to_numpy()[nearest]
    audit["combined_rival_family_id"] = inventory.family_id.to_numpy()[nearest]
    audit["combined_rival_cosine"] = scores[np.arange(n), nearest]
    audit["combined_rival_overlap_energy"] = coverage[np.arange(n), nearest]
    audit["new_cross_bank_alias_ge_0p95"] = (audit.combined_rival_bank != audit.bank) & (audit.combined_rival_cosine >= .95)
    memberships = []
    safe = np.nan_to_num(similarity, nan=-1.0)
    for threshold in (.90, .95, .97):
        distance = squareform(np.clip(1 - safe, 0, 2), checks=False)
        labels = fcluster(linkage(distance, method="complete"), 1 - threshold, criterion="distance")
        memberships.extend({"threshold": threshold, "combined_group": int(label), **inventory.iloc[i].to_dict()} for i, label in enumerate(labels))
    return inventory, templates, similarity, coverage, audit, pd.DataFrame(memberships)


def exact_overlap(a: np.ndarray, b: np.ndarray) -> int:
    if not len(a) or not len(b): return 0
    a = np.asarray(a, dtype=float); b = np.sort(np.asarray(b, dtype=float)); i = np.searchsorted(b, a)
    return int((np.minimum(np.abs(a - b[np.clip(i - 1, 0, len(b) - 1)]), np.abs(a - b[np.clip(i, 0, len(b) - 1)])) <= SAMPLE_TOLERANCE).sum())


def accepted_duplicate_audit(banks, candidates: pd.DataFrame) -> pd.DataFrame:
    selected = candidates[candidates.seed_qualified_v3 | candidates.broad_proposal_compact_local_core | candidates.broad_proposal_motion_scale_core]
    rows = []
    for i, a in selected.reset_index(drop=True).iterrows():
        for _, b in selected.reset_index(drop=True).iloc[i + 1:].iterrows():
            ea = banks[a.bank]["heldout"]
            eb = banks[b.bank]["heldout"]
            ta = ea[(ea.family_id == a.family_id) & (ea.status == "strict")].time_s.to_numpy()
            tb = eb[(eb.family_id == b.family_id) & (eb.status == "strict")].time_s.to_numpy()
            rows.append({"bank_a": a.bank, "family_a": a.family_id, "bank_b": b.bank, "family_b": b.family_id,
                         "strict_a": len(ta), "strict_b": len(tb), "exact_a_to_b": exact_overlap(ta, tb),
                         "exact_b_to_a": exact_overlap(tb, ta)})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    banks = {tag: load_bank(tag, path) for tag, path in BANKS.items()}
    cores = pd.concat([core_profile(bank) for bank in banks.values()], ignore_index=True)
    cores.to_csv(OUTPUT / "proposal_core_audit.csv", index=False)
    inventory, templates, similarity, coverage, rivals, groups = combined_similarity(banks)
    rivals.to_csv(OUTPUT / "combined_bank_rivals.csv", index=False)
    groups.to_csv(OUTPUT / "combined_complete_link_memberships.csv", index=False)
    np.savez_compressed(OUTPUT / "combined_template_similarity.npz", templates=templates, similarity=similarity,
                        overlap_energy=coverage, bank=inventory.bank.to_numpy(), family_id=inventory.family_id.to_numpy())
    duplicates = accepted_duplicate_audit(banks, cores)
    duplicates.to_csv(OUTPUT / "qualified_and_core_exact_event_overlap.csv", index=False)
    failure = {}
    for tag, q in cores.groupby("bank"):
        failure[tag] = {
            "proposals": len(q),
            "fail_member_similarity": int((q.median_member_cosine < .8).sum()),
            "fail_strict_seed_recovery": int((~((q.strict_seed_recovered >= 5) & (q.strict_seed_recovery_fraction >= .2))).sum()),
            "fail_global_distinctness": int((q.global_rival_cosine_within_bank >= .95).sum()),
            "fail_split_half_repeatability": int((q.split_half_cosine < .9).sum()),
            "compact_local_cores_inside_broad_proposals": int(q.broad_proposal_compact_local_core.sum()),
            "motion_scale_cores_inside_broad_proposals": int(q.broad_proposal_motion_scale_core.sum()),
        }
    summary = {
        "schema": "luke0804-imec1-two-bank-proposal-audit-v1", "status": "complete",
        "motion_estimate_read": False, "raw_voltage_read": False, "sorter_input_read": False,
        "grain": "one phase-specific morphology proposal per seed bank; core is member_cosine>=0.86 and is diagnostic, not retrained",
        "failure_profile": failure,
        "combined_bank": {
            "templates": len(inventory),
            "selected_with_new_cross_bank_alias_ge_0p95": int((rivals.selected & rivals.new_cross_bank_alias_ge_0p95).sum()),
            "seed_qualified_with_new_cross_bank_alias_ge_0p95": int((rivals.seed_qualified_v3 & rivals.new_cross_bank_alias_ge_0p95).sum()),
            "complete_link_groups": {str(t): int(groups[groups.threshold == t].combined_group.nunique()) for t in (.90, .95, .97)},
        },
        "limitations": ["No seed waveforms were saved, so cores use cached member cosine/depth only.",
                        "Core events helped form the original template; this audit nominates a split experiment but does not validate improvement.",
                        "Combined rival scores are template-level; detections were not rescored against the combined bank."],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
