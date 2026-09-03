"""Does full-session family stitching reconstitute corrected unmatched units?

`docs/luke_20250804_family_stitch_candidate.md`: post-sort family stitching is
safe and helps mild drift at snippet scale, but the C2 fragments at 120 s are
contamination-dominated so it does not close the drift penalty there. A2's
The original 2-recovered/4-destroyed result is retracted: it inherited the
non-exclusive identity matcher and invalid 127-unit cohort. This v2 script
recomputes both before and after cohorts with exclusive event matching.

This is a pure post-processing evaluation on the cached full-session sorts. No
sorter runs, nothing is written under /mnt, and the 28 GB `cur_output` is not
copied -- the stitch remap is applied to the spike-cluster vector in memory and
the reconstitution is measured with the same `mutual_best_matches` machinery
that defined the 127 in the first place.

    python testing/luke_rescue_stitch_fullsession_eval.py

Outputs to testing/outputs/luke_rescue_stitch_fullsession_eval_v2/ (untracked).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_stitch import StitchConfig, stitch_families
from testing.luke_rescue_lost_units_audit import (
    DOMINANT_FRAC,
    classify,
    dominant_partner_counts,
)
from testing.luke_rescue_unique_units_audit import (
    DURATION_S,
    FS,
    LEGACY,
    REFRACTORY_MS,
    RESCUE,
    load_sort,
    mutual_best_matches,
    spatial_null_distribution,
    template_depth_by_cluster,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_stitch_fullsession_eval_v2"


def _apply_remap(sort: dict, families: list[list[int]]) -> tuple[dict, dict]:
    """Relabel each family to its largest good member; return (new sort, remap)."""
    counts = pd.Series(sort["cl"]).value_counts()
    good_set = set(sort["good"])
    remap: dict[int, int] = {}
    for family in families:
        good_members = [u for u in family if u in good_set] or family
        keep = max(good_members, key=lambda u: int(counts.get(u, 0)))
        for u in family:
            if u != keep:
                remap[u] = keep

    new_cl = np.array([remap.get(int(c), int(c)) for c in sort["cl"]], dtype=np.int64)
    order = np.argsort(sort["st"], kind="stable")
    absorbed = set(remap)
    new_label = {c: v for c, v in sort["label"].items() if c not in absorbed}
    new = {
        "st": sort["st"][order],
        "cl": new_cl[order],
        "label": new_label,
        "good": {c for c, v in new_label.items() if v == "good"},
    }
    return new, remap


def _reconstituted(
    lost: list[int],
    legacy: dict,
    rescue_after: dict,
    legacy_depth: dict[int, float],
    rescue_depth: dict[int, float],
) -> pd.DataFrame:
    """For each lost legacy good unit, does a stitched rescue good unit now own it?"""
    merge_targets = dominant_partner_counts(
        legacy, rescue_after, lost, legacy_depth, rescue_depth
    )
    rows = []
    for cid in lost:
        a = legacy["st"][legacy["cl"] == cid]
        n = int(a.size)
        frac, ranked, evidence = spatial_null_distribution(
            a,
            legacy_depth.get(int(cid), np.nan),
            rescue_after,
            rescue_depth,
        )
        cls = classify(
            frac,
            ranked,
            merge_targets,
            shared_detection_supported=evidence["shared_detection_supported"],
        )
        top_cl, p1, top_label = (ranked[0] if ranked else (-1, 0.0, "none"))
        isi = np.diff(np.sort(a)) / FS * 1000.0
        rv = float((isi < REFRACTORY_MS).mean()) if n > 1 else np.nan
        rows.append({
            "legacy_cluster": int(cid),
            "n_spikes": n,
            "rate_hz": n / DURATION_S,
            "rv_frac": rv,
            "frac_found": round(frac, 3),
            "null_median_fraction": evidence["null_median_fraction"],
            "coincidence_excess": evidence["coincidence_excess"],
            "shared_detection_supported": evidence["shared_detection_supported"],
            "best_rescue_cluster": int(top_cl),
            "best_rescue_label": top_label,
            "best_rescue_frac": round(p1, 3),
            "classification_after": cls,
            "owned_by_good": bool(top_label == "good" and p1 >= DOMINANT_FRAC),
        })
    return pd.DataFrame(rows)


def run(config: StitchConfig | None = None) -> dict:
    config = config or StitchConfig()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    legacy = load_sort(LEGACY)
    rescue = load_sort(RESCUE)
    legacy_depth = template_depth_by_cluster(LEGACY)
    rescue_depth = template_depth_by_cluster(RESCUE)

    before = mutual_best_matches(rescue, legacy)
    matched_before = set(before.legacy_cluster)
    lost = sorted(legacy["good"] - matched_before)

    fam = stitch_families(RESCUE, config)
    families = fam["families"]
    rescue_after, remap = _apply_remap(rescue, families)

    after = mutual_best_matches(rescue_after, legacy)
    matched_after = set(after.legacy_cluster)

    newly_matched = sorted(matched_after - matched_before)
    lost_by_stitch = sorted(matched_before - matched_after)

    recon = _reconstituted(
        lost, legacy, rescue_after, legacy_depth, rescue_depth
    )
    recon.sort_values("n_spikes", ascending=False).to_csv(
        OUTPUT / "lost_units_after_stitch.csv", index=False
    )

    owned = recon[recon.owned_by_good]
    clean_owned = owned[owned.rv_frac <= 0.01]

    summary = {
        "stitch_config": fam["config"],
        "config_digest": fam["config_digest"],
        "n_good_before": fam["n_good_before"],
        "n_good_after": fam["n_good_after"],
        "n_families": fam["n_families"],
        "n_units_absorbed": fam["n_units_absorbed"],
        "n_good_absorbed": fam["n_good_absorbed"],
        "n_stitch_edges": fam["n_stitch_edges"],
        "legacy_good": len(legacy["good"]),
        "matched_before": len(matched_before),
        "lost_before": len(lost),
        "matched_after": len(matched_after),
        "newly_matched_from_baseline_unmatched": sorted(
            c for c in newly_matched if c in set(lost)
        ),
        "n_newly_matched_from_baseline_unmatched": sum(
            1 for c in newly_matched if c in set(lost)
        ),
        "n_matches_lost_to_overmerge": len(lost_by_stitch),
        "matches_lost_to_overmerge": lost_by_stitch,
        "reconstituted_owned_by_good_unit": int(owned.shape[0]),
        "reconstituted_and_refractory_clean": int(clean_owned.shape[0]),
        "classification_after_breakdown": recon.classification_after.value_counts().to_dict(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n", 1)[0]).parse_args()
    summary = run()
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
