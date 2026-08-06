#!/usr/bin/env python3
"""
compare_sortings_after_qc.py

:contentReference[oaicite:9]{index=9}ls pipeline outputs (post-curation) using
spikeinterface.comparison.compare_two_sorters().

Typical usage:
  python compare_sortings_after:contentReference[oaicite:10]{index=10}/pipeline_run_A \
    --pipe2 /path/to/pipeline_run_B \
    --outdir /path/to/compare_out \
    --delta-ms 0.4 \
    --min-agreement 0.5

By default it loads:
  <pipeX>/cur/cur_sorter_output   (Phy fold:contentReference[oaicite:11]{index=11}od
will attempt to keep only units labeled "good" in cluster info (if present).
QC-based filtering is intentionally left as a hook because your QC code
computes QC artifacts but does not define a single "good unit" threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

import spikeinterface.extractors as se
import spikeinterface.comparison as scmp
import spikeinterface.widgets as sw


def read_phy_sorting(phy_folder: Path):
    if not phy_folder.exists():
        raise FileNotFoundError(f"Phy folder not found: {phy_folder}")
    # spikeinterface.extractors.read_phy is imported/used in your repo curation code
    # so this matches your existing workflow. :contentReference[oaicite:12]{index=12}
    return se.read_phy(folder_path=str(phy_folder), load_all_cluster_properties=True)


def maybe_filter_good_units_from_phy(sorting):
    """
    If Phy cluster labels exist, keep units labeled 'good'.

    This is conservative: if no labels/properties are found, it returns sorting unchanged.
    """
    props = sorting.get_property_keys()
    label_key = None
    for k in ["KSLabel", "group", "quality", "cluster_group"]:
        if k in props:
            label_key = k
            break

    if la:contentReference[oaicite:13]{index=13}rn sorting

    labels = sorting.get_property(label_key)
    if labels is None:
        return sorting

    # normalize to lowercase strings
    labels_norm = [str(x).strip().lower() for x in labels]
    unit_ids = sorting.unit_ids
    good_ids = [uid for uid, lab in zip(unit_ids, labels_norm) if lab == "good"]

    if len(good_ids) == 0:
        return sorting

    return sorting.select_units(good_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe1", type=str, required=True, help="Pipeline directory A")
    ap.add_argument("--pipe2", type=str, required=True, help="Pipeline directory B")
    ap.add_argument("--stage", type=str, default="cur", choices=["cur", "sorter_output"],
                    help="Which folder to compare: curated output or raw sorter output.")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for reports")
    ap.add_argument("--delta-ms", type=float, default=0.4, help="Spike matching tolerance in ms")
    ap.add_argument("--min-agreement", type=float, default=0.5, help="Threshold for 'matched' summary table")
    ap.add_argument("--units", type=str, default="all", choices=["all", "good"],
                    help="If 'good', filters by Phy cluster labels when available.")
    args = ap.parse_args()

    pipe1 = Path(args.pipe1)
    pipe2 = Path(args.pipe2)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.stage == "cur":
        phy1 = pipe1 / "cur" / "cur_sorter_output"
        phy2 = pipe2 / "cur" / "cur_sorter_output"
    else:
        phy1 = pipe1 / "sorting" / "sorter_output"
        phy2 = pipe2 / "sorting" / "sorter_output"

    sorting1 = read_phy_sorting(phy1)
    sorting2 = read_phy_sorting(phy2)

    if args.units == "good":
        sorting1 = maybe_filter_good_units_from_phy(sorting1)
        sorting2 = maybe_filter_good_units_from_phy(sorting2)

    # Compare two sorters (symmetric comparison). :contentReference[oaicite:14]{index=14}
    cmp = scmp.compare_two_sorters(
        sorting1=sorting1,
        sorting2=sorting2,
        sorting1_name=pipe1.name,
        sorting2_name=pipe2.name,
        delta_time=args.delta_ms / 1000.0,
    )

    # Export key matrices
    match_event_count = cmp.match_event_count.copy()
    agreement_scores = cmp.agreement_scores.copy()

    # Convert to CSV
    match_event_count.to_csv(outdir / "match_event_count.csv")
    agreement_scores.to_csv(outdir / "agreement_scores.csv")

    # Matching (Hungarian; unmatched show as -1). :contentReference[oaicite:15]{index=15}
    m1_to_2, m2_to_1 = cmp.get_matching()

    # Build a tidy table of matched pairs with their agreement
    rows = []
    for u1, u2 in m1_to_2.items():
        if u2 == -1:
            rows.append({"unit_1": u1, "unit_2": -1, "agreement": 0.0})
        else:
            # agreement_scores is a dataframe indexed by unit ids
            agr = float(agreement_scores.loc[u1, u2])
            rows.append({"unit_1": u1, "unit_2": u2, "agreement": agr})
    df_match = pd.DataFrame(rows)
    df_match.to_csv(outdir / "matched_units.csv", index=False)

    # Summary JSON
    summary = {
        "pipe1": str(pipe1),
        "pipe2": str(pipe2),
        "stage": args.stage,
        "delta_ms": args.delta_ms,
        "units_filter": args.units,
        "n_units_1": int(len(sorting1.unit_ids)),
        "n_units_2": int(len(sorting2.unit_ids)),
        "n_matched": int((df_match["unit_2"] != -1).sum()),
        "n_unmatched_1": int((df_match["unit_2"] == -1).sum()),
        "matched_ge_min_agreement": int(((df_match["unit_2"] != -1) & (df_match["agreement"] >= args.min_agreement)).sum()),
        "min_agreement_threshold": args.min_agreement,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plots
    # Agreement matrix visualization is the canonical first look. :contentReference[oaicite:16]{index=16}
    fig1 = sw.plot_agreement_matrix(cmp, ordered=True)
    fig1.figure.savefig(outdir / "agreement_matrix_ordered.png", dpi=200, bbox_inches="tight")

    fig2 = sw.plot_agreement_matrix(cmp, ordered=False)
    fig2.figure.savefig(outdir / "agreement_matrix.png", dpi=200, bbox_inches="tight")

    # Also save the "high agreement" subset for quick eyeballing
    df_high = df_match[(df_match["unit_2"] != -1) & (df_match["agreement"] >= args.min_agreement)].sort_values("agreement", ascending=False)
    df_high.to_csv(outdir / f"matched_units_agreement_ge_{args.min_agreement:.2f}.csv", index=False)

    print("Wrote outputs to:", outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


