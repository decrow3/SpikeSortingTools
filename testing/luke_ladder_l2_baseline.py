"""L2 baseline — rescue vs legacy-style KS4 on every development snippet.

The frozen panel's development half, scored under the two named sorter configs
(`ladder_sorter.RESCUE`, `LEGACY_STYLE`), with:

* the `score_sort` guardrails and context for each config;
* the **snippet-scale** symmetric agreement between the two — the apples-to-apples
  comparator the Checkpoint B.5 caveat identified (a snippet sort must be
  compared to a *same-length* sort, not the full-session one);
* wall clock per config.

This is the baseline Phase D candidates are measured against on the real-data
and guardrail axes. The injected-truth baseline (Checkpoint C) is separate —
`luke_ladder_panel_baseline.py`.

    python testing/luke_ladder_l2_baseline.py

Outputs to testing/outputs/luke_ladder_l2_baseline/. Uses the L1 cache, so
re-runs are cheap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from testing.ladder_l1 import l1_run
from testing.ladder_score import symmetric_agreement
from testing.ladder_snippets import load_snippet, snippet_root
from testing.ladder_sorter import LEGACY_STYLE, RESCUE
from testing.luke_rescue_unique_units_audit import load_sort

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_ladder_l2_baseline"
PANEL_MANIFEST = "panel_manifest.json"


def _dev_snippets() -> list[Path]:
    root = snippet_root()
    panel = json.loads((root / PANEL_MANIFEST).read_text())
    return [
        root / s["directory"]
        for s in panel["snippets"]
        if s["split"] == "development"
    ]


def _config_row(result: dict, config_label: str) -> dict:
    s = result["score"]
    g = s["guardrails"]
    return {
        "sorter": config_label,
        "ks_good": s["context"]["ks_good_count"],
        "n_clusters": s["context"]["n_clusters"],
        "total_spikes": s["context"]["total_spikes"],
        "similar_good_pairs": g["similar_good_good_pairs"],
        "similar_pairs_per_good": round(g["similar_pairs_per_good_unit"], 4),
        "rv_median": round(g["refractory_violation_median"], 5),
        "rv_frac_over_1pct": round(g["refractory_violation_frac_over_1pct"], 4),
        "edge_spike_frac_40um": round(g["edge_spike_fraction_40um"], 4),
        "pipeline_s": round(result["wall_clock"]["pipeline_s"], 1),
        "sort_cached": result["wall_clock"]["sort_was_cached"],
    }


def run() -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for snip_dir in _dev_snippets():
        snip = load_snippet(snip_dir)
        axes = snip.manifest.get("axes", {})

        rescue_res = l1_run(snip_dir, sorter=RESCUE)
        legacy_res = l1_run(snip_dir, sorter=LEGACY_STYLE)

        rescue_cur = Path(rescue_res["score"]["sorter_output"])
        legacy_cur = Path(legacy_res["score"]["sorter_output"])
        agree = symmetric_agreement(load_sort(rescue_cur), load_sort(legacy_cur))

        for res, label in ((rescue_res, "rescue"), (legacy_res, "legacy_style")):
            rows.append({
                "snippet": snip.manifest["name"],
                "regime": axes.get("motion_regime"),
                "strip": axes.get("depth_strip"),
                "snr": axes.get("snr"),
                "artifact": axes.get("artifact_proximity"),
                **_config_row(res, label),
                # rescue-vs-legacy agreement is symmetric; store it on both rows
                "rescue_vs_legacy_matched": agree["matched_good_pairs"],
                "rescue_only_good": agree["gained_good"],
                "legacy_only_good": agree["lost_good"],
                "legacy_only_absent_in_rescue": agree["lost_absent_at_detection"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "l2_baseline.csv", index=False)

    summary = {
        "n_dev_snippets": df["snippet"].nunique(),
        "median_ks_good": {
            s: int(sub["ks_good"].median())
            for s, sub in df.groupby("sorter")
        },
        "median_similar_pairs_per_good": {
            s: round(float(sub["similar_pairs_per_good"].median()), 4)
            for s, sub in df.groupby("sorter")
        },
        "median_rv_median": {
            s: round(float(sub["rv_median"].median()), 5)
            for s, sub in df.groupby("sorter")
        },
        "median_edge_spike_frac": {
            s: round(float(sub["edge_spike_frac_40um"].median()), 4)
            for s, sub in df.groupby("sorter")
        },
        "median_pipeline_s": {
            s: round(float(sub["pipeline_s"].median()), 1)
            for s, sub in df.groupby("sorter")
        },
        "rescue_vs_legacy_good_agreement": {
            "median_matched": int(df["rescue_vs_legacy_matched"].median()),
            "median_rescue_only": int(df["rescue_only_good"].median()),
            "median_legacy_only": int(df["legacy_only_good"].median()),
            "total_legacy_only_absent_in_rescue": int(
                df.drop_duplicates("snippet")["legacy_only_absent_in_rescue"].sum()
            ),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return df


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n", 1)[0]).parse_args()
    df = run()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False))
    print(f"\nwrote {OUTPUT}")
    print(json.dumps(json.loads((OUTPUT / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
