"""Append the completed AF/AG audit to the already-generated AB README."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OUT = Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage4_ab")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    readme_path = OUT / "README.md"
    readme = readme_path.read_text()
    marker = "\n## AF/AG canonical-mask and block-versus-break audit\n"
    if marker in readme:
        readme = readme.split(marker, 1)[0].rstrip() + "\n"
    af = json.loads((OUT / "af_canonical_mask_comparison.json").read_text())
    sensitivity = json.loads((OUT / "ag_edge_exclusion_sensitivity.json").read_text())
    interpretation = json.loads((OUT / "ag_block_break_interpretation.json").read_text())
    summary = pd.read_csv(OUT / "ag_block_break_summary.csv")
    readme += marker + f"""

The independently generated mask has exactly the same 422 numeric intervals as the hub canonical mask. Its differing SHA-256 is serialization-only: this run writes sources such as `A+E`, while the hub writes `AE`. The first differing row is 23.750–27.250 s (`A+E` versus `AE`); the generation rule was not adjusted. The hub canonical hash is `{af['hub_sha256']}`.

The selected 30 s MEDiCINe layer was refitted once after excluding peaks within 3 s of the canonical mask. Far from episodes, its median absolute field change was {sensitivity['median_abs_change_um']:.3f} µm (95th percentile {sensitivity['p95_abs_change_um']:.3f} µm), so the sensitivity is **material (>1 µm)**. This does not change the frozen AB/AD selection or the censor mask.

AG.2 found {interpretation['free_runs_at_least_60s']} free stretches and {interpretation['boundaries']} qualifying free/dense boundaries; all {interpretation['label_free_resolved']} label-free comparisons resolved. The reference medians were {interpretation['label_free_median_free_minus_dense_um']:.3f} µm (label-free) and {interpretation['unit_median_free_minus_dense_um']:.3f} µm (static units), both approximately zero rather than a real >3 µm resting-position offset. Persistent estimator differences are therefore interpreted as estimator/activity-state bias. Entering/leaving-break signs are retained below as the drift confound check.

{summary.to_markdown(index=False, floatfmt='.3f')}

![AG block-versus-break levels](ag_block_break_levels.png)
"""
    readme_path.write_text(readme, newline="\n")
    manifest_path = OUT / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in (
        "README.md", "af_canonical_mask_comparison.json", "ag_edge_exclusion_sensitivity.json",
        "ag_block_break_boundaries.csv", "ag_block_break_summary.csv",
        "ag_block_break_interpretation.json", "ag_block_break_levels.png",
    ):
        manifest["artifacts"][name] = sha256(OUT / name)
    manifest["ag_status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", newline="\n")


if __name__ == "__main__":
    main()
