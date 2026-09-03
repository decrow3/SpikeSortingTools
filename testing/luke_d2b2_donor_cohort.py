"""D2b-2 — build and characterise a spatially-compact imec0 donor cohort.

D2b-1 established the pilot donors (10 single reviewed-event snippets, 3 imec1
units, one common-mode dominated) are flat ±160 µm plateaus — not single-neuron
footprints — because a single-event snippet of an unfiltered recording is
dominated by LFP/common-mode. So they cannot anchor the waveform-preservation
guardrail and their drift-penalty magnitudes may not transfer to compact real
neurons.

This builds the replacement from **de-whitened Kilosort templates** of
well-isolated good units in the imec0 rescue sort — real neurons, real
amplitude-decay footprint, the probe the promotion question is about, no new
manual review.

    python testing/luke_d2b2_donor_cohort.py
    python testing/luke_d2b2_donor_cohort.py --compare   # + old-vs-new compactness

Outputs to testing/outputs/luke_d2b2_donor_cohort/. Nothing is written under /mnt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_donors import DonorConfig, _compactness, build_donor_cohort

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort"
LUKE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0")
SORT_DIR = LUKE / "cur/cur_output"
RECORDING_DIR = LUKE / "recording"
PILOT_DONORS = REPO_ROOT / "testing/outputs/luke_injected_ground_truth_pilot/donor_templates.npz"


def _pilot_compactness() -> pd.DataFrame:
    d = np.load(PILOT_DONORS)
    rows = []
    for tid in d.keys():
        c = _compactness(d[tid])
        rows.append({
            "cohort": "pilot_imec1", "template_id": tid,
            "energy_frac_pm3": c["energy_frac_pm3"],
            "half_energy_width_ch": c["half_energy_width_ch"],
            "peak_uv": round(float(np.max(np.abs(d[tid]))), 1),
        })
    return pd.DataFrame(rows)


def run(n_donors: int, compare: bool) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = build_donor_cohort(
        SORT_DIR, RECORDING_DIR, OUTPUT, n_donors=n_donors, config=DonorConfig()
    )
    print(json.dumps(result, indent=2, default=str))
    new = pd.read_csv(OUTPUT / "donor_manifest.csv")
    print("\n=== new imec0 cohort ===")
    print(new.to_string(index=False))

    if compare:
        new_c = new[["template_id", "energy_frac_pm3", "half_energy_width_ch", "peak_uv"]].copy()
        new_c.insert(0, "cohort", "imec0_dewht")
        comp = pd.concat([_pilot_compactness(), new_c], ignore_index=True)
        comp.to_csv(OUTPUT / "cohort_compactness_compare.csv", index=False)
        print("\n=== compactness: pilot (imec1 single-event) vs new (imec0 de-whitened) ===")
        print(comp.groupby("cohort")[["energy_frac_pm3", "half_energy_width_ch", "peak_uv"]]
              .agg(["median", "min", "max"]).to_string())
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--n-donors", type=int, default=12)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    run(args.n_donors, args.compare)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
