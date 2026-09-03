"""D2b-2 sanity — do the new compact donors recover cleanly when injected static?

A compact template is necessary but not sufficient: the donor must also sort
cleanly when injected stationary into real background (the C2 plan-C sanity
bar, accuracy >= 0.8), or it is not a usable ground-truth unit. This injects a
sample of the new imec0 cohort static into the C2 quiet background and scores it.

    python testing/luke_d2b2_donor_sanity.py --donors D01 D04 D08 D12

Reuses the C2 background window and the L1 runner. Diagnostic. Outputs to
testing/outputs/luke_d2b2_donor_sanity/. Nothing under /mnt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_inject import paired_injection, static_trajectory, write_injected_recording
from testing.ladder_l1 import l1_run
from testing.luke_injected_ground_truth_benchmark import validate_template
from testing.luke_rescue_c2_drift_challenge import _train, load_background

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_d2b2_donor_sanity"
COHORT = REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort/donor_templates.npz"
SANITY_ACC = 0.8


def run(donor_ids: list[str] | None = None) -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    donors = np.load(COHORT)
    donor_ids = donor_ids or list(donors)
    manifest = pd.read_csv(
        REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort/donor_manifest.csv"
    ).set_index("template_id")

    bg_uv, geometry, fs, gain, _ = load_background()
    duration_s = bg_uv.shape[0] / fs
    train = _train(duration_s, fs)
    n_bg_ch = bg_uv.shape[1]

    rows = []
    for tid in donor_ids:
        template = validate_template(np.asarray(donors[tid], dtype=np.float32), edge_guard_samples=3)
        base_channel = n_bg_ch // 2 - template.shape[1] // 2
        static_uv, _, truth = paired_injection(
            bg_uv, template, train, fs=fs, base_channel=base_channel,
            moving_trajectory=static_trajectory(), unit_id="inj0", edge_guard_samples=3,
        )
        rec_dir = OUTPUT / "runs" / tid
        write_injected_recording(
            rec_dir, static_uv, channel_positions=geometry, fs=fs,
            gain_uv_per_count=gain, name=f"d2b2_sanity_{tid}",
        )
        score = l1_run(rec_dir, truth=truth, out_root=OUTPUT / "_l1")["score"]
        u = score["primary"]["units"][0]
        rows.append({
            "donor": tid,
            "peak_uv": float(manifest.loc[tid, "peak_uv"]),
            "polarity": manifest.loc[tid, "polarity"],
            "energy_frac_pm3": float(manifest.loc[tid, "energy_frac_pm3"]),
            "accuracy": round(u["accuracy"], 3),
            "identities": u["n_output_units_capturing"],
            "tp": u["tp"], "fp": u["fp"], "fn": u["fn"],
            "passes_sanity": bool(u["accuracy"] >= SANITY_ACC),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "donor_sanity.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps({
        "n_tested": len(df),
        "n_pass": int(df.passes_sanity.sum()),
        "accuracy_median": round(float(df.accuracy.median()), 3),
        "fails": df.loc[~df.passes_sanity, "donor"].tolist(),
    }, indent=2) + "\n")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--donors", nargs="+")
    args = ap.parse_args()
    df = run(args.donors)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(json.dumps(json.loads((OUTPUT / "summary.json").read_text()), indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
