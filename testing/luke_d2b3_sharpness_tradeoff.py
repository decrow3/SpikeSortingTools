"""D2b-3 — rerun interpolation tradeoffs with geometry-consistent motion.

The original D2b-3 result is retracted. The D2b plan asks whether the
motion-benefit / interpolation-cost tradeoff depends on waveform class. D2b-2's
real donors cannot answer it (they thin out above ~250 µV and span a narrow
sharpness range), so this uses the D2b-3 **synthetic** grid (sharpness ×
amplitude × polarity) plus a couple of real anchors.

For every donor it injects the same train static and along two known
trajectories (40 µm rigid, 20 µm/40 s oscillation) and sorts:

* `rescue`  static + moving          -> **uncorrected drift penalty** (the cost of motion)
* `oracle`  static + moving          -> candidate residual under exact correction

The decisive per-class number is
`interpolation_cost = static_accuracy(rescue) - moving_accuracy(oracle)` at
40 µm: how much exact correction still loses, if any, by waveform class.
If sharp / high-SNR classes have a larger interpolation cost, they belong on the
"correct less" side of a selective policy.

    python testing/luke_d2b3_sharpness_tradeoff.py

Diagnostic. Outputs to testing/outputs/luke_d2b3_sharpness_tradeoff_v2/. Nothing
under /mnt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_donors import _taper
from testing.ladder_inject import write_injected_recording
from testing.ladder_l1 import l1_run
from testing.ladder_motion import (
    observed_waveform,
    oracle_corrected_recording,
    paired_geometry_motion_injection,
    waveform_preservation,
)
from testing.ladder_synthetic_donors import default_grid, synthetic_template
from testing.luke_injected_ground_truth_benchmark import validate_template
from testing.luke_rescue_c2_drift_challenge import (
    PRESPEC as C2_PRESPEC,
    _train,
    _trajectory_fn,
    load_background,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_d2b3_sharpness_tradeoff_v2"
REAL_COHORT = REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort/donor_templates.npz"

PRESPEC = {
    "schema": "luke-d2b3-sharpness-tradeoff-v4",
    "frozen": "2026-09-03",
    "status": "corrected_geometry_aware_rerun_pending",
    "question": (
        "Does the motion-benefit / interpolation-cost tradeoff depend on "
        "waveform sharpness and SNR? First pass: interpolation cost tracks "
        "amplitude/SNR, not the synthetic sharpness knob — pin the SNR crossover."
    ),
    "trajectories": ["rigid_40um"],
    "real_anchors": ["D08", "D04", "D06", "D01", "D14", "D02"],
    "metric": "interpolation_cost = static_acc(rescue) - moving_acc(oracle) at 40um rigid",
}


def _freeze_prespec() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p = OUTPUT / "prespec.json"
    if p.exists():
        if json.loads(p.read_text()) != PRESPEC:
            raise SystemExit(f"{p} differs from frozen PRESPEC; delete the tree to re-freeze.")
    else:
        p.write_text(json.dumps(PRESPEC, indent=2) + "\n")


def _cohort(fs: float) -> dict[str, dict]:
    """{template_id: {template, meta}} for the synthetic grid + real anchors."""
    out: dict[str, dict] = {}
    for i, spec in enumerate(default_grid(), 1):
        out[f"S{i:02d}"] = {
            "template": synthetic_template(spec, fs=fs),
            "meta": {
                "source": "synthetic", "peak_uv": spec.peak_uv,
                "trough_width_ms": spec.trough_width_ms,
                "spatial_lambda_um": spec.spatial_lambda_um, "polarity": spec.polarity,
            },
        }
    real = np.load(REAL_COHORT)
    man = pd.read_csv(
        REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort/donor_manifest.csv"
    ).set_index("template_id")
    for tid in PRESPEC["real_anchors"]:
        if tid not in real.files:
            continue
        out[tid] = {
            "template": validate_template(np.asarray(real[tid], dtype=np.float32), edge_guard_samples=3),
            "meta": {
                "source": "real", "peak_uv": float(man.loc[tid, "peak_uv"]),
                "trough_width_ms": np.nan,
                "spatial_lambda_um": np.nan, "polarity": man.loc[tid, "polarity"],
            },
        }
    return out


def run(only: list[str] | None = None) -> pd.DataFrame:
    _freeze_prespec()
    bg_uv, geometry, fs, gain, _ = load_background()
    duration_s = bg_uv.shape[0] / fs
    train = _train(C2_PRESPEC["background"]["duration_s"], fs)
    truth = {"inj0": train}
    n_bg_ch = bg_uv.shape[1]
    cohort = _cohort(fs)
    ids = only or list(cohort)

    rows = []
    for tid in ids:
        template = cohort[tid]["template"]
        meta = cohort[tid]["meta"]
        base_channel = n_bg_ch // 2 - template.shape[1] // 2

        # build the static + each moving recording once
        recs: dict[str, Path] = {}
        for traj_name in ["static", *PRESPEC["trajectories"]]:
            if traj_name == "static":
                traj_fn = None
            else:
                traj_fn, _ = _trajectory_fn(traj_name, geometry, duration_s)
            static_uv, moving_uv, _ = paired_geometry_motion_injection(
                bg_uv, template, train, fs=fs, base_channel=base_channel,
                moving_trajectory=traj_fn or (lambda t: np.zeros_like(np.asarray(t, float))),
                unit_id="inj0", edge_guard_samples=3,
                channel_positions=geometry,
            )
            arm_uv = static_uv if traj_name == "static" else moving_uv
            rec_dir = OUTPUT / "runs" / f"{tid}_{traj_name}"
            write_injected_recording(
                rec_dir, arm_uv, channel_positions=geometry, fs=fs,
                gain_uv_per_count=gain, name=f"d2b3_{tid}_{traj_name}",
            )
            recs[traj_name] = rec_dir

        # rescue scores
        scores = {
            t: l1_run(recs[t], truth=truth, out_root=OUTPUT / "_l1")["score"]["primary"]["units"][0]
            for t in recs
        }
        static_acc = scores["static"]["accuracy"]

        for traj_name in PRESPEC["trajectories"]:
            traj_fn, _ = _trajectory_fn(traj_name, geometry, duration_s)
            corrected = OUTPUT / "corrected" / f"{tid}_{traj_name}"
            oracle_corrected_recording(
                recs[traj_name], corrected, trajectory_fn=traj_fn, duration_s=duration_s,
                fs=fs, gain_uv_per_count=gain, name=f"d2b3_oracle_{tid}_{traj_name}",
            )
            o = l1_run(corrected, truth=truth, out_root=OUTPUT / "_l1")["score"]["primary"]["units"][0]
            wf = waveform_preservation(
                observed_waveform(
                    corrected, train, base_channel=base_channel,
                    width=template.shape[1], n_samples=template.shape[0],
                ),
                template,
            )
            u_resc = scores[traj_name]
            rows.append({
                "donor": tid, **meta, "trajectory": traj_name,
                "static_acc": round(static_acc, 3),
                "moving_acc_rescue": round(u_resc["accuracy"], 3),
                "moving_acc_oracle": round(o["accuracy"], 3),
                "uncorrected_penalty": round(u_resc["accuracy"] - static_acc, 3),
                "oracle_penalty": round(o["accuracy"] - static_acc, 3),
                "interpolation_cost": round(static_acc - o["accuracy"], 3),
                "oracle_recovers": round(o["accuracy"] - u_resc["accuracy"], 3),
                "waveform_cosine": wf["waveform_cosine"],
                "oracle_fp": o["fp"], "rescue_fp": u_resc["fp"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "sharpness_tradeoff.csv", index=False)

    rigid = df[df.trajectory == "rigid_40um"].copy()
    syn = rigid[rigid.source == "synthetic"]
    summary = {
        "prespec_schema": PRESPEC["schema"],
        "n_donors": int(df.donor.nunique()),
        "rigid_40um": {
            "interpolation_cost_by_trough_width": {
                str(w): round(float(g.interpolation_cost.median()), 3)
                for w, g in syn.groupby("trough_width_ms")
            },
            "interpolation_cost_by_amplitude": {
                str(a): round(float(g.interpolation_cost.median()), 3)
                for a, g in syn.groupby("peak_uv")
            },
            "oracle_recovers_median": round(float(rigid.oracle_recovers.median()), 3),
            "donors_oracle_hurts": rigid.loc[rigid.oracle_recovers < -0.02, "donor"].tolist(),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args()
    df = run(args.only)
    pd.set_option("display.width", 240)
    print(df.to_string(index=False))
    print(json.dumps(json.loads((OUTPUT / "summary.json").read_text()), indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
