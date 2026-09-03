"""D2b-3 addendum — test interpolation-kernel sensitivity after corrected C2.

The original sharpness/SNR and kernel conclusions are retracted. This rerun
compares `InterpolateMotionRecording` kernels without presuming that a residual
exists or is intrinsic.

Uses the versioned, geometry-aware D2b-3 recordings and redoes the
oracle-corrected 40 µm arm per kernel.

    python testing/luke_d2b3_interp_kernel.py

Diagnostic. Outputs to testing/outputs/luke_d2b3_interp_kernel/. Nothing under /mnt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_inject import write_injected_recording
from testing.ladder_l1 import l1_run
from testing.ladder_motion import oracle_corrected_recording, paired_geometry_motion_injection
from testing.luke_injected_ground_truth_benchmark import validate_template
from testing.luke_rescue_c2_drift_challenge import (
    PRESPEC as C2_PRESPEC,
    _train,
    _trajectory_fn,
    load_background,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_d2b3_interp_kernel_v2"
D2B3 = REPO_ROOT / "testing/outputs/luke_d2b3_sharpness_tradeoff_v2"
REAL_COHORT = REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort/donor_templates.npz"

DONORS = ["D02", "D14", "D04", "D06"]
KERNELS = [
    ("kriging", 20.0), ("kriging", 40.0), ("idw", 30.0),
]
TRAJ = "rigid_40um"


def run() -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bg_uv, geometry, fs, gain, _ = load_background()
    duration_s = bg_uv.shape[0] / fs
    train = _train(C2_PRESPEC["background"]["duration_s"], fs)
    truth = {"inj0": train}
    real = np.load(REAL_COHORT)
    man = pd.read_csv(
        REPO_ROOT / "testing/outputs/luke_d2b2_donor_cohort/donor_manifest.csv"
    ).set_index("template_id")
    traj_fn, _ = _trajectory_fn(TRAJ, geometry, duration_s)

    d2b3 = pd.read_csv(D2B3 / "sharpness_tradeoff.csv").set_index("donor")

    rows = []
    for tid in DONORS:
        template = validate_template(np.asarray(real[tid], dtype=np.float32), edge_guard_samples=3)
        base_channel = bg_uv.shape[1] // 2 - template.shape[1] // 2
        _, moving_uv, _ = paired_geometry_motion_injection(
            bg_uv, template, train, fs=fs, base_channel=base_channel,
            moving_trajectory=traj_fn, unit_id="inj0", edge_guard_samples=3,
            channel_positions=geometry,
        )
        rec_dir = OUTPUT / "runs" / f"{tid}_{TRAJ}"
        write_injected_recording(
            rec_dir, moving_uv, channel_positions=geometry, fs=fs,
            gain_uv_per_count=gain, name=f"d2b3k_{tid}_{TRAJ}",
        )
        static_acc = float(d2b3.loc[tid, "static_acc"])
        rescue_acc = float(d2b3.loc[tid, "moving_acc_rescue"])
        for method, sigma in KERNELS:
            corrected = OUTPUT / "corrected" / f"{tid}_{method}_{sigma:.0f}"
            oracle_corrected_recording(
                rec_dir, corrected, trajectory_fn=traj_fn, duration_s=duration_s,
                fs=fs, gain_uv_per_count=gain,
                spatial_interpolation_method=method, sigma_um=sigma,
                name=f"d2b3k_oracle_{tid}_{method}_{sigma:.0f}",
            )
            u = l1_run(corrected, truth=truth, out_root=OUTPUT / "_l1")["score"]["primary"]["units"][0]
            rows.append({
                "donor": tid, "peak_uv": float(man.loc[tid, "peak_uv"]),
                "polarity": man.loc[tid, "polarity"],
                "kernel": f"{method}_s{sigma:.0f}",
                "static_acc": round(static_acc, 3),
                "rescue_moving_acc": round(rescue_acc, 3),
                "oracle_moving_acc": round(u["accuracy"], 3),
                "interpolation_cost": round(static_acc - u["accuracy"], 3),
                "oracle_recovers": round(u["accuracy"] - rescue_acc, 3),
                "oracle_fp": u["fp"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "interp_kernel.csv", index=False)
    piv = df.pivot_table(index=["donor", "peak_uv"], columns="kernel", values="oracle_moving_acc")
    (OUTPUT / "summary.json").write_text(json.dumps({
        "best_kernel_per_donor": {
            d: df[df.donor == d].sort_values("oracle_moving_acc").iloc[-1]["kernel"]
            for d in DONORS
        },
        "oracle_moving_acc": json.loads(piv.to_json()),
    }, indent=2, default=str) + "\n")
    return df


def main() -> None:
    df = run()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(json.dumps(json.loads((OUTPUT / "summary.json").read_text()), indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
