"""D2b-1 — how accurate must an estimated motion field be to still help?

`docs/pipeline_improvement_plan.md` Phase D / D2b-1. The original tolerance
envelope is retracted. This corrected rerun perturbs a geometry-consistent
oracle field and tests whether a defensible envelope can be established.

Cheap: reuses the corrected cached C2 injected recordings, no new injection. For each
(donor, trajectory) where the exact oracle beats rescue, and each perturbation
level, it builds the perturbed-oracle recording, runs L1 (frozen rescue sort),
scores against the injected train, and also measures waveform preservation on
the corrected voltage.

    python testing/luke_rescue_d2b1_field_tolerance.py

Status: corrected rerun pending (C2 discovery-cohort donors). Outputs to
testing/outputs/luke_rescue_d2b1_field_tolerance_v2/. Nothing under /mnt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_inject import drift_penalty
from testing.ladder_l1 import l1_run
from testing.ladder_motion import (
    observed_waveform,
    oracle_corrected_recording,
    perturb_bias,
    perturb_depth_gradient,
    perturb_gain,
    perturb_none,
    perturb_time_lag,
    perturb_time_smooth,
    waveform_preservation,
)
from testing.ladder_score import score_sort
from testing.luke_rescue_c2_drift_challenge import (
    OUTPUT as C2_OUTPUT,
    PRESPEC as C2_PRESPEC,
    _train,
    _trajectory_fn,
    load_background,
    prepare_template,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_d2b1_field_tolerance_v2"
C2_RUNS = C2_OUTPUT / "runs"
DONOR_TEMPLATES = (
    REPO_ROOT / "testing/outputs/luke_injected_ground_truth_pilot/donor_templates.npz"
)

PRESPEC = {
    "schema": "luke-rescue-d2b1-field-tolerance-v2",
    "frozen": "2026-09-03",
    "status": "corrected_geometry_aware_rerun_pending",
    "question": (
        "Starting from the exact injected trajectory, how much error in "
        "displacement amplitude, timing, and spatial structure can be tolerated "
        "before corrected voltage performs no better than the uncorrected rescue "
        "baseline?"
    ),
    # (donor, trajectory): the pairs where Candidate 2's exact oracle beat rescue,
    # plus T01/rigid_40um as the wrong-side-of-the-tradeoff control.
    "focus": [
        ["T04", "rigid_40um"],
        ["T06", "rigid_40um"],
        ["T04", "osc_20um_40s"],
        ["T01", "rigid_40um"],
    ],
    "perturbations": {
        "gain": [0.5, 0.75, 1.25, 1.5],
        "time_smooth_s": [3.0, 10.0],
        "time_lag_s": [2.0, 6.0],
        "bias_um": [8.0, 20.0],
        "depth_gradient": [0.3, 0.7],
    },
    "envelope_rule": (
        "recovery_fraction = (acc - rescue_acc) / (oracle_acc - rescue_acc); the "
        "tolerated error is the largest magnitude with recovery_fraction >= 0 "
        "(still beats no correction) and, more strictly, >= 0.5"
    ),
}

BG = C2_PRESPEC["background"]
TP = C2_PRESPEC["template_prep"]


def _freeze_prespec() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "prespec.json"
    if path.exists():
        if json.loads(path.read_text()) != PRESPEC:
            raise SystemExit(f"{path} differs from frozen PRESPEC; delete the tree to re-freeze.")
    else:
        path.write_text(json.dumps(PRESPEC, indent=2) + "\n")


def _perturbations(geometry) -> list[tuple[str, str, object]]:
    depth_um = float(np.asarray(geometry)[:, 1].mean())
    span_um = float(np.ptp(np.asarray(geometry)[:, 1]))
    out: list[tuple[str, str, object]] = [("exact", "exact", perturb_none())]
    for g in PRESPEC["perturbations"]["gain"]:
        out.append(("gain", f"gain_{g}", perturb_gain(g)))
    for s in PRESPEC["perturbations"]["time_smooth_s"]:
        out.append(("time_smooth", f"smooth_{s}s", perturb_time_smooth(s)))
    for l in PRESPEC["perturbations"]["time_lag_s"]:
        out.append(("time_lag", f"lag_{l}s", perturb_time_lag(l)))
    for b in PRESPEC["perturbations"]["bias_um"]:
        out.append(("bias", f"bias_{b}um", perturb_bias(b)))
    for f in PRESPEC["perturbations"]["depth_gradient"]:
        out.append(("depth_gradient", f"grad_{f}", perturb_depth_gradient(f, depth_um, span_um)))
    return out


def run() -> dict:
    _freeze_prespec()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bg_uv, geometry, fs, _, _ = load_background()
    duration_s = bg_uv.shape[0] / fs
    truth = {"inj0": _train(C2_PRESPEC["background"]["duration_s"], fs)}
    donors = np.load(DONOR_TEMPLATES)
    perturbs = _perturbations(geometry)

    rows = []
    for tid, traj_name in PRESPEC["focus"]:
        rec_dir = C2_RUNS / f"{tid}_{traj_name}"
        if not (rec_dir / "snippet_manifest.json").exists():
            continue
        manifest = json.loads((rec_dir / "snippet_manifest.json").read_text())
        gain = float(manifest["gain_uv_per_count"])
        traj_fn, _ = _trajectory_fn(traj_name, geometry, duration_s)

        template, peak_col = prepare_template(donors[tid])
        base_channel = BG["channel_count"] // 2 - peak_col

        # floor: uncorrected rescue on this exact recording (cached from C2)
        rescue_score = l1_run(rec_dir, truth=truth, out_root=C2_RUNS / "_l1")["score"]
        rescue_acc = rescue_score["primary"]["units"][0]["accuracy"]

        oracle_acc = None
        for kind, label, fn in perturbs:
            corrected = OUTPUT / "corrected" / f"{tid}_{traj_name}_{label}"
            oracle_corrected_recording(
                rec_dir, corrected, trajectory_fn=traj_fn, duration_s=duration_s,
                fs=fs, gain_uv_per_count=gain, perturbation=fn,
                perturbation_label=label, name=f"d2b1_{tid}_{traj_name}_{label}",
            )
            score = l1_run(corrected, truth=truth, out_root=OUTPUT / "_l1")["score"]
            u = score["primary"]["units"][0]
            wf = waveform_preservation(
                observed_waveform(
                    corrected, truth["inj0"], base_channel=base_channel,
                    width=template.shape[1], n_samples=template.shape[0],
                ),
                template,
            )
            if label == "exact":
                oracle_acc = u["accuracy"]
            rec_frac = (
                (u["accuracy"] - rescue_acc) / (oracle_acc - rescue_acc)
                if oracle_acc is not None and oracle_acc > rescue_acc
                else np.nan
            )
            rows.append({
                "donor": tid, "trajectory": traj_name,
                "perturbation_kind": kind, "perturbation": label,
                "accuracy": round(u["accuracy"], 3),
                "rescue_acc": round(rescue_acc, 3),
                "oracle_acc": round(oracle_acc, 3) if oracle_acc is not None else None,
                "recovery_fraction": round(rec_frac, 3) if rec_frac == rec_frac else None,
                "beats_no_correction": bool(u["accuracy"] > rescue_acc + 0.02),
                "identities": u["n_output_units_capturing"],
                "tp": u["tp"], "fp": u["fp"], "fn": u["fn"],
                **wf,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "field_tolerance.csv", index=False)

    # tolerance envelope: per (donor, trajectory, perturbation_kind), the levels
    # that still beat no correction
    env = {}
    for (tid, traj, kind), sub in df.groupby(["donor", "trajectory", "perturbation_kind"]):
        if kind == "exact":
            continue
        good = sub[sub.beats_no_correction]
        env[f"{tid}:{traj}:{kind}"] = {
            "levels_tested": sub.perturbation.tolist(),
            "levels_still_helping": good.perturbation.tolist(),
            "min_recovery_fraction": (
                round(float(sub.recovery_fraction.min()), 3)
                if sub.recovery_fraction.notna().any() else None
            ),
        }

    summary = {
        "prespec_schema": PRESPEC["schema"],
        "focus": PRESPEC["focus"],
        "exact_oracle_vs_rescue": {
            f"{r.donor}:{r.trajectory}": {
                "rescue_acc": r.rescue_acc, "oracle_acc": r.oracle_acc,
            }
            for r in df[df.perturbation == "exact"].itertuples()
        },
        "envelope": env,
        "waveform_cosine_range": [
            round(float(df.waveform_cosine.min()), 3),
            round(float(df.waveform_cosine.max()), 3),
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n", 1)[0]).parse_args()
    summary = run()
    print(json.dumps(summary, indent=2))
    df = pd.read_csv(OUTPUT / "field_tolerance.csv")
    pd.set_option("display.width", 220)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
