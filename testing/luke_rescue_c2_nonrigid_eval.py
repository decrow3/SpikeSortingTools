"""Phase D candidate 2 — rerun the non-rigid representation test correctly.

The original C2 penalty and Candidate 2 conclusions are retracted. This version
uses the geometry-aware C2 recordings to determine whether a reproducible drift
penalty exists and whether non-rigid correction closes it.

This evaluates two non-rigid representations against the corrected cached C2
injected recordings, scored the identical way (drift penalty = moving − static
on the injected unit):

* **`nonrigid`** — KS4's own non-rigid datashift (`do_correction=True,
  nblocks=6`), rescue detection thresholds unchanged. The estimated case.
* **`oracle`** — correct the recording with the **exact** injected trajectory
  (`ladder_motion.oracle_corrected_recording`), then the frozen rescue sort.
  The ceiling: the best any motion representation could do here.

Reading:

* oracle closes it, nonrigid does not  → the lever is estimation quality;
* neither closes it                    → the residual is KS4 clustering on a
  moving footprint, not the motion representation — candidate 2 is the wrong
  lever and Phase D needs a different idea;
* both close it                        → adopt `nonrigid`, take it to L2.

    python testing/luke_rescue_c2_nonrigid_eval.py

Status: corrected rerun pending (reuses the compact-donor C2 v3 recordings).
Outputs to testing/outputs/luke_rescue_c2_nonrigid_eval_v3/.
Nothing under /mnt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_inject import drift_penalty
from testing.ladder_l1 import l1_run
from testing.ladder_motion import oracle_corrected_recording
from testing.ladder_sorter import NAMED_CONFIGS
from testing.luke_rescue_c2_drift_challenge import (
    OUTPUT as C2_OUTPUT,
    PRESPEC as C2_PRESPEC,
    _train,
    _trajectory_fn,
    load_background,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_c2_nonrigid_eval_v3"
C2_RUNS = C2_OUTPUT / "runs"

PRESPEC = {
    "schema": "luke-rescue-c2-nonrigid-v3",
    "frozen": "2026-09-03",
    "status": "corrected_geometry_aware_rerun_pending",
    "question": (
        "Does a non-rigid motion representation (KS4 datashift nblocks=6, or "
        "oracle correction with the known trajectory) close the C2 drift penalty "
        "that rigid correction did not?"
    ),
    "arms": {
        "rescue": "frozen rescue config, no motion representation (the C2 baseline)",
        "nonrigid": "KS4 non-rigid datashift do_correction=True nblocks=6",
        "oracle": "InterpolateMotionRecording with the exact injected trajectory, then rescue sort",
    },
    "reuses": "testing/outputs/luke_rescue_c2_drift_challenge_v3/runs",
    "oracle_sign_check": (
        "the static arm must stay >= 0.9 accuracy after oracle correction; a "
        "sign error would corrupt it"
    ),
    "decision": {
        "oracle_closes_nonrigid_does_not": "lever is estimation quality",
        "neither_closes": "residual is clustering on a moving footprint; wrong lever",
        "both_close": "adopt nonrigid, take to L2",
    },
}

TRAJECTORIES = C2_PRESPEC["trajectories"]


def _freeze_prespec() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "prespec.json"
    if path.exists():
        if json.loads(path.read_text()) != PRESPEC:
            raise SystemExit(
                f"{path} differs from the frozen PRESPEC. Delete the output tree "
                "to re-freeze."
            )
    else:
        path.write_text(json.dumps(PRESPEC, indent=2) + "\n")


def _conditions() -> list[str]:
    """C2 donors that passed static qualification under both comparators."""
    summary_path = C2_OUTPUT / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("run compact-donor C2 v3 before its nonrigid evaluation")
    summary = json.loads(summary_path.read_text())
    return list(summary.get("qualified_templates", []))


def _score_arm(rec_dir: Path, truth: dict, arm: str, geometry, fs, duration_s):
    """Return the score dict for one (recording, arm)."""
    if arm == "rescue":
        # reuse C2's cached rescue sort/curation for this exact recording
        return l1_run(rec_dir, truth=truth, out_root=C2_RUNS / "_l1")["score"]
    if arm == "nonrigid":
        return l1_run(
            rec_dir, sorter=NAMED_CONFIGS["nonrigid"], truth=truth,
            out_root=OUTPUT / "_l1",
        )["score"]
    if arm == "oracle":
        traj_name = rec_dir.name.split("_", 1)[1]
        traj_fn, _ = _trajectory_fn(traj_name, geometry, duration_s)
        manifest = json.loads((rec_dir / "snippet_manifest.json").read_text())
        gain = float(manifest["gain_uv_per_count"])
        corrected = OUTPUT / "corrected" / rec_dir.name
        oracle_corrected_recording(
            rec_dir, corrected, trajectory_fn=traj_fn, duration_s=duration_s,
            fs=fs, gain_uv_per_count=gain,
        )
        return l1_run(corrected, truth=truth, out_root=OUTPUT / "_l1")["score"]
    raise ValueError(arm)


def run(templates: list[str] | None = None, arms: list[str] | None = None) -> dict:
    _freeze_prespec()
    templates = templates or _conditions()
    arms = arms or ["rescue", "nonrigid", "oracle"]

    bg_uv, geometry, fs, _, _ = load_background()
    duration_s = bg_uv.shape[0] / fs
    truth = {"inj0": _train(C2_PRESPEC["background"]["duration_s"], fs)}

    rows = []
    for tid in templates:
        for arm in arms:
            scores = {}
            for traj_name in TRAJECTORIES:
                rec_dir = C2_RUNS / f"{tid}_{traj_name}"
                if not (rec_dir / "snippet_manifest.json").exists():
                    continue
                s = _score_arm(rec_dir, truth, arm, geometry, fs, duration_s)
                u = s["primary"]["units"][0]
                scores[traj_name] = s
                rows.append({
                    "template": tid, "arm": arm, "trajectory": traj_name,
                    "accuracy": round(u["accuracy"], 3),
                    "identities": u["n_output_units_capturing"],
                    "label_switches": u["label_switches"],
                    "tp": u["tp"], "fp": u["fp"], "fn": u["fn"],
                })
            if "static" not in scores:
                continue
            for traj_name in TRAJECTORIES:
                if traj_name == "static" or traj_name not in scores:
                    continue
                pen = drift_penalty(scores["static"], scores[traj_name], "inj0")
                rows.append({
                    "template": tid, "arm": arm,
                    "trajectory": f"PENALTY:{traj_name}",
                    "accuracy": pen["delta_accuracy"],
                    "identities": pen["delta_n_identities"],
                    "label_switches": pen["delta_label_switches"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "nonrigid_eval.csv", index=False)

    pen = df[df.trajectory.str.startswith("PENALTY")]
    static = df[df.trajectory == "static"]
    # a sign error in the oracle correction would corrupt the static arm; check
    # it against the same template's uncorrected static rather than an absolute
    # bar. C2 v3 has already excluded donors that fail its paired static gate.
    rescue_static = static[static.arm == "rescue"].set_index("template").accuracy
    oracle_static = static[static.arm == "oracle"].set_index("template").accuracy
    oracle_static_ok = bool(
        ((oracle_static - rescue_static).dropna() > -0.05).all()
    ) if len(oracle_static) else None
    summary = {
        "prespec_schema": PRESPEC["schema"],
        "templates": templates,
        "arms": arms,
        "static_accuracy_by_arm": {
            a: round(float(sub.accuracy.min()), 3)
            for a, sub in static.groupby("arm")
        },
        "median_drift_penalty_by_arm": {
            a: round(float(sub.accuracy.median()), 3)
            for a, sub in pen.groupby("arm")
        },
        "penalty_by_arm_trajectory": {
            a: {
                t: round(float(g.accuracy.median()), 3)
                for t, g in sub.groupby("trajectory")
            }
            for a, sub in pen.groupby("arm")
        },
        "oracle_static_preserved": oracle_static_ok,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--templates", nargs="+")
    ap.add_argument("--arms", nargs="+", choices=["rescue", "nonrigid", "oracle"])
    args = ap.parse_args()
    summary = run(args.templates, args.arms)
    print(json.dumps(summary, indent=2))
    df = pd.read_csv(OUTPUT / "nonrigid_eval.csv")
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
