"""Does motion-aware family stitching recover the C2 drift penalty?

C2 (`luke_rescue_c2_drift_challenge.py`) injected one neuron on a known
trajectory and showed the no-motion pipeline shatters it — accuracy −0.3 to
−0.8, up to +15 output identities. Phase A2 + C2 make **post-sort family
stitching** the first Phase D candidate.

This is its first known-truth test. For every C2 injected condition it:

1. loads the frozen truth train and the rescue L1 curated sort;
2. scores it against truth (`score_sort`) — the pre-stitch baseline;
3. runs `ladder_stitch.apply_stitch`;
4. scores the stitched sort against the same truth;
5. reports Δ accuracy, Δ output identities, Δ label switches from stitching.

A candidate that works recovers accuracy on the moving arms and leaves the
static arms alone (it must not over-merge a cleanly-recovered neuron).

    python testing/luke_rescue_stitch_c2_eval.py

Outputs to testing/outputs/luke_rescue_stitch_c2_eval/. No sorter is run — this
is a pure post-processing evaluation on cached C2 outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.ladder_score import score_sort
from testing.ladder_stitch import StitchConfig, apply_stitch
from testing.luke_rescue_c2_drift_challenge import (
    OUTPUT as C2_OUTPUT,
    PRESPEC,
    _train,
    load_background,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_stitch_c2_eval"
C2_RUNS = C2_OUTPUT / "runs"


def _truth_train() -> np.ndarray:
    _, _, fs, _, _ = load_background()
    return _train(PRESPEC["background"]["duration_s"], fs)


def _condition_dirs() -> list[tuple[str, Path]]:
    """(tag, rescue curated cur_output) for every injected C2 condition."""
    out = []
    for rec_dir in sorted(C2_RUNS.glob("T*_*")):
        man = json.loads((rec_dir / "snippet_manifest.json").read_text())
        digest16 = man["spec_digest"][:16]
        cur = C2_RUNS / "_l1" / digest16
        # rescue leaf: cur-<curation digest> (no sorter prefix)
        leaves = [d for d in cur.glob("cur-*") if d.name.count("-") == 1]
        if leaves and (leaves[0] / "cur_output" / "spike_times.npy").exists():
            out.append((rec_dir.name, leaves[0] / "cur_output"))
    return out


def run(config: StitchConfig | None = None) -> pd.DataFrame:
    config = config or StitchConfig()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    truth = {"inj0": _truth_train()}

    rows = []
    for tag, cur in _condition_dirs():
        before = score_sort(cur, truth=truth)
        u0 = before["primary"]["units"][0]

        stitched = OUTPUT / "stitched" / tag
        result = apply_stitch(cur, stitched, config)
        after = score_sort(stitched, truth=truth)
        u1 = after["primary"]["units"][0]

        template, trajectory = tag.split("_", 1)
        rows.append({
            "template": template,
            "trajectory": trajectory,
            "n_good_before": result["n_good_before"],
            "n_good_after": result["n_good_after"],
            "families": len(result["families"]),
            "units_absorbed": result["n_units_absorbed"],
            "acc_before": round(u0["accuracy"], 3),
            "acc_after": round(u1["accuracy"], 3),
            "delta_acc": round(u1["accuracy"] - u0["accuracy"], 3),
            "identities_before": u0["n_output_units_capturing"],
            "identities_after": u1["n_output_units_capturing"],
            "switches_before": u0["label_switches"],
            "switches_after": u1["label_switches"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "stitch_c2_eval.csv", index=False)

    moving = df[df.trajectory != "static"]
    static = df[df.trajectory == "static"]
    summary = {
        "stitch_config": {k: v for k, v in vars(config).items()}
        if hasattr(config, "__dict__") else config.digest,
        "n_conditions": len(df),
        "moving_arms": {
            "median_delta_acc": round(float(moving.delta_acc.median()), 3),
            "n_improved": int((moving.delta_acc > 0.02).sum()),
            "n_hurt": int((moving.delta_acc < -0.02).sum()),
            "median_identities_before": float(moving.identities_before.median()),
            "median_identities_after": float(moving.identities_after.median()),
        },
        "static_arms": {
            "median_delta_acc": round(float(static.delta_acc.median()), 3),
            "n_hurt": int((static.delta_acc < -0.02).sum()),
            "note": "stitching must not damage a cleanly-recovered static neuron",
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return df


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n", 1)[0]).parse_args()
    df = run()
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(json.dumps(json.loads((OUTPUT / "summary.json").read_text()), indent=2))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
