"""Threshold candidates under displacement — the exact 40 µm staircase.

The static sweep nominated two candidates against production, on a response
surface with deterministic cliffs:

* **12/9** production baseline — 11/14 recovered, worst donor 0.475, 289 FP;
* **8/8** robustness candidate — 12/14, no donor below 0.976, 12 FP;
* **9/9** recovery candidate — 13/14, worst donor 0.891, 97 FP.

Static evidence alone cannot say whether a candidate stays viable when the
neuron moves. This runs all three against the **lattice-commensurate 40 µm
staircase**, the one displacement condition carrying no forward-model artifact
(the operator is exact to ~1e-6 there), so a difference between arms is sorter
behaviour rather than resampling damage.

Design
------
* **Correction off in every cell.** Thresholds remain the only variable; the
  staircase supplies displacement, not `nblocks`.
* **Paired static and moved arms** from one injected voltage and one crop, so
  the drift penalty is a within-donor, within-config Δ.
* **A fixed 687-event truth denominator for every cell** — the staircase
  admitted train, filtered before injection, used for the static arm too. A
  static baseline on a different train could not be subtracted from the moved
  arm.
* **Everything is retained** — injected recordings, sorter outputs and curation
  outputs — because the truncation diagnostic and the stage trace consume them
  after scoring. Earlier runners deleted recordings to save disk; that would
  make the follow-up analyses impossible.

Selection (per the plan): drift penalty plus FP/FN, splits, label switches and
truncation. Drop a candidate for a material fragmentation or contamination
regression. Not KS-good yield. No fractional cells here.

Run: `python testing/luke_c2_threshold_staircase.py [--donors ...]`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from testing.ladder_inject import inject_trajectory, static_trajectory, write_injected_recording
from testing.ladder_l1 import l1_run
from testing.ladder_score import assert_paired_truth
from testing.ladder_sorter import SorterConfig, effective_settings
from testing.luke_c2_staircase_control import (
    STAIRCASE,
    build_arms,
    load_wide_background,
    staircase_admitted_truth,
    staircase_truth_contract,
    staircase_um,
)
from testing.luke_c2_staircase_smoke import donor_placement
from testing.luke_rescue_c2_drift_challenge import (
    DONOR_GEOMETRY,
    DONOR_MANIFEST,
    DONOR_TEMPLATES,
    PRESPEC as C2_PRESPEC,
    _recording_dir,
    _verify_donor_cohort,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_c2_threshold_staircase"

COMPARISON = {
    "schema": "luke-c2-threshold-staircase-v1",
    "status": "development evidence; survivors go to L1C, then held-out",
    "candidates": {
        "th_12_9": "production baseline",
        "th_8_8": "robustness candidate",
        "th_9_9": "recovery candidate",
    },
    "correction": "do_correction=False in every cell",
    "arms": ["static", "staircase"],
    "condition": "lattice-commensurate 40 µm staircase — no forward-model artifact",
    "truth": "687 admitted events, identical for every arm and every candidate",
    "retain": "recordings, sorter outputs and curation outputs, for truncation + stage trace",
    "selection": [
        "drift_penalty", "fp", "fn", "n_output_units_capturing", "label_switches",
        "truncation",
    ],
    "drop_rule": "material fragmentation or contamination regression",
    "excluded": ["ks_good_yield_alone", "fractional_threshold_cells"],
}

CANDIDATES = [
    SorterConfig("th_12_9", {"do_correction": False, "Th_universal": 12, "Th_learned": 9}),
    SorterConfig("th_8_8", {"do_correction": False, "Th_universal": 8, "Th_learned": 8}),
    SorterConfig("th_9_9", {"do_correction": False, "Th_universal": 9, "Th_learned": 9}),
]


def output_root(explicit=None) -> Path:
    root = Path(explicit or os.environ.get("LUKE_THRESHOLD_STAIRCASE_ROOT") or DEFAULT_OUTPUT)
    if str(root.resolve()).startswith("/mnt/"):
        raise ValueError("refusing an output root under /mnt")
    return root


def run(donors=None, root=None) -> dict:
    root = output_root(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "prespec.json").write_text(json.dumps(COMPARISON, indent=2) + "\n")
    _verify_donor_cohort()
    import pandas as pd

    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    wide_uv, wide_geometry, wide_ids, fs, gain, crop, margin = load_wide_background()
    donors_npz = np.load(DONOR_TEMPLATES)
    meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id").to_dict("index")
    source_geometry = np.load(DONOR_GEOMETRY)
    crop_geometry = np.ascontiguousarray(wide_geometry[crop])
    tids = list(donors or sorted(donors_npz.files))

    regular = np.arange(
        int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(STAIRCASE["duration_s"] * fs) - int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(round(fs / C2_PRESPEC["train"]["rate_hz"])), dtype=np.int64,
    )
    # filter first; the same 687-event train is injected into both arms
    truth, admission = staircase_admitted_truth(regular, fs)
    admitted = truth["inj0"]

    rows = []
    for tid in tids:
        template, base_wide, _, _ = donor_placement(
            tid, donors_npz, meta, source_geometry, crop_geometry, margin
        )
        injected_wide = inject_trajectory(
            wide_uv.copy(), template, admitted, fs=fs, base_channel=base_wide,
            trajectory=static_trajectory(),
            amplitude_scale=C2_PRESPEC["amplitude_scale"], template_id=tid,
            edge_guard_samples=C2_PRESPEC["template_prep"]["edge_guard_samples"],
        )
        arms = build_arms(
            injected_wide, wide_geometry, fs, crop=crop, margin=margin,
            wide_channel_ids=wide_ids, trajectory_fn=staircase_um,
            labels=("static", "staircase", "unused"), with_corrected=False,
        )
        del injected_wide
        contract = staircase_truth_contract(truth, {"inj0": admitted}, arms, admission)

        rec_dirs = {}
        for arm in ("static", "staircase"):
            rec_dir = runs / f"{tid}_{arm}"
            write_injected_recording(
                rec_dir, arms[arm], channel_positions=arms["geometry"], fs=fs,
                gain_uv_per_count=gain, source_snippet_dir=str(_recording_dir()),
                name=f"{tid}_{arm}",
            )
            rec_dirs[arm] = rec_dir
        del arms

        for config in CANDIDATES:
            for arm in ("static", "staircase"):
                result = l1_run(rec_dirs[arm], sorter=config, truth=truth,
                                truth_contract=contract, out_root=runs / "_l1")
                obs = result["stage_observables"]
                eff = effective_settings({"summary": obs["sort_summary"],
                                          "sorter_params": obs.get("sort_request", {})})
                if eff["effective_nblocks"] != 0:
                    raise RuntimeError(f"{config.label} ran with correction on")
                unit = result["score"]["primary"]["units"][0]
                guard = result["score"]["guardrails"]
                rows.append({
                    "template": tid, "peak_uv": meta[tid]["peak_uv"],
                    "polarity": meta[tid]["polarity"], "candidate": config.label,
                    "Th_universal": eff["Th_universal"], "Th_learned": eff["Th_learned"],
                    "arm": arm, "n_truth": unit["n_truth"],
                    "accuracy": unit["accuracy"], "tp": unit["tp"], "fp": unit["fp"],
                    "fn": unit["fn"], "recovered": unit["recovered"],
                    "n_output_units_capturing": unit["n_output_units_capturing"],
                    "label_switches": unit["label_switches"],
                    "refractory_violation_median": guard.get("refractory_violation_median"),
                    "similar_pairs_per_good_unit": guard.get("similar_pairs_per_good_unit"),
                    "truth_sha256": contract["truth_sha256"][:12],
                })
        # recordings and sorts are deliberately retained for the follow-ups
        pd.DataFrame(rows).to_csv(root / "staircase_candidates.partial.csv", index=False)
        (root / "progress.json").write_text(json.dumps({
            "donors_done": tids[: tids.index(tid) + 1],
            "donors_remaining": tids[tids.index(tid) + 1:], "cells": len(rows),
        }, indent=2) + "\n")

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "staircase_candidates.csv", index=False)
    summary = {
        "comparison": COMPARISON, "n_cells": int(len(frame)), "n_donors": len(tids),
        "denominator": sorted(map(int, frame.n_truth.unique())),
        "truth_hashes": int(frame.truth_sha256.nunique()),
        "paired": assert_paired_truth([contract, contract], labels=["static", "staircase"]),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--donors", nargs="*", default=None)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    print(json.dumps(run(donors=args.donors, root=args.out_root), indent=2, default=str))


if __name__ == "__main__":
    main()
