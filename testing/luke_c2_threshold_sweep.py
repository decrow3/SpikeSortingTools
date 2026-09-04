"""Static detection-threshold sweep — the cheapest untested pipeline lever.

Why this, and why now
---------------------
C2 v4's actionable findings had nothing to do with motion. `rescue` fails D10
outright with no motion present (0.475 static, against 0.992 for legacy-style),
and on D12's staircase arm `legacy_style` recovered the neuron (0.991, 0 FP)
where `rescue_rigid` did not (0.529, 682 TP but **603 FP**) — same `nblocks=1`,
only the thresholds differ. That is not better *detection*: `rescue_rigid` finds
the spikes and buries them in contamination. Lower thresholds change the
resulting **clustering**.

So this sweep holds motion correction **off in every cell**, making
`Th_universal` and `Th_learned` the only variables, and asks whether the frozen
rescue thresholds (12/9) are costing real neurons on ground truth.

Design
------
* **Static arms only.** No warping — the expensive part of C2 vanishes, and the
  forward-model confound with it. One injected recording per donor, sorted under
  every threshold cell.
* **A two-dimensional grid**, not paired values, so `Th_universal` (detection)
  and `Th_learned` (template learning) can be separated. Constrained to
  `Th_learned <= Th_universal`.
* **The decisive missing configuration is included**: legacy thresholds `9/8`
  with correction *off*. Neither C2 v4 nor any earlier run has that cell, so
  "legacy recovers what rescue loses" has never been separated from "legacy
  applies motion correction".
* **Contamination endpoints alongside accuracy.** A threshold that raises TP
  while flooding FP is not an improvement; D12 is exactly that failure. Refractory
  violations and similar-pair burden come from the same guardrails the plan
  already uses for promotion.

Status: **development evidence, not confirmatory.** D10 motivated this
experiment, and D10 is in the cohort, so a winner here is a hypothesis. Per the
plan's held-out discipline it must be confirmed on donors or snippets not used
to choose it.

Run: `python testing/luke_c2_threshold_sweep.py [--donors ...] [--out-root ...]`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from testing.ladder_inject import inject_trajectory, static_trajectory, write_injected_recording
from testing.ladder_l1 import l1_run
from testing.ladder_score import build_truth_contract
from testing.ladder_sorter import SorterConfig, effective_settings
from testing.luke_c2_staircase_control import load_wide_background
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
DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_c2_threshold_sweep"

SWEEP = {
    "schema": "luke-c2-threshold-sweep-v1",
    "status": "development evidence; a winner needs held-out confirmation",
    "why_not_confirmatory": (
        "D10 motivated this experiment and D10 is in the cohort, so this cohort "
        "cannot also certify the answer"
    ),
    "question": (
        "with motion correction off, do the frozen rescue thresholds (12/9) cost "
        "recovery relative to lower thresholds, and at what contamination cost?"
    ),
    "arm": "static only — no warping, so no forward-model confound",
    "correction": "do_correction=False in every cell; thresholds are the only variable",
    "grid": {
        "Th_universal": [8, 9, 10, 12],
        "Th_learned": [7, 8, 9],
        "constraint": "Th_learned <= Th_universal",
    },
    "reference_cells": {
        "rescue": [12, 9],
        "legacy_thresholds_no_correction": [9, 8],
    },
    "endpoints": [
        "accuracy", "tp", "fp", "fn", "n_output_units_capturing", "label_switches",
        "refractory_violation_median", "similar_pairs_per_good_unit",
        "ks_good_count",
    ],
    "contamination_guard": (
        "a cell that raises TP while raising FP is not an improvement; D12's "
        "rescue_rigid arm scored 682 TP with 603 FP"
    ),
    "background": C2_PRESPEC["background"],
    "donor_cohort": C2_PRESPEC["template_prep"]["templates_sha256"],
}


def grid_configs() -> list[SorterConfig]:
    """Every (Th_universal, Th_learned) cell, correction explicitly off."""
    cells = []
    for tu in SWEEP["grid"]["Th_universal"]:
        for tl in SWEEP["grid"]["Th_learned"]:
            if tl > tu:
                continue
            cells.append(SorterConfig(
                f"th_{tu}_{tl}",
                {"do_correction": False, "Th_universal": tu, "Th_learned": tl},
            ))
    return cells


def check_cell(label: str, sort_manifest: dict, tu: int, tl: int) -> dict:
    """Correction really off, thresholds really applied. Fails closed."""
    observed = effective_settings(sort_manifest)
    problems = {}
    if observed["effective_nblocks"] != 0:
        problems["effective_nblocks"] = observed["effective_nblocks"]
    if observed["Th_universal"] != tu:
        problems["Th_universal"] = observed["Th_universal"]
    if observed["Th_learned"] != tl:
        problems["Th_learned"] = observed["Th_learned"]
    if problems:
        raise RuntimeError(f"{label} did not resolve as requested: {problems}")
    return observed


def output_root(explicit=None) -> Path:
    root = Path(explicit or os.environ.get("LUKE_THRESHOLD_ROOT") or DEFAULT_OUTPUT)
    if str(root.resolve()).startswith("/mnt/"):
        raise ValueError("refusing a sweep output root under /mnt")
    return root


def run(donors=None, root=None, keep_recordings: bool = False) -> dict:
    root = output_root(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "prespec.json").write_text(json.dumps(SWEEP, indent=2) + "\n")
    _verify_donor_cohort()
    import pandas as pd

    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    configs = grid_configs()

    wide_uv, wide_geometry, wide_ids, fs, gain, crop, margin = load_wide_background()
    donors_npz = np.load(DONOR_TEMPLATES)
    meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id").to_dict("index")
    source_geometry = np.load(DONOR_GEOMETRY)
    crop_geometry = np.ascontiguousarray(wide_geometry[crop])
    channel_ids = np.asarray(wide_ids)[crop]
    tids = list(donors or sorted(donors_npz.files))

    train = np.arange(
        int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(C2_PRESPEC["background"]["duration_s"] * fs)
        - int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(round(fs / C2_PRESPEC["train"]["rate_hz"])),
        dtype=np.int64,
    )
    truth = {"inj0": train}
    admission = {
        "schema": SWEEP["schema"], "rule": "static arm; every injected event is scored",
        "n_total": int(train.size), "n_admitted": int(train.size),
        "counts_by_level_um": {"0.0": int(train.size)},
    }
    contract = build_truth_contract(
        truth, injected=truth, admission=admission,
        channel_ids=channel_ids, geometry=crop_geometry,
        crop=(crop.start, crop.stop),
    )

    rows = []
    for tid in tids:
        template, base_wide, _, _ = donor_placement(
            tid, donors_npz, meta, source_geometry, crop_geometry, margin
        )
        injected = inject_trajectory(
            wide_uv.copy(), template, train, fs=fs, base_channel=base_wide,
            trajectory=static_trajectory(),
            amplitude_scale=C2_PRESPEC["amplitude_scale"], template_id=tid,
            edge_guard_samples=C2_PRESPEC["template_prep"]["edge_guard_samples"],
        )
        rec_dir = runs / f"{tid}_static"
        write_injected_recording(
            rec_dir, np.ascontiguousarray(injected[:, crop]),
            channel_positions=crop_geometry, fs=fs, gain_uv_per_count=gain,
            source_snippet_dir=str(_recording_dir()), name=f"{tid}_static",
        )
        del injected

        for config in configs:
            tu = config.overrides["Th_universal"]
            tl = config.overrides["Th_learned"]
            result = l1_run(rec_dir, sorter=config, truth=truth,
                            truth_contract=contract, out_root=runs / "_l1")
            obs = result["stage_observables"]
            check_cell(config.label, {"summary": obs["sort_summary"],
                                      "sorter_params": obs.get("sort_request", {})}, tu, tl)
            unit = result["score"]["primary"]["units"][0]
            guard = result["score"]["guardrails"]
            rows.append({
                "template": tid, "peak_uv": meta[tid]["peak_uv"],
                "polarity": meta[tid]["polarity"],
                "Th_universal": tu, "Th_learned": tl, "cell": config.label,
                "accuracy": unit["accuracy"], "tp": unit["tp"], "fp": unit["fp"],
                "fn": unit["fn"], "recovered": unit["recovered"],
                "n_output_units_capturing": unit["n_output_units_capturing"],
                "label_switches": unit["label_switches"],
                "refractory_violation_median": guard.get("refractory_violation_median"),
                "similar_pairs_per_good_unit": guard.get("similar_pairs_per_good_unit"),
                "ks_good_count": result["score"]["context"]["ks_good_count"],
                "is_rescue_reference": [tu, tl] == SWEEP["reference_cells"]["rescue"],
                "is_legacy_thresholds": (
                    [tu, tl] == SWEEP["reference_cells"]["legacy_thresholds_no_correction"]
                ),
            })
        if not keep_recordings:
            shutil.rmtree(rec_dir, ignore_errors=True)
        pd.DataFrame(rows).to_csv(root / "threshold_sweep.partial.csv", index=False)
        (root / "progress.json").write_text(json.dumps({
            "donors_done": tids[: tids.index(tid) + 1],
            "donors_remaining": tids[tids.index(tid) + 1:],
            "cells": len(rows),
        }, indent=2) + "\n")

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "threshold_sweep.csv", index=False)
    summary = {
        "sweep": SWEEP,
        "n_cells": int(len(frame)),
        "n_donors": len(tids),
        "n_configs": len(configs),
        "truth_sha256": contract["truth_sha256"],
        "n_expected": contract["n_expected"],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--donors", nargs="*", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--keep-recordings", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(donors=args.donors, root=args.out_root,
                         keep_recordings=args.keep_recordings), indent=2, default=str))


if __name__ == "__main__":
    main()
