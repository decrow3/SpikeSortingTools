"""Stage 1: what actually triggers the threshold cliffs — count, or which events?

The static comparison collapsed when the injected train went from 708 to 687
events: D02 at 8/8 fell 0.989 -> 0.492, D04 0.990 -> 0.553, D07 at 9/9
0.993 -> 0.453, each gaining 500-800 false positives, while production 12/9 was
unmoved. That invalidated the single-train ranking of 8/8 and 9/9.

But the 21 removed events are **structured** — they cluster at the staircase's
three step boundaries — so that observation cannot separate:

* **count** — 687 events instead of 708;
* **which events** — those particular 21, versus any 21;
* **phase** — absolute spike timing relative to the background;
* **composition** — general sensitivity to any train perturbation.

Every realisation here holds the count fixed at 687 except the `full_708`
reference, so count is varied on exactly one axis and identity/phase on the
others. All realisations are frozen (fixed seeds, hashes recorded) and every
configuration sees the identical realisation, so the comparison is paired.

Sentinels are the five donors that actually moved: D02, D04, D07 (collapsed),
D10 (the original motivating failure) and D14 (moved the other way).

This is a **diagnostic**, not a ranking. Stage 2 — preregistered fixed-count
realisations across all 14 donors, judged on median, lower tail, failure
probability and FP tail — only runs if scores here resolve into interpretable
distributions.

Run: `python testing/luke_c2_train_sentinel.py`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from testing.ladder_inject import inject_trajectory, static_trajectory, write_injected_recording
from testing.ladder_l1 import l1_run
from testing.ladder_score import build_truth_contract, truth_digest
from testing.ladder_sorter import SorterConfig, effective_settings
from testing.luke_c2_staircase_control import (
    STAIRCASE,
    load_wide_background,
    staircase_admitted_truth,
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
DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_c2_train_sentinel"

SENTINEL = {
    "schema": "luke-c2-train-sentinel-v1",
    "status": "diagnostic; not a ranking and not a candidate selection",
    "question": (
        "does a threshold collapse follow event count, the identity of the "
        "removed events, absolute phase, or general train composition?"
    ),
    "donors": ["D02", "D04", "D07", "D10", "D14"],
    "why_these": {
        "D02": "collapsed 0.989 -> 0.492 at 8/8 on the boundary-deleted train",
        "D04": "collapsed 0.990 -> 0.553 at 8/8",
        "D07": "collapsed 0.993 -> 0.453 at 9/9",
        "D10": "the original 12/9 detection failure that motivated the sweep",
        "D14": "moved the other way, 0.891 -> 0.994 at 9/9",
    },
    "candidates": ["th_12_9", "th_8_8", "th_9_9"],
    "correction": "do_correction=False everywhere",
    "arm": "static only — displacement is not the variable here",
    "realisations": {
        "full_708": "the sweep's train; the only cell with a different count",
        "boundary_687": "the staircase-admitted train — structured deletion at the three steps",
        "random_687_s1": "21 uniformly-random deletions, seed 1",
        "random_687_s2": "21 uniformly-random deletions, seed 2",
        "random_687_s3": "21 uniformly-random deletions, seed 3",
        "uniform_687": "every 34th event deleted — spread, not clustered",
        "phase_687": "boundary_687 shifted by half an inter-spike interval",
    },
    "reads": {
        "count": "full_708 vs any 687 realisation",
        "which_events": "boundary_687 vs random/uniform at the same count",
        "phase": "boundary_687 vs phase_687 — same events, shifted timing",
        "composition": "spread across the three random seeds",
    },
    "paired": "every configuration sees the identical frozen realisation",
}

CANDIDATES = [
    SorterConfig("th_12_9", {"do_correction": False, "Th_universal": 12, "Th_learned": 9}),
    SorterConfig("th_8_8", {"do_correction": False, "Th_universal": 8, "Th_learned": 8}),
    SorterConfig("th_9_9", {"do_correction": False, "Th_universal": 9, "Th_learned": 9}),
]


def realisations(fs: float) -> dict[str, np.ndarray]:
    """Frozen trains. Every 687-event variant removes exactly 21 of the 708."""
    full = np.arange(
        int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(STAIRCASE["duration_s"] * fs) - int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(round(fs / C2_PRESPEC["train"]["rate_hz"])), dtype=np.int64,
    )
    boundary, _ = staircase_admitted_truth(full, fs)
    boundary = boundary["inj0"]
    n_removed = full.size - boundary.size

    out = {"full_708": full, "boundary_687": boundary}
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        drop = rng.choice(full.size, size=n_removed, replace=False)
        out[f"random_687_s{seed}"] = np.delete(full, np.sort(drop))
    step = full.size // n_removed
    out["uniform_687"] = np.delete(full, np.arange(n_removed) * step)
    # same events as boundary_687, every time shifted by half an ISI: isolates
    # absolute phase from which events were removed
    isi = int(round(fs / C2_PRESPEC["train"]["rate_hz"]))
    out["phase_687"] = boundary + isi // 2

    for name, train in out.items():
        expected = full.size if name == "full_708" else full.size - n_removed
        if train.size != expected:
            raise RuntimeError(f"{name} has {train.size} events, expected {expected}")
    return out


def output_root(explicit=None) -> Path:
    root = Path(explicit or os.environ.get("LUKE_SENTINEL_ROOT") or DEFAULT_OUTPUT)
    if str(root.resolve()).startswith("/mnt/"):
        raise ValueError("refusing an output root under /mnt")
    return root


def run(donors=None, root=None) -> dict:
    root = output_root(root)
    root.mkdir(parents=True, exist_ok=True)
    _verify_donor_cohort()
    import pandas as pd

    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    wide_uv, wide_geometry, wide_ids, fs, gain, crop, margin = load_wide_background()
    donors_npz = np.load(DONOR_TEMPLATES)
    meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id").to_dict("index")
    source_geometry = np.load(DONOR_GEOMETRY)
    crop_geometry = np.ascontiguousarray(wide_geometry[crop])
    channel_ids = np.asarray(wide_ids)[crop]
    tids = list(donors or SENTINEL["donors"])

    trains = realisations(fs)
    frozen = {name: {"n": int(t.size), "sha256": truth_digest({"inj0": t})}
              for name, t in trains.items()}
    (root / "prespec.json").write_text(
        json.dumps({**SENTINEL, "frozen_realisations": frozen}, indent=2) + "\n")

    rows = []
    for tid in tids:
        template, base_wide, _, _ = donor_placement(
            tid, donors_npz, meta, source_geometry, crop_geometry, margin
        )
        for name, train in trains.items():
            truth = {"inj0": train}
            contract = build_truth_contract(
                truth, injected=truth,
                admission={"schema": SENTINEL["schema"], "rule": name,
                           "n_total": int(trains["full_708"].size),
                           "n_admitted": int(train.size),
                           "counts_by_level_um": {"0.0": int(train.size)}},
                channel_ids=channel_ids, geometry=crop_geometry,
                crop=(crop.start, crop.stop),
            )
            injected = inject_trajectory(
                wide_uv.copy(), template, train, fs=fs, base_channel=base_wide,
                trajectory=static_trajectory(),
                amplitude_scale=C2_PRESPEC["amplitude_scale"], template_id=tid,
                edge_guard_samples=C2_PRESPEC["template_prep"]["edge_guard_samples"],
            )
            rec_dir = runs / f"{tid}_{name}"
            write_injected_recording(
                rec_dir, np.ascontiguousarray(injected[:, crop]),
                channel_positions=crop_geometry, fs=fs, gain_uv_per_count=gain,
                source_snippet_dir=str(_recording_dir()), name=f"{tid}_{name}",
            )
            del injected

            for config in CANDIDATES:
                result = l1_run(rec_dir, sorter=config, truth=truth,
                                truth_contract=contract, out_root=runs / "_l1")
                obs = result["stage_observables"]
                eff = effective_settings({"summary": obs["sort_summary"],
                                          "sorter_params": obs.get("sort_request", {})})
                if eff["effective_nblocks"] != 0:
                    raise RuntimeError(f"{config.label} ran with correction on")
                unit = result["score"]["primary"]["units"][0]
                guard = result["score"]["guardrails"]
                rows.append({
                    "template": tid, "realisation": name,
                    "n_events": int(train.size), "candidate": config.label,
                    "accuracy": unit["accuracy"], "tp": unit["tp"], "fp": unit["fp"],
                    "fn": unit["fn"], "recovered": unit["recovered"],
                    "n_output_units_capturing": unit["n_output_units_capturing"],
                    "refractory_violation_median": guard.get("refractory_violation_median"),
                    "truth_sha256": contract["truth_sha256"][:12],
                })
            pd.DataFrame(rows).to_csv(root / "sentinel.partial.csv", index=False)
        (root / "progress.json").write_text(json.dumps({
            "donors_done": tids[: tids.index(tid) + 1], "cells": len(rows),
        }, indent=2) + "\n")

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "sentinel.csv", index=False)
    summary = {"sentinel": SENTINEL, "frozen_realisations": frozen,
               "n_cells": int(len(frame)), "n_donors": len(tids)}
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
