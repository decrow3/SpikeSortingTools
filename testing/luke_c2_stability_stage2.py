"""Stage 2: rank threshold configurations on distributions, not single scores.

Prespec: `docs/luke_c2_train_stability_stage2_prespec.md`.

Stage 1 ([`luke_c2_train_sentinel_result.md`](../docs/luke_c2_train_sentinel_result.md))
established that threshold collapses follow train **composition and phase**, not
event count, and that a given realisation is a lottery ticket for a given
donor × configuration. It also showed 2-3 sporadic failures per configuration in
28 cells cannot separate candidates. This run gives the failure rate enough
precision to be usable, or establishes that it cannot be had at this cost.

Design
------
14 donors × 14 realisations × 3 configurations = **588 cells**, static only,
correction off, **paired**: every configuration sees the identical frozen
realisation, so comparisons are realisation-matched.

Realisations hold the count at 687 throughout and cross the two axes stage 1
showed can each trigger a collapse alone:

* `random_s1..s6` at phase 0 — composition;
* `random_s1..s6` at phase +½ ISI — the *same* deletions, shifted;
* `boundary_687`, `uniform_687` — named references from stage 1.

Disk
----
Retaining recordings would need ~270 GB (measured from stage 1: 0.75 GB per
recording, 0.21 GB per sort; every figure in this module derives from those two
constants rather than being written out separately). Each recording is therefore
deleted once its three sorts are done, which caps the run at ~124 GB. Sorter and
curation outputs **are** retained, so any stage-2 collapse can still be
stage-traced; only the regenerable voltage goes. `--keep-recordings` overrides
this and needs the full amount.

Deletion is **fail-closed** and free space is re-checked every realisation: if a
recording cannot be removed the run stops rather than silently filling the
volume, which the single start-up guard would not catch.

This module runs cells. It computes no endpoints and makes no decisions — see
`luke_c2_stability_stage2_analysis.py`, which implements the prespecified
endpoints and decision rules.

Run: `python testing/luke_c2_stability_stage2.py [--dry-run]`
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
from testing.ladder_score import build_truth_contract, truth_digest
from testing.ladder_sorter import SorterConfig, effective_settings
from testing.luke_c2_staircase_control import STAIRCASE, load_wide_background, staircase_admitted_truth
from testing.luke_c2_staircase_smoke import donor_placement
from testing.luke_rescue_c2_drift_challenge import (
    DONOR_GEOMETRY,
    DONOR_MANIFEST,
    DONOR_TEMPLATES,
    PRESPEC as C2_PRESPEC,
    _recording_dir,
    _resolve_frozen_cohort,
    _verify_donor_cohort,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_c2_stability_stage2"

# measured from stage 1: 0.75 GB per injected recording, 0.21 GB per retained sort
GB_PER_SORT = 0.21
GB_PER_RECORDING = 0.75
DISK_HEADROOM_GB = 25
EXPECTED_EVENTS = 687   # every realisation, always; count is not a variable
EXPECTED_DONORS = 14

STAGE2 = {
    "schema": "luke-c2-stability-stage2-v1",
    "prespec": "docs/luke_c2_train_stability_stage2_prespec.md",
    "status": "development evidence; L1C, held-out and fractional cells stay paused",
    "donors": "all 14 compact D2b-2 donors",
    "n_realisations": 14,
    "candidates": ["th_12_9", "th_8_8", "th_9_9"],
    "arm": "static only — displacement is not the variable",
    "correction": "do_correction=False everywhere",
    "paired": "every configuration sees the identical frozen realisation",
    "failure_threshold": 0.9,
    "systematic_min_failures": 12,
    "events_per_realisation": EXPECTED_EVENTS,
    "n_donors": EXPECTED_DONORS,
    "operational_baseline": "th_12_9 — unchanged by this run",
}

CANDIDATES = [
    SorterConfig("th_12_9", {"do_correction": False, "Th_universal": 12, "Th_learned": 9}),
    SorterConfig("th_8_8", {"do_correction": False, "Th_universal": 8, "Th_learned": 8}),
    SorterConfig("th_9_9", {"do_correction": False, "Th_universal": 9, "Th_learned": 9}),
]


def realisations(fs: float) -> dict[str, np.ndarray]:
    """The 14 frozen trains. Every one holds 687 events.

    Six random deletion sets are each rendered at phase 0 and at +½ ISI, so
    composition and phase are crossed rather than confounded; stage 1 showed
    either can trigger a collapse on its own.
    """
    full = np.arange(
        int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(STAIRCASE["duration_s"] * fs) - int(C2_PRESPEC["train"]["guard_s"] * fs),
        int(round(fs / C2_PRESPEC["train"]["rate_hz"])), dtype=np.int64,
    )
    boundary = staircase_admitted_truth(full, fs)[0]["inj0"]
    n_removed = full.size - boundary.size
    isi = int(round(fs / C2_PRESPEC["train"]["rate_hz"]))

    out: dict[str, np.ndarray] = {}
    for seed in range(1, 7):
        rng = np.random.default_rng(seed)
        keep = np.delete(full, np.sort(rng.choice(full.size, n_removed, replace=False)))
        out[f"random_s{seed}_p0"] = keep
        out[f"random_s{seed}_phalf"] = keep + isi // 2
    out["boundary_687"] = boundary
    step = full.size // n_removed
    out["uniform_687"] = np.delete(full, np.arange(n_removed) * step)

    if len(out) != STAGE2["n_realisations"]:
        raise RuntimeError(f"expected {STAGE2['n_realisations']} realisations, built {len(out)}")
    # Equality alone is not enough: if upstream admission shifted every train to
    # 686 the count would still be "constant" but no longer the prespecified one.
    sizes = {int(t.size) for t in out.values()}
    if sizes != {EXPECTED_EVENTS}:
        raise RuntimeError(
            f"every realisation must hold exactly {EXPECTED_EVENTS} events, got {sizes}"
        )
    return out


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename, so an interrupted run leaves no torn file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_csv(frame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


THRESHOLD_KEYS = ("Th_universal", "Th_learned")


def assert_applied_settings(label: str, eff: dict, requested: dict) -> dict:
    """Fail unless KS4 *applied* the requested thresholds with correction off.

    effective_settings() falls back to the requested value when a manifest
    records no applied one — a normalisation that is right for production
    manifests and fatal here, because comparing that fallback to the request it
    came from succeeds by construction. So provenance is checked first: every
    threshold must carry an `applied:` source, or the cell is refused.
    """
    sources = eff.get("_sources", {})
    unproven = {k: sources.get(k) for k in THRESHOLD_KEYS
                if not str(sources.get(k, "")).startswith("applied:")}
    if unproven:
        raise RuntimeError(
            f"{label} thresholds are not applied-derived ({unproven}); this "
            "runner will not verify a threshold against the request that "
            "produced it"
        )
    mismatch = {k: {"requested": requested[k], "effective": eff.get(k)}
                for k in THRESHOLD_KEYS if eff.get(k) != requested[k]}
    if eff["effective_nblocks"] != 0:
        mismatch["effective_nblocks"] = {"requested": 0,
                                         "effective": eff["effective_nblocks"]}
    if mismatch:
        raise RuntimeError(f"{label} did not resolve as requested: {mismatch}")
    return eff


def output_root(explicit=None) -> Path:
    root = Path(explicit or os.environ.get("LUKE_STAGE2_ROOT") or DEFAULT_OUTPUT)
    if str(root.resolve()).startswith("/mnt/"):
        raise ValueError("refusing an output root under /mnt")
    return root


def plan(n_donors: int, n_realisations: int, keep_recordings: bool) -> dict:
    """Cell count and the disk this will actually need."""
    cells = n_donors * n_realisations * len(CANDIDATES)
    recordings = n_donors * n_realisations
    gb = cells * GB_PER_SORT + (recordings if keep_recordings else 1) * GB_PER_RECORDING
    return {"cells": cells, "recordings": recordings,
            "estimated_gb": round(gb, 1), "keep_recordings": keep_recordings}


def check_disk(required_gb: float, root: Path) -> dict:
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < required_gb + DISK_HEADROOM_GB:
        raise RuntimeError(
            f"needs ~{required_gb:.0f} GB plus {DISK_HEADROOM_GB} GB headroom, "
            f"{free_gb:.0f} GB free. Retaining recordings costs ~278 GB; the "
            "default deletes each recording after its sorts."
        )
    return {"free_gb": round(free_gb, 1), "required_gb": required_gb}


def run(donors=None, root=None, keep_recordings: bool = False,
        dry_run: bool = False) -> dict:
    root = output_root(root)
    root.mkdir(parents=True, exist_ok=True)
    _verify_donor_cohort()
    import pandas as pd

    donors_npz = np.load(DONOR_TEMPLATES)
    # hash-verified cohort *and* exact membership: a subset or a duplicated
    # donor would silently change the estimand and the pairing
    tids = _resolve_frozen_cohort(donors_npz.files, list(donors) if donors else None)
    if len(tids) != EXPECTED_DONORS or len(set(tids)) != EXPECTED_DONORS:
        raise RuntimeError(f"stage 2 requires exactly {EXPECTED_DONORS} unique donors")
    budget = plan(len(tids), STAGE2["n_realisations"], keep_recordings)
    disk = check_disk(budget["estimated_gb"], root)
    if dry_run:
        return {"stage2": STAGE2, "plan": budget, "disk": disk, "dry_run": True}

    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    wide_uv, wide_geometry, wide_ids, fs, gain, crop, margin = load_wide_background()
    meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id").to_dict("index")
    source_geometry = np.load(DONOR_GEOMETRY)
    crop_geometry = np.ascontiguousarray(wide_geometry[crop])
    channel_ids = np.asarray(wide_ids)[crop]

    trains = realisations(fs)
    frozen = {n: {"n": int(t.size), "sha256": truth_digest({"inj0": t})}
              for n, t in trains.items()}
    if len({v["sha256"] for v in frozen.values()}) != len(frozen):
        raise RuntimeError("two realisations are identical; the pairing would be degenerate")
    manifest = {**STAGE2, "plan": budget, "frozen_realisations": frozen}
    manifest_path = root / "prespec.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit(
                f"{manifest_path} differs from the frozen stage-2 manifest. This "
                "run is prespecified; a changed protocol needs a new schema, not "
                "an edited file."
            )
    else:
        _atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")

    rows = []
    for tid in tids:
        template, base_wide, _, _ = donor_placement(
            tid, donors_npz, meta, source_geometry, crop_geometry, margin)
        for name, train in trains.items():
            truth = {"inj0": train}
            contract = build_truth_contract(
                truth, injected=truth,
                admission={"schema": STAGE2["schema"], "rule": name,
                           "n_total": int(train.size), "n_admitted": int(train.size),
                           "counts_by_level_um": {"0.0": int(train.size)}},
                channel_ids=channel_ids, geometry=crop_geometry,
                crop=(crop.start, crop.stop))
            injected = inject_trajectory(
                wide_uv.copy(), template, train, fs=fs, base_channel=base_wide,
                trajectory=static_trajectory(),
                amplitude_scale=C2_PRESPEC["amplitude_scale"], template_id=tid,
                edge_guard_samples=C2_PRESPEC["template_prep"]["edge_guard_samples"])
            rec_dir = runs / f"{tid}_{name}"
            write_injected_recording(
                rec_dir, np.ascontiguousarray(injected[:, crop]),
                channel_positions=crop_geometry, fs=fs, gain_uv_per_count=gain,
                source_snippet_dir=str(_recording_dir()), name=f"{tid}_{name}")
            del injected

            for config in CANDIDATES:
                result = l1_run(rec_dir, sorter=config, truth=truth,
                                truth_contract=contract, out_root=runs / "_l1")
                obs = result["stage_observables"]
                eff = effective_settings({"summary": obs["sort_summary"],
                                          "sorter_params": obs.get("sort_request", {})})
                assert_applied_settings(config.label, eff, config.overrides)
                unit = result["score"]["primary"]["units"][0]
                guard = result["score"]["guardrails"]
                rows.append({
                    "template": tid, "realisation": name,
                    "phase": "phalf" if name.endswith("phalf") else "p0",
                    "n_events": int(train.size), "candidate": config.label,
                    "Th_universal": eff["Th_universal"], "Th_learned": eff["Th_learned"],
                    "accuracy": unit["accuracy"], "tp": unit["tp"], "fp": unit["fp"],
                    "fn": unit["fn"], "recovered": unit["recovered"],
                    "n_output_units_capturing": unit["n_output_units_capturing"],
                    "label_switches": unit["label_switches"],
                    "refractory_violation_median": guard.get("refractory_violation_median"),
                    "similar_pairs_per_good_unit": guard.get("similar_pairs_per_good_unit"),
                    "truth_sha256": contract["truth_sha256"][:12],
                })
            # The voltage is regenerable; the sorts are what the analysis needs.
            # Fail closed: a silently-failed delete would break the one-recording
            # peak the start-up disk guard assumes.
            if not keep_recordings:
                try:
                    shutil.rmtree(rec_dir)
                except OSError as exc:
                    raise RuntimeError(
                        f"could not delete {rec_dir}: {exc}. Recordings would "
                        "accumulate past the disk budget; stopping."
                    ) from exc
                if rec_dir.exists():
                    raise RuntimeError(f"{rec_dir} still present after deletion")
            free_gb = shutil.disk_usage(root).free / 1e9
            if free_gb < DISK_HEADROOM_GB:
                raise RuntimeError(
                    f"only {free_gb:.0f} GB free, below the {DISK_HEADROOM_GB} GB "
                    f"headroom, after {len(rows)} cells; stopping before the "
                    "volume fills"
                )
            _atomic_csv(pd.DataFrame(rows), root / "stage2.partial.csv")
        _atomic_write(root / "progress.json", json.dumps({
            "donors_done": tids[: tids.index(tid) + 1],
            "donors_remaining": tids[tids.index(tid) + 1:],
            "cells": len(rows),
            "free_gb": round(shutil.disk_usage(root).free / 1e9, 1),
        }, indent=2) + "\n")

    frame = pd.DataFrame(rows)
    _atomic_csv(frame, root / "stage2.csv")
    summary = {"stage2": STAGE2, "plan": budget, "disk": disk,
               "frozen_realisations": frozen, "n_cells": int(len(frame)),
               "n_donors": len(tids),
               "realisation_hashes": int(frame.truth_sha256.nunique())}
    _atomic_write(root / "summary.json",
                  json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--donors", nargs="*", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--keep-recordings", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="plan and disk check only")
    args = ap.parse_args()
    print(json.dumps(run(donors=args.donors, root=args.out_root,
                         keep_recordings=args.keep_recordings,
                         dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
