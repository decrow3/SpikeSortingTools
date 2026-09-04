"""C2 v4 — the paired static-vs-moving drift challenge, frozen and executable.

This is a **distinct runner**, not v3 pointed at a renamed directory: its own
schema, its own output namespace, its own cache root, and fail-closed checks on
every input it depends on. It refuses to read or write anything under a v3 tree,
and refuses any output root under `/mnt`.

What v4 changes relative to the void v3 run
-------------------------------------------
* **Luke-calibrated rigid ramps** (5 / 11 / 22 µm) replace v3's stale
  15 / 40 / osc-20 set, per decision 0013.
* **`rescue_rigid`** isolates what `nblocks=1` buys, which `legacy_style` cannot
  because it moves correction and detection thresholds together. `legacy_style`
  is retained as the operational comparator only.
* **A stationary `rescue_rigid` control** — datashift re-registers a recording
  and changes clustering with no motion present, so a moving-arm gain is not
  attributable to motion recovery without it. The correction effect is the
  interaction, not a single difference:

      (moved_rigid − moved_rescue) − (static_rigid − static_rescue)

* **Every ramp arm is labelled forward-model confounded.** The operator
  calibration measured 10–32 % mean peak attenuation across these excursions,
  and the exact inverse costs a further ~6–10 points, biasing the correction
  contrast against correction. Ramp results may not be read without it.
* **A lattice-commensurate staircase positive control**, interpolation-free,
  reported separately — it answers whether the machinery can show a correction
  benefit at all, never whether correction helps at Luke scale.
* **Truth is filtered before injection** and bound by a truth contract whose
  "filtered before injection" claim is derived from the array actually injected.
* **Contrast-specific static qualification**, with a common primary cohort,
  all-donor results, and a separately labelled operator-qualified sensitivity
  analysis. Per-arm exclusion is forbidden: it makes each magnitude run on a
  different, progressively easier cohort.

Modes
-----
``--verify``  hashes, configs and prespec only; no sorting, seconds.
``--smoke``   the frozen smoke subset; engineering validation.
``--full``    all 14 donors.

Output root is host-configurable (``--out-root`` or ``$LUKE_C2_V4_ROOT``) so each
host writes its own namespace.
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
from testing.ladder_score import assert_paired_truth, build_truth_contract
from testing.ladder_sorter import NAMED_CONFIGS, check_effective_settings
from testing.luke_c2_staircase_control import (
    STAIRCASE,
    build_arms,
    expected_shift_channels,
    load_wide_background,
    staircase_admitted_truth,
    staircase_um,
)
from testing.luke_c2_staircase_smoke import donor_placement
from testing.luke_rescue_c2_drift_challenge import (
    DONOR_GEOMETRY,
    DONOR_MANIFEST,
    DONOR_TEMPLATES,
    PRESPEC as V3_PRESPEC,
    _recording_dir,
    _sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_c2_drift_challenge_v3"
DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_rescue_c2_drift_challenge_v4"

SCHEMA = "luke-rescue-c2-drift-challenge-v4"

PRESPEC = {
    "schema": SCHEMA,
    "frozen": "2026-09-04",
    "supersedes": "luke-rescue-c2-drift-challenge-v3 (void: decision 0014 scorer defect)",
    "question": (
        "At the rigid motion Luke imec0 actually experiences, how much neuron "
        "recovery does no-correction KS4 lose, and how much does standard rigid "
        "correction recover — after accounting for the forward model's own cost?"
    ),
    "probe": V3_PRESPEC["probe"],
    "background": V3_PRESPEC["background"],
    "template_prep": V3_PRESPEC["template_prep"],
    "train": V3_PRESPEC["train"],
    "amplitude_scale": V3_PRESPEC["amplitude_scale"],
    "spatial": {
        "margin_channels": STAIRCASE["spatial_margin_channels"],
        "policy": (
            "warp a wider strip and crop, so every sorter channel has a defined "
            "source site; all arms of a condition share one crop"
        ),
    },
    "motion_operator": {
        "spatial_interpolation_method": "kriging",
        "sigma_um": 20.0,
        "border_mode": "force_extrapolate",
        "bin_s": STAIRCASE["bin_s"],
        "trajectory_units": "um",
        "sign_convention": "forward warp sign=-1, exact inverse sign=+1",
        "calibration": "luke-c2-operator-calibration-v1",
        "measured_forward_model_attenuation": {
            "ramp_mean_peak_retention": {"5": 0.912, "11": 0.812, "22": 0.680},
            "exact_inverse_round_trip_rel_rms": {"5": 0.162, "11": 0.302, "22": 0.433},
            "exact_inverse_amplitude_cost": {"5": -0.064, "11": -0.098, "22": -0.094},
            "note": "a registration reference, not a performance ceiling",
        },
    },
    "conditions": {
        "ramp_5um": {"kind": "rigid_ramp", "total_um": 5.0, "family": "luke_calibrated",
                     "forward_model_confounded": True},
        "ramp_11um": {"kind": "rigid_ramp", "total_um": 11.0, "family": "luke_calibrated",
                      "forward_model_confounded": True},
        "ramp_22um": {"kind": "rigid_ramp", "total_um": 22.0, "family": "luke_calibrated",
                      "forward_model_confounded": True},
        "staircase_40um": {"kind": "commensurate_staircase", "family": "positive_control",
                           "forward_model_confounded": False,
                           "reported": "separately from the Luke-calibrated dose-response"},
    },
    # Each condition is self-contained: its own static baseline, built from the
    # same injected voltage and scored against the SAME truth train. A shared
    # standalone static condition cannot serve them all — the staircase admits
    # 687 events and the ramps carry all 708, so a shared baseline would compare
    # the staircase's moving arms against a different denominator, which is the
    # exact error the truth contract exists to prevent.
    "arms_per_condition": ["static", "moved", "moved_corrected"],
    "baseline_policy": (
        "within-condition: the static arm of a condition shares that condition's "
        "truth train, injected voltage and crop"
    ),
    "sorters": {
        "rescue": "primary, no correction",
        "rescue_rigid": "the isolated internal-correction contrast",
        "legacy_style": "operational comparator only; confounds correction with thresholds",
    },
    "required_stationary_control": {"arm": "static", "sorter": "rescue_rigid"},
    "correction_effect": (
        "(moved_rigid - moved_rescue) - (static_rigid - static_rescue); compare "
        "the motion penalty within each configuration before comparing configurations"
    ),
    "static_qualification": {
        "accuracy_min": 0.8,
        "rule": (
            "contrast-specific: a donor enters a given contrast if it qualifies "
            "under both sorter configs in that contrast"
        ),
        "forbidden": "per-arm exclusion that varies the cohort across magnitudes",
        "reported_cohorts": ["common_primary", "all_donor", "operator_qualified_sensitivity"],
    },
    "scoring": {
        "score_schema": "luke-ladder-score-sort-v3",
        "truth_contract_schema": "luke-ladder-truth-contract-v1",
        "accuracy_gate": 0.8,
        "capture_frac": 0.05,
        "chance_margin": 3.0,
        "tol_ms": 0.5,
        "truth_order": "filtered before injection; attestation derived from the injected array",
    },
    "isolation": {
        "forbids_v3_outputs": True,
        "forbids_mnt_outputs": True,
        "cache_root": "<out_root>/_l1 — never a v3 cache leaf",
    },
}


class V4IsolationError(RuntimeError):
    """An output root or cache that would mix v4 with v3, or write under /mnt."""


def output_root(explicit: str | Path | None = None) -> Path:
    root = Path(explicit or os.environ.get("LUKE_C2_V4_ROOT") or DEFAULT_OUTPUT)
    resolved = root.resolve()
    if str(resolved).startswith("/mnt/"):
        raise V4IsolationError(f"refusing a v4 output root under /mnt: {resolved}")
    v3 = V3_OUTPUT.resolve()
    if resolved == v3 or v3 in resolved.parents or resolved in v3.parents:
        raise V4IsolationError(
            f"v4 output root {resolved} overlaps the v3 tree {v3}; v4 may not "
            "read, write or reuse any v3 result or sort cache"
        )
    return root


def freeze_prespec(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prespec.json"
    if path.exists():
        if json.loads(path.read_text()) != PRESPEC:
            raise SystemExit(
                f"{path} differs from the frozen v4 PRESPEC. C2 is run-once; "
                "a changed protocol needs a new schema, not an edited file."
            )
    else:
        path.write_text(json.dumps(PRESPEC, indent=2) + "\n")
    return path


def prespec_digest() -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(PRESPEC, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def trajectory_for(name: str):
    """A µm-valued trajectory for a frozen condition name."""
    spec = PRESPEC["conditions"][name]
    if spec["kind"] == "static":
        return None
    if spec["kind"] == "commensurate_staircase":
        return staircase_um
    if spec["kind"] == "rigid_ramp":
        total = float(spec["total_um"])
        duration = float(V3_PRESPEC["background"]["duration_s"])
        return lambda t, total=total, duration=duration: (
            total * np.asarray(t, dtype=float) / duration
        )
    raise ValueError(f"unknown condition kind {spec['kind']!r}")


def verify(root: Path | None = None) -> dict:
    """Every hash and config check, with no sorting. Fails closed."""
    root = output_root(root)
    checks: dict = {"output_root": str(root), "prespec_sha256": prespec_digest()}

    expected = PRESPEC["template_prep"]
    for path, key in (
        (DONOR_TEMPLATES, "templates_sha256"),
        (DONOR_MANIFEST, "manifest_sha256"),
        (DONOR_GEOMETRY, "source_geometry_sha256"),
    ):
        observed = _sha256(path) if path.exists() else None
        if observed != expected[key]:
            raise V4IsolationError(f"donor cohort input changed or missing: {path}")
    checks["donor_cohort"] = "verified"

    for name in PRESPEC["conditions"]:
        fn = trajectory_for(name)
        if fn is None:
            continue
        probe = np.linspace(0.0, float(V3_PRESPEC["background"]["duration_s"]), 9)
        values = np.asarray(fn(probe), dtype=float)
        spec = PRESPEC["conditions"][name]
        if spec["kind"] == "rigid_ramp":
            if not np.isclose(values[-1], spec["total_um"]):
                raise V4IsolationError(
                    f"{name} reaches {values[-1]} µm, not {spec['total_um']}"
                )
        else:
            if not set(np.unique(values)) <= set(STAIRCASE["levels_um"]):
                raise V4IsolationError(f"{name} visits non-commensurate levels")
    checks["trajectories"] = "verified"

    for label in PRESPEC["sorters"]:
        if label not in NAMED_CONFIGS:
            raise V4IsolationError(f"sorter config {label!r} is not registered")
    rescue, rigid = NAMED_CONFIGS["rescue"], NAMED_CONFIGS["rescue_rigid"]
    if rescue.digest == rigid.digest:
        raise V4IsolationError("rescue and rescue_rigid share a cache digest")
    for threshold in ("Th_universal", "Th_learned"):
        if rigid.params()[threshold] != rescue.params()[threshold]:
            raise V4IsolationError(
                "rescue_rigid must not change detection thresholds; it would "
                "stop isolating what nblocks=1 buys"
            )
    checks["sorter_configs"] = {
        label: NAMED_CONFIGS[label].digest[:12] for label in PRESPEC["sorters"]
    }
    control = PRESPEC["required_stationary_control"]
    if control["sorter"] not in PRESPEC["sorters"]:
        raise V4IsolationError("the stationary control names an unregistered sorter")
    checks["stationary_control"] = control

    checks["prespec_path"] = str(freeze_prespec(root))
    checks["v3_isolation"] = "output root does not overlap the v3 tree"
    return checks


def run(mode: str = "smoke", root: Path | None = None, donors=None,
        conditions=None, keep_recordings: bool = False) -> dict:
    root = output_root(root)
    checks = verify(root)
    import pandas as pd

    tids = list(donors or (
        ["D03", "D01", "D10", "D12"] if mode == "smoke"
        else sorted(np.load(DONOR_TEMPLATES).files)
    ))
    names = list(conditions or (
        ["static", "ramp_11um", "staircase_40um"] if mode == "smoke"
        else list(PRESPEC["conditions"])
    ))
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    wide_uv, wide_geometry, wide_ids, fs, gain, crop, margin = load_wide_background()
    donors_npz = np.load(DONOR_TEMPLATES)
    donor_meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id").to_dict("index")
    source_geometry = np.load(DONOR_GEOMETRY)
    crop_geometry = np.ascontiguousarray(wide_geometry[crop])

    regular = np.arange(
        int(PRESPEC["train"]["guard_s"] * fs),
        int(V3_PRESPEC["background"]["duration_s"] * fs)
        - int(PRESPEC["train"]["guard_s"] * fs),
        int(round(fs / PRESPEC["train"]["rate_hz"])),
        dtype=np.int64,
    )
    # Filter FIRST, and only for the condition that has an exact subset. The
    # ramps have none — every spike sits at a fractional offset — so they carry
    # the full train and their attenuation is recorded, not filtered away.
    staircase_truth, staircase_admission = staircase_admitted_truth(regular, fs)
    full_truth = {"inj0": regular}
    full_admission = {
        "schema": SCHEMA, "rule": "all events; a ramp has no exact subset",
        "n_total": int(regular.size), "n_admitted": int(regular.size),
        "counts_by_level_um": {"fractional": int(regular.size)},
    }

    rows, contracts = [], {}
    for tid in tids:
        template, base_wide, base_crop, peak_crop = donor_placement(
            tid, donors_npz, donor_meta, source_geometry, crop_geometry, margin
        )
        for name in names:
            spec = PRESPEC["conditions"][name]
            is_staircase = spec["kind"] == "commensurate_staircase"
            truth = staircase_truth if is_staircase else full_truth
            admission = staircase_admission if is_staircase else full_admission
            injected_train = truth["inj0"]

            injected_wide = inject_trajectory(
                wide_uv.copy(), template, injected_train, fs=fs,
                base_channel=base_wide, trajectory=static_trajectory(),
                amplitude_scale=PRESPEC["amplitude_scale"], template_id=tid,
                edge_guard_samples=PRESPEC["template_prep"]["edge_guard_samples"],
            )
            labels = ("static", "moved", "moved_corrected")
            arms = build_arms(
                injected_wide, wide_geometry, fs, crop=crop, margin=margin,
                wide_channel_ids=wide_ids, trajectory_fn=trajectory_for(name),
                labels=labels,
            )
            arm_names = list(labels)
            del injected_wide

            contract = build_truth_contract(
                truth, injected={"inj0": injected_train}, admission=admission,
                channel_ids=arms["channel_ids"], geometry=arms["geometry"],
                crop=(crop.start, crop.stop),
            )
            contracts[f"{tid}:{name}"] = contract

            rec_dirs = {}
            for arm in arm_names:
                rec_dir = runs / f"{tid}_{name}_{arm}"
                write_injected_recording(
                    rec_dir, arms[arm], channel_positions=arms["geometry"], fs=fs,
                    gain_uv_per_count=gain, source_snippet_dir=str(_recording_dir()),
                    name=f"{tid}_{name}_{arm}",
                )
                rec_dirs[arm] = rec_dir
            del arms

            for arm in arm_names:
                for label in PRESPEC["sorters"]:
                    if arm == "moved_corrected" and label != "rescue":
                        continue  # the registration reference needs only rescue
                    result = l1_run(
                        rec_dirs[arm],
                        sorter=None if label == "rescue" else NAMED_CONFIGS[label],
                        truth=truth, truth_contract=contract,
                        out_root=runs / "_l1",
                    )
                    obs = result["stage_observables"]
                    effective = check_effective_settings(label, {
                        "summary": obs["sort_summary"],
                        "sorter_params": obs.get("sort_request", {}),
                    })
                    unit = result["score"]["primary"]["units"][0]
                    rows.append({
                        "template": tid, "condition": name,
                        "family": spec["family"],
                        "forward_model_confounded": spec.get(
                            "forward_model_confounded", False),
                        "arm": arm, "sorter": label,
                        "n_truth": unit["n_truth"], "accuracy": unit["accuracy"],
                        "tp": unit["tp"], "fp": unit["fp"], "fn": unit["fn"],
                        "n_output_units_capturing": unit["n_output_units_capturing"],
                        "label_switches": unit["label_switches"],
                        "recovered": unit["recovered"],
                        "truth_sha256": contract["truth_sha256"][:12],
                        **{f"eff_{k}": v for k, v in effective.items() if k != "_sources"},
                    })
            if not keep_recordings:
                for rec_dir in rec_dirs.values():
                    shutil.rmtree(rec_dir, ignore_errors=True)

        # Checkpoint after every donor: a 10 h unattended run must not lose
        # everything to a failure in its last hour. Sorts are content-cached, so
        # a resumed run reuses them, but the warps are not free.
        pd.DataFrame(rows).to_csv(root / "c2_v4.partial.csv", index=False)
        (root / "progress.json").write_text(json.dumps({
            "donors_done": tids[: tids.index(tid) + 1],
            "donors_remaining": tids[tids.index(tid) + 1:],
            "cells_so_far": len(rows),
        }, indent=2) + "\n")

    frame = pd.DataFrame(rows)
    frame.to_csv(root / "c2_v4.csv", index=False)
    summary = {
        "prespec": PRESPEC,
        "checks": checks,
        "mode": mode,
        "donors": tids,
        "conditions": names,
        "n_cells": int(len(frame)),
        "paired_truth": {
            key: assert_paired_truth(
                [contract] * 3, labels=["static", "moved", "moved_corrected"]
            )
            for key, contract in contracts.items()
        },
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="hashes and configs only")
    mode.add_argument("--smoke", action="store_true", help="the frozen smoke subset")
    mode.add_argument("--full", action="store_true", help="all 14 donors")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--donors", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--keep-recordings", action="store_true")
    args = ap.parse_args()
    if args.verify or not (args.smoke or args.full):
        print(json.dumps(verify(args.out_root), indent=2, default=str))
        return
    summary = run(mode="full" if args.full else "smoke", root=args.out_root,
                  donors=args.donors, conditions=args.conditions,
                  keep_recordings=args.keep_recordings)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("rows", "prespec")}, indent=2, default=str))


if __name__ == "__main__":
    main()
