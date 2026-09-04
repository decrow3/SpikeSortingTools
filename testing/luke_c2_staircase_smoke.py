"""C2 v4 staircase smoke matrix — does the machinery show the expected behaviour?

Four cells on a small prespecified donor subset:

| arm                   | sorter        | asks |
|-----------------------|---------------|------|
| `static`              | `rescue`      | is the donor recovered with no motion at all? |
| `staircase`           | `rescue`      | does commensurate motion fragment or lose it? |
| `staircase`           | `rescue_rigid`| does KS4's own rigid correction improve that? |
| `staircase_corrected` | `rescue`      | does exact external correction return it to static? |

Because the staircase is interpolation-free (see
`luke_20250804_c2_staircase_positive_control.md`), any difference between these
cells is sorter behaviour, not forward-model damage. This is **engineering
validation of the machinery**, not a Luke-scale result: 40 µm is ~2x the largest
proposed Luke displacement and the motion is discontinuous.

Every cell scores the **identical admitted train** under one truth contract, on
identically cropped channels, and every sort's *saved effective* settings are
checked — the rescue baseline requests `nblocks=1` with `do_correction=False`
and KS4 forces the effective value to 0, so the requested parameter dict proves
nothing.

Run: `python testing/luke_c2_staircase_smoke.py [--donors D03 D01 ...]`
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from testing.ladder_inject import inject_trajectory, static_trajectory, write_injected_recording
from testing.ladder_l1 import l1_run
from testing.ladder_score import assert_paired_truth
from testing.ladder_sorter import NAMED_CONFIGS, check_effective_settings
from testing.luke_c2_staircase_control import (
    STAIRCASE,
    build_arms,
    expected_shift_channels,
    load_wide_background,
    staircase_admitted_truth,
    staircase_truth_contract,
)
from testing.luke_rescue_c2_drift_challenge import (
    DONOR_GEOMETRY,
    DONOR_MANIFEST,
    DONOR_TEMPLATES,
    PRESPEC,
    _recording_dir,
    _verify_donor_cohort,
    donor_base_channel,
    prepare_template,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "testing/outputs/luke_c2_staircase_smoke"

SMOKE = {
    "schema": "luke-c2-staircase-smoke-v1",
    "status": "engineering validation only; not a C2 result",
    # spans polarity, amplitude and static performance: two donors clean under
    # both configs, one that fails statically under legacy_style (D01) and one
    # that fails statically under rescue (D10), plus the amplitude floor (D03)
    "donors": ["D03", "D01", "D10", "D12"],
    "cells": [
        {"arm": "static", "sorter": "rescue"},
        {"arm": "staircase", "sorter": "rescue"},
        {"arm": "staircase", "sorter": "rescue_rigid"},
        {"arm": "staircase_corrected", "sorter": "rescue"},
        # Required control, not optional. `rescue_rigid` differs from `rescue`
        # in more than motion handling -- datashift re-registers the recording
        # and changes clustering even with no motion present. Without this cell,
        # "rigid correction rescued the moving arm" cannot be separated from
        # "rigid correction sorts this donor better regardless". Run 1 made that
        # concrete: D10 scored 0.956 on staircase+rescue_rigid but only 0.470 on
        # static+rescue, so the gain cannot all be motion recovery.
        {"arm": "static", "sorter": "rescue_rigid"},
    ],
    "expected_qualitative": {
        "static_rescue": "recovered",
        "staircase_rescue": "degraded or fragmented vs static",
        "staircase_rescue_rigid": "better than staircase+rescue if rigid correction works",
        "staircase_corrected_rescue": "close to static; the machinery's ceiling",
        "static_rescue_rigid": (
            "the control: how much of rescue_rigid's gain is motion recovery "
            "rather than a different sort of the same stationary neuron"
        ),
    },
}


def donor_placement(tid, donors, donor_meta, source_geometry, crop_geometry, margin):
    """Place the donor inside the crop, with room for the excursion at both edges."""
    template, peak_col = prepare_template(donors[tid])
    base_in_crop, peak_in_crop = donor_base_channel(
        template, peak_col, int(donor_meta[tid]["peak_channel"]),
        source_geometry, crop_geometry,
    )
    shift = expected_shift_channels(max(STAIRCASE["levels_um"]))
    width = int(template.shape[1])
    n_crop = crop_geometry.shape[0]
    if base_in_crop - shift < 0 or base_in_crop + width + shift > n_crop:
        raise RuntimeError(
            f"{tid} sits within {shift} channels of a crop edge; the excursion "
            "would move it out of the verified region"
        )
    return template, base_in_crop + margin, base_in_crop, peak_in_crop


def build_donor_arms(tid, wide_uv, wide_geometry, wide_ids, fs, crop, margin,
                     donors, donor_meta, source_geometry, train):
    """Inject once at rest into the wide strip, then warp the whole field."""
    crop_geometry = np.ascontiguousarray(wide_geometry[crop])
    template, base_wide, base_crop, peak_crop = donor_placement(
        tid, donors, donor_meta, source_geometry, crop_geometry, margin
    )
    injected_wide = inject_trajectory(
        wide_uv.copy(), template, train, fs=fs, base_channel=base_wide,
        trajectory=static_trajectory(), amplitude_scale=PRESPEC["amplitude_scale"],
        template_id=tid,
        edge_guard_samples=PRESPEC["template_prep"]["edge_guard_samples"],
    )
    arms = build_arms(injected_wide, wide_geometry, fs, crop=crop, margin=margin,
                      wide_channel_ids=wide_ids)
    arms["placement"] = {
        "base_channel_wide": int(base_wide), "base_channel_crop": int(base_crop),
        "peak_channel_crop": int(peak_crop), "template_width": int(template.shape[1]),
    }
    # the array that actually went into the voltage — the contract is built from
    # this, so injecting anything but the admitted train fails closed
    arms["injected_train"] = np.asarray(train, dtype=np.int64)
    return arms


def run(donors_requested=None, out_root=None, keep_recordings=False) -> dict:
    if str(OUTPUT).startswith("/mnt/"):
        raise ValueError("refusing to write the smoke matrix under /mnt")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _verify_donor_cohort()
    import pandas as pd

    out_root = Path(out_root) if out_root else OUTPUT / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    tids = list(donors_requested or SMOKE["donors"])

    wide_uv, wide_geometry, wide_ids, fs, gain, crop, margin = load_wide_background()
    donors = np.load(DONOR_TEMPLATES)
    donor_meta = pd.read_csv(DONOR_MANIFEST).set_index("template_id").to_dict("index")
    source_geometry = np.load(DONOR_GEOMETRY)

    regular = np.arange(
        int(PRESPEC["train"]["guard_s"] * fs),
        int(STAIRCASE["duration_s"] * fs) - int(PRESPEC["train"]["guard_s"] * fs),
        int(round(fs / PRESPEC["train"]["rate_hz"])),
        dtype=np.int64,
    )

    # Filter FIRST. Only the admitted array exists below this line, so the
    # boundary-straddling events never enter the voltage at all — they cannot
    # shape detection or template formation, not merely go unscored.
    truth, admission_record = staircase_admitted_truth(regular, fs)
    unit_id, admitted = next(iter(truth.items()))

    rows, contracts = [], {}
    for tid in tids:
        arms = build_donor_arms(tid, wide_uv, wide_geometry, wide_ids, fs, crop,
                                margin, donors, donor_meta, source_geometry,
                                admitted)
        contract = staircase_truth_contract(
            truth, {unit_id: arms["injected_train"]}, arms, admission_record
        )
        contracts[tid] = contract

        rec_dirs = {}
        for arm in ("static", "staircase", "staircase_corrected"):
            rec_dir = out_root / f"{tid}_{arm}"
            write_injected_recording(
                rec_dir, arms[arm], channel_positions=arms["geometry"], fs=fs,
                gain_uv_per_count=gain, source_snippet_dir=str(_recording_dir()),
                name=f"{tid}_{arm}",
            )
            rec_dirs[arm] = rec_dir
        del arms

        for cell in SMOKE["cells"]:
            label = cell["sorter"]
            config = NAMED_CONFIGS[label]
            result = l1_run(
                rec_dirs[cell["arm"]],
                sorter=None if label == "rescue" else config,
                truth=truth, truth_contract=contract,
                out_root=out_root / "_l1",
            )
            observables = result["stage_observables"]
            effective = check_effective_settings(label, {
                "summary": observables["sort_summary"],
                "sorter_params": observables.get("sort_request", {}),
            })
            unit = result["score"]["primary"]["units"][0]
            rows.append({
                "template": tid,
                "peak_uv": donor_meta[tid]["peak_uv"],
                "polarity": donor_meta[tid]["polarity"],
                "arm": cell["arm"], "sorter": label,
                "n_truth": unit["n_truth"],
                "accuracy": unit["accuracy"],
                "tp": unit["tp"], "fp": unit["fp"], "fn": unit["fn"],
                "n_output_units_capturing": unit["n_output_units_capturing"],
                "label_switches": unit["label_switches"],
                "recovered": unit["recovered"],
                "truth_sha256": result["score"]["truth_contract"]["truth_sha256"][:12],
                **{f"eff_{k}": v for k, v in effective.items() if k != "_sources"},
            })
        if not keep_recordings:
            for rec_dir in rec_dirs.values():
                shutil.rmtree(rec_dir, ignore_errors=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "smoke_matrix.csv", index=False)

    paired = {
        tid: assert_paired_truth(
            [contracts[tid]] * len(SMOKE["cells"]),
            labels=[f"{c['arm']}+{c['sorter']}" for c in SMOKE["cells"]],
        )
        for tid in tids
    }
    denominators = frame.groupby("template").n_truth.nunique().to_dict()
    summary = {
        "smoke": SMOKE,
        "donors": tids,
        "n_cells": int(len(frame)),
        "identical_denominator_per_donor": {
            tid: int(n) == 1 for tid, n in denominators.items()
        },
        "n_truth_per_donor": frame.groupby("template").n_truth.first().to_dict(),
        "paired_truth": paired,
        "effective_settings_ok": True,
        "matrix": rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--donors", nargs="*", default=None)
    ap.add_argument("--keep-recordings", action="store_true")
    args = ap.parse_args()
    summary = run(donors_requested=args.donors, keep_recordings=args.keep_recordings)
    print(json.dumps({k: v for k, v in summary.items() if k != "matrix"}, indent=2,
                     default=str))
    import pandas as pd
    print("\n" + pd.DataFrame(summary["matrix"]).to_string(index=False))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
