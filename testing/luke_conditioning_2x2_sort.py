"""Run the missing cells of Luke's good-snippet conditioning 2x2.

Factors are saturation policy (legacy samplewise blanking versus unchanged
voltage) and channel-191 policy (interpolate/include versus exclude in
Kilosort). Two diagonal cells reuse completed sorts; only the two missing cells
run. Motion correction and the cross-template claim mask remain disabled.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import build_sorter_params
from testing.luke_two_axis_pilot import (
    CLAIM_OFF,
    DEFAULT_REVIEW,
    PILOTS,
    assert_gpu_and_patch,
    bad_channel_rows,
    score_pilot,
)


LEGACY_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_two_axis_pilot_imec1"
)
V2_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_conditioning_v2_pilot_imec1"
)
DEFAULT_OUTPUT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_conditioning_2x2_good_imec1"
)
PILOT = PILOTS["good_pre_shared"]


@dataclass(frozen=True)
class Cell:
    name: str
    saturation_policy: str
    channel_191_policy: str
    recording_path: Path
    completed_sort: Path | None = None


CELLS = {
    cell.name: cell
    for cell in (
        Cell(
            "blank_interpolate_include",
            "legacy_samplewise_prefilter_blank",
            "interpolate_and_include",
            LEGACY_ROOT / "recordings/good_pre_shared",
            LEGACY_ROOT
            / "sorts/good_pre_shared/single_ks_preprocessing_claim_off",
        ),
        Cell(
            "blank_exclude191",
            "legacy_samplewise_prefilter_blank",
            "exclude_in_kilosort",
            LEGACY_ROOT / "recordings/good_pre_shared",
        ),
        Cell(
            "unchanged_interpolate_include",
            "unchanged_voltage",
            "interpolate_and_include",
            V2_ROOT / "recordings/good_pre_shared",
        ),
        Cell(
            "unchanged_exclude191",
            "unchanged_voltage",
            "exclude_in_kilosort",
            V2_ROOT / "recordings/good_pre_shared",
            V2_ROOT / "sorts/good_pre_shared/single_ks_preprocessing_claim_off",
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--cell", action="append", choices=tuple(CELLS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def selected_cells(names: list[str] | None) -> list[Cell]:
    if names:
        return [CELLS[name] for name in dict.fromkeys(names)]
    return list(CELLS.values())


def cell_sort_path(cell: Cell, output_dir: Path) -> Path:
    return cell.completed_sort or output_dir / "sorts" / cell.name


def plan(cells: list[Cell], output_dir: Path) -> dict:
    return {
        "pilot": PILOT.name,
        "duration_s": PILOT.duration_s,
        "motion_correction": False,
        "claim_mask": "off",
        "positive_polarity_excess": {
            "status": "deferred_not_resolved",
            "follow_up": (
                "test Luke across sessions and acquisition rigs to distinguish "
                "session-specific contamination from a recurring setup issue"
            ),
        },
        "cells": [
            {
                **asdict(cell),
                "recording_path": str(cell.recording_path),
                "completed_sort": (
                    str(cell.completed_sort) if cell.completed_sort else None
                ),
                "sort_path": str(cell_sort_path(cell, output_dir)),
                "new_sort_required": cell.completed_sort is None,
            }
            for cell in cells
        ],
    }


def run_cell(cell: Cell, output_dir: Path) -> None:
    if cell.completed_sort is not None:
        print(f"Reusing {cell.name}: {cell.completed_sort}")
        return
    import spikeinterface.core as sc
    from spikeinterface.preprocessing import interpolate_bad_channels
    from spikeinterface.sorters import run_sorter

    target = cell_sort_path(cell, output_dir)
    result = target / "sorter_output/spike_times.npy"
    if result.exists():
        print(f"Reusing completed {cell.name}: {target}")
        return
    if target.exists():
        raise RuntimeError(f"Partial or ambiguous sort: {target}")
    recording = sc.load(cell.recording_path)
    if cell.channel_191_policy == "interpolate_and_include":
        recording = interpolate_bad_channels(recording, bad_channel_ids=[191])
        bad_rows = None
    else:
        bad_rows = bad_channel_rows(recording.get_channel_ids())
    params = build_sorter_params(CLAIM_OFF, bad_channels=bad_rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / f"{cell.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        with log_path.open("a") as log_file, contextlib.redirect_stdout(
            log_file
        ), contextlib.redirect_stderr(log_file):
            run_sorter(
                "kilosort4",
                recording,
                folder=str(target),
                verbose=True,
                remove_existing_folder=False,
                **params,
            )
    finally:
        root_logger.removeHandler(handler)
        handler.close()
    print(f"Completed {cell.name}; log: {log_path}")


def score_cells(
    cells: list[Cell], output_dir: Path, review_events: Path
) -> pd.DataFrame:
    rows = []
    for cell in cells:
        target = cell_sort_path(cell, output_dir)
        log = (
            (LEGACY_ROOT if cell.name == "blank_interpolate_include" else V2_ROOT)
            / "logs/good_pre_shared.log"
            if cell.completed_sort is not None
            else output_dir / "logs" / f"{cell.name}.log"
        )
        summary = score_pilot(
            PILOT,
            output_dir,
            review_events,
            10.0,
            result_override=target / "sorter_output",
            score_name=cell.name,
            log_override=log,
        )
        rows.append(
            {
                "cell": cell.name,
                "saturation_policy": cell.saturation_policy,
                "channel_191_policy": cell.channel_191_policy,
                **{key: value for key, value in summary.items() if key != "pilot"},
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "conditioning_2x2_scores.csv", index=False)
    indexed = frame.set_index("cell")
    metrics = [
        "n_final_spikes",
        "n_units",
        "n_ks_good",
        "median_contamination_pct",
        "cross_unit_coincidence_excess",
        "neural_unmatched_recovery_excess",
    ]
    contrasts = {
        "exclude_minus_interpolate_with_legacy_blanking": {
            metric: float(
                indexed.loc["blank_exclude191", metric]
                - indexed.loc["blank_interpolate_include", metric]
            )
            for metric in metrics
        },
        "exclude_minus_interpolate_with_unchanged_voltage": {
            metric: float(
                indexed.loc["unchanged_exclude191", metric]
                - indexed.loc["unchanged_interpolate_include", metric]
            )
            for metric in metrics
        },
        "unchanged_minus_blank_with_interpolation": {
            metric: float(
                indexed.loc["unchanged_interpolate_include", metric]
                - indexed.loc["blank_interpolate_include", metric]
            )
            for metric in metrics
        },
        "unchanged_minus_blank_with_exclusion": {
            metric: float(
                indexed.loc["unchanged_exclude191", metric]
                - indexed.loc["blank_exclude191", metric]
            )
            for metric in metrics
        },
    }
    decision = {
        "decision": "conditioning_not_yet_selected_evidence_conflict_requires_harder_window",
        "good_snippet_result": (
            "All cells recover all 14 reviewed unmatched-neural events. On this sparsely "
            "saturated window, legacy blanking and interpolation/inclusion each improve "
            "KS-good yield and median contamination relative to their matched alternative."
        ),
        "conflicting_upstream_evidence": (
            "Saturation-enriched raw-voltage audits show that samplewise prefilter blanking "
            "and interval interpolation create false peaks, event-density expansion, and "
            "covariance damage. The easy good window cannot adjudicate that risk."
        ),
        "next_gate": (
            "Run unchanged-voltage plus interpolate/include first in neutral and pathological "
            "120-second windows. Compare with their legacy controls; expand to full 2x2 only "
            "where the diagonal comparison is ambiguous."
        ),
        "positive_polarity_excess": {
            "status": "deferred_not_resolved",
            "follow_up": (
                "test multiple Luke sessions and another acquisition rig/setup before "
                "deciding whether the excess is session-specific or recurrent"
            ),
        },
        "factor_contrasts": contrasts,
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return frame


def main() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-conditioning-2x2-numba")
    args = parse_args()
    cells = selected_cells(args.cell)
    current_plan = plan(cells, args.output_dir)
    if args.plan_only:
        print(json.dumps(current_plan, indent=2))
        return
    if not (args.run or args.score):
        raise SystemExit("Choose --plan-only, --run, or --score")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(json.dumps(current_plan, indent=2) + "\n")
    if args.run:
        assert_gpu_and_patch()
        for cell in cells:
            run_cell(cell, args.output_dir)
    if args.score:
        print(score_cells(cells, args.output_dir, args.review_events).to_string(index=False))


if __name__ == "__main__":
    main()
