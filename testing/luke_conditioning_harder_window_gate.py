"""Test the least-destructive conditioning candidate in harder Luke windows.

For neutral and pathological 120-second snippets, compare the existing legacy
blank+interpolate sort with unchanged voltage plus channel-191 interpolation.
Motion correction and the cross-template claim mask remain disabled.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import build_sorter_params
from testing.luke_two_axis_pilot import (
    CLAIM_OFF,
    DEFAULT_REVIEW,
    PIPELINE_ROOT,
    PILOTS,
    assert_gpu_and_patch,
    score_pilot,
)


LEGACY_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_two_axis_pilot_imec1"
)
V2_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_conditioning_v2_pilot_imec1"
)
DEFAULT_OUTPUT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_conditioning_harder_gate_imec1"
)


@dataclass(frozen=True)
class GateWindow:
    name: str
    legacy_sort: Path
    legacy_log: Path


WINDOWS = {
    window.name: window
    for window in (
        GateWindow(
            "neutral_template",
            LEGACY_ROOT
            / "sorts/neutral_template/single_ks_preprocessing_claim_off",
            LEGACY_ROOT / "logs/neutral_template.log",
        ),
        GateWindow(
            "pathological",
            PIPELINE_ROOT / "upstream_sorter_ablation/sorts/single_ks_preprocessing",
            PIPELINE_ROOT / "upstream_sorter_ablation/logs/single_ks_preprocessing.log",
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--window", action="append", choices=tuple(WINDOWS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    return parser.parse_args()


def selected_windows(names: list[str] | None) -> list[GateWindow]:
    if names:
        return [WINDOWS[name] for name in dict.fromkeys(names)]
    return list(WINDOWS.values())


def candidate_sort(window: GateWindow, output_dir: Path) -> Path:
    return output_dir / "sorts" / window.name / "unchanged_interpolate_include"


def make_plan(windows: list[GateWindow], output_dir: Path) -> dict:
    return {
        "motion_correction": False,
        "claim_mask": "off",
        "candidate": "phase_shift_then_unchanged_voltage_interpolate_191_include",
        "windows": [
            {
                "name": window.name,
                "duration_s": PILOTS[window.name].duration_s,
                "recording": str(V2_ROOT / "recordings" / window.name),
                "legacy_sort": str(window.legacy_sort),
                "candidate_sort": str(candidate_sort(window, output_dir)),
            }
            for window in windows
        ],
        "positive_polarity_excess": {
            "status": "deferred_not_resolved",
            "follow_up": "cross-session and cross-rig recurrence audit",
        },
    }


def run_candidate(window: GateWindow, output_dir: Path) -> None:
    import spikeinterface.core as sc
    from spikeinterface.preprocessing import interpolate_bad_channels
    from spikeinterface.sorters import run_sorter

    target = candidate_sort(window, output_dir)
    if (target / "sorter_output/spike_times.npy").exists():
        print(f"Reusing completed {window.name}: {target}")
        return
    if target.exists():
        raise RuntimeError(f"Partial or ambiguous sort: {target}")
    recording = sc.load(V2_ROOT / "recordings" / window.name)
    recording = interpolate_bad_channels(recording, bad_channel_ids=[191])
    params = build_sorter_params(CLAIM_OFF)
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "logs" / f"{window.name}.log"
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
    print(f"Completed {window.name}; log: {log_path}")


def score_windows(
    windows: list[GateWindow], output_dir: Path, review_events: Path
) -> pd.DataFrame:
    rows = []
    for window in windows:
        pilot = PILOTS[window.name]
        conditions = (
            ("legacy_blank_interpolate", window.legacy_sort, window.legacy_log),
            (
                "unchanged_interpolate",
                candidate_sort(window, output_dir),
                output_dir / "logs" / f"{window.name}.log",
            ),
        )
        for condition, sort, log in conditions:
            summary = score_pilot(
                pilot,
                output_dir,
                review_events,
                10.0,
                result_override=sort / "sorter_output",
                score_name=f"{window.name}_{condition}",
                log_override=log,
            )
            rows.append(
                {
                    "window": window.name,
                    "condition": condition,
                    **{key: value for key, value in summary.items() if key != "pilot"},
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "harder_window_scores.csv", index=False)
    indexed = frame.set_index(["window", "condition"])
    metrics = [
        "n_final_spikes",
        "n_units",
        "n_ks_good",
        "median_contamination_pct",
        "cross_unit_coincidence_excess",
        "neural_unmatched_recovery_excess",
    ]
    contrasts = {
        window.name: {
            metric: float(
                indexed.loc[(window.name, "unchanged_interpolate"), metric]
                - indexed.loc[(window.name, "legacy_blank_interpolate"), metric]
            )
            for metric in metrics
        }
        for window in windows
    }
    decision = {
        "decision": "pending_interpretation",
        "contrast_definition": "unchanged_interpolate minus legacy_blank_interpolate",
        "contrasts": contrasts,
        "positive_polarity_excess": {
            "status": "deferred_not_resolved",
            "follow_up": "cross-session and cross-rig recurrence audit",
        },
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return frame


def main() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-conditioning-harder-numba")
    args = parse_args()
    windows = selected_windows(args.window)
    current_plan = make_plan(windows, args.output_dir)
    if args.plan_only:
        print(json.dumps(current_plan, indent=2))
        return
    if not (args.run or args.score):
        raise SystemExit("Choose --plan-only, --run, or --score")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(json.dumps(current_plan, indent=2) + "\n")
    if args.run:
        assert_gpu_and_patch()
        for window in windows:
            run_candidate(window, args.output_dir)
    if args.score:
        print(score_windows(windows, args.output_dir, args.review_events).to_string(index=False))


if __name__ == "__main__":
    main()
