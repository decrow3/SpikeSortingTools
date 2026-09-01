"""Plan or run the drift-architecture sorter bake-off on an accepted recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.bakeoff import (
    CANDIDATES,
    accept_ks4_reference,
    build_bakeoff_plan,
    run_dartsort_challenger,
    run_kiasort_challenger,
    resolve_bakeoff_window,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-output-dir", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--run", choices=("ks4_no_motion", "dartsort_native", "kiasort")
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(CANDIDATES),
        default=[
            "ks4_no_motion",
            "dartsort_native",
            "kiasort",
            "si_motion_aware_peeler",
        ],
    )
    parser.add_argument("--dartsort-preprocessing", default="ibllikecmr")
    parser.add_argument("--dartsort-no-work-in-tmpdir", action="store_true")
    parser.add_argument("--kiasort-path", type=Path)
    parser.add_argument("--kiasort-python-executable", type=Path)
    parser.add_argument("--kiasort-numba-threads", type=int, default=2)
    parser.add_argument("--kiasort-channel-start-index", type=int, default=0)
    parser.add_argument("--kiasort-channel-count", type=int)
    parser.add_argument("--kiasort-config-json", type=Path)
    parser.add_argument("--kiasort-keep-intermediate", action="store_true")
    parser.add_argument("--window-name", default="full_recording")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.plan and args.run is None:
        raise SystemExit("Choose --plan or --run")
    root = args.rescue_output_dir
    recording_dir = root / "recording"
    recording_manifest = json.loads(
        (recording_dir / "rescue_recording_manifest.json").read_text()
    )
    window = resolve_bakeoff_window(
        recording_manifest,
        name=args.window_name,
        start_s=args.start_s,
        duration_s=args.duration_s,
    )
    bakeoff_dir = root / "sorter_bakeoff" / "windows" / window.directory_name
    window_args = {
        "window_name": args.window_name,
        "start_s": args.start_s,
        "duration_s": args.duration_s,
    }
    if args.plan:
        print(
            json.dumps(
                build_bakeoff_plan(recording_dir, args.candidates, **window_args),
                indent=2,
            )
        )
    if args.run == "ks4_no_motion":
        result = accept_ks4_reference(
            recording_dir,
            root / "kilosort4",
            bakeoff_dir / "ks4_no_motion",
            **window_args,
        )
        print(json.dumps(result, indent=2))
    elif args.run == "dartsort_native":
        result = run_dartsort_challenger(
            recording_dir,
            bakeoff_dir / "dartsort_native",
            preprocessing=args.dartsort_preprocessing,
            work_in_tmpdir=not args.dartsort_no_work_in_tmpdir,
            **window_args,
        )
        print(json.dumps(result, indent=2))
    elif args.run == "kiasort":
        overrides = (
            {}
            if args.kiasort_config_json is None
            else json.loads(args.kiasort_config_json.read_text())
        )
        if not isinstance(overrides, dict):
            raise ValueError("KIASORT config JSON must contain an object")
        kiasort_output_name = "kiasort"
        if args.kiasort_channel_count is not None:
            channel_end = (
                args.kiasort_channel_start_index + args.kiasort_channel_count
            )
            kiasort_output_name = (
                f"kiasort_channels_{args.kiasort_channel_start_index}_{channel_end}"
            )
        result = run_kiasort_challenger(
            recording_dir,
            bakeoff_dir / kiasort_output_name,
            kiasort_path=args.kiasort_path,
            python_executable=args.kiasort_python_executable,
            numba_threads=args.kiasort_numba_threads,
            channel_start_index=args.kiasort_channel_start_index,
            channel_count=args.kiasort_channel_count,
            config_overrides=overrides,
            keep_intermediate=args.kiasort_keep_intermediate,
            **window_args,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
