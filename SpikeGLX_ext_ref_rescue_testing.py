"""Prepare and run the minimal external-reference rescue pipeline.

Examples
--------
Inspect the frozen plan without writing data::

    python SpikeGLX_ext_ref_rescue_testing.py --data-dir /path/to/run --stream-id imec1.ap --plan

Materialize the tested no-motion recording and run Kilosort 4::

    python SpikeGLX_ext_ref_rescue_testing.py --data-dir /path/to/run --stream-id imec1.ap --prepare --sort
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline import (
    PIPELINE_VERSION,
    RescueConfig,
    materialize_rescue_recording,
    phase_correct,
    rescue_kilosort4_overrides,
    run_kilosort4,
    write_artifact_sidecar,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--artifact-sidecar", action="store_true")
    parser.add_argument(
        "--bad-channel",
        action="append",
        type=int,
        default=None,
        help="Physical channel number; repeat as needed. Omit to use frozen metrics.",
    )
    parser.add_argument("--recompute-channel-metrics", action="store_true")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument(
        "--duration-s",
        type=float,
        help="Optional bounded smoke-test duration; omit for the full recording.",
    )
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--chunk-duration", default="10s")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def default_output_dir(data_dir: Path, stream_id: str) -> Path:
    session = data_dir.resolve().name
    stream = stream_id.split(".", 1)[0]
    return data_dir.resolve().parent / f"rescue_pipeline_results_{session}_{stream}"


def physical_channel_ids(recording, requested: list[int] | None):
    if requested is None:
        return None
    available = recording.get_channel_ids().tolist()
    resolved = []
    for physical in requested:
        matches = [
            channel_id
            for channel_id in available
            if str(channel_id) == str(physical)
            or str(channel_id).endswith(f"AP{physical}")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Physical channel {physical} resolved to {matches}; available IDs are not unambiguous"
            )
        resolved.append(matches[0])
    return resolved


def load_raw(args: argparse.Namespace):
    import spikeinterface.full as si

    raw = si.read_spikeglx(
        folder_path=args.data_dir,
        load_sync_channel=False,
        stream_id=args.stream_id,
    )
    if args.start_s < 0 or (args.duration_s is not None and args.duration_s <= 0):
        raise ValueError("Requested time range must be positive")
    return raw


def requested_frames(raw, args: argparse.Namespace) -> tuple[int, int]:
    fs = float(raw.get_sampling_frequency())
    start = int(round(args.start_s * fs))
    stop = (
        int(raw.get_num_samples())
        if args.duration_s is None
        else start + int(round(args.duration_s * fs))
    )
    if start < 0 or stop > raw.get_num_samples() or start >= stop:
        raise ValueError("Requested time range falls outside the recording")
    return start, stop


def plan_payload(
    args: argparse.Namespace, output_dir: Path, config: RescueConfig
) -> dict:
    overrides = rescue_kilosort4_overrides()
    return {
        "pipeline_version": PIPELINE_VERSION,
        "source_folder": str(args.data_dir.resolve()),
        "stream_id": args.stream_id,
        "output_dir": str(output_dir),
        "time_range": {"start_s": args.start_s, "duration_s": args.duration_s},
        "config": config.as_dict(),
        "physical_bad_channels": args.bad_channel,
        "graph": [
            "neuropixels_phase_correction_if_present",
            "samplewise_bilateral_blanking_500uv",
            "bad_channel_interpolation",
            "materialize_int16",
            "single_internal_kilosort_car_highpass_whitening",
        ],
        "disabled": [
            "external_filter",
            "external_reference",
            "external_voltage_motion_correction",
            "kilosort_internal_motion_correction",
            "cross_peel_claim_mask",
            "kilosort_batch_artifact_rejection",
        ],
        "sorter_overrides": {
            key: ("Infinity" if value == float("inf") else value)
            for key, value in overrides.items()
        },
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spikeglx-rescue-numba-cache")
    output_dir = args.output_dir or default_output_dir(args.data_dir, args.stream_id)
    config = RescueConfig(
        materialize_n_jobs=args.n_jobs,
        materialize_chunk_duration=args.chunk_duration,
    )
    if args.plan:
        print(json.dumps(plan_payload(args, output_dir, config), indent=2))
    if not any((args.prepare, args.sort, args.artifact_sidecar)):
        if not args.plan:
            raise SystemExit("Choose --plan, --prepare, --sort, or --artifact-sidecar")
        return
    raw = load_raw(args)
    start_frame, end_frame = requested_frames(raw, args)
    bad_ids = physical_channel_ids(raw, args.bad_channel)
    recording_dir = output_dir / "recording"
    if args.prepare:
        _, manifest = materialize_rescue_recording(
            raw,
            recording_dir,
            source_folder=args.data_dir,
            stream_id=args.stream_id,
            config=config,
            bad_channel_ids=bad_ids,
            recompute_channel_metrics=args.recompute_channel_metrics,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        print(json.dumps(manifest, indent=2))
    if args.sort:
        manifest = run_kilosort4(recording_dir, output_dir / "kilosort4")
        print(json.dumps(manifest, indent=2))
    if args.artifact_sidecar:
        if bad_ids is None:
            recording_manifest = json.loads(
                (recording_dir / "rescue_recording_manifest.json").read_text()
            )
            available = raw.get_channel_ids().tolist()
            bad_ids = [
                channel_id
                for channel_id in available
                if str(channel_id) in recording_manifest["bad_channel_ids"]
            ]
        phase_recording = phase_correct(raw)
        if start_frame != 0 or end_frame != raw.get_num_samples():
            phase_recording = phase_recording.frame_slice(
                start_frame=start_frame, end_frame=end_frame
            )
        result = write_artifact_sidecar(
            phase_recording,
            output_dir / "artifacts/raw_over_500uv.h5",
            threshold_uv=config.saturation_threshold_uv,
            excluded_channel_ids=bad_ids,
            chunk_duration_s=args.chunk_duration,
            n_jobs=args.n_jobs,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
