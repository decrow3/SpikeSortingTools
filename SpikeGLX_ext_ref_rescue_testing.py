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
    JobConfig,
    MotionBackend,
    MotionSidecarConfig,
    PIPELINE_VERSION,
    RescueConfig,
    build_motion_estimator_input,
    materialize_rescue_recording,
    phase_correct,
    rescue_kilosort4_overrides,
    run_kilosort4,
    run_motion_sidecar_safely,
    validate_accepted_recording,
    write_artifact_sidecar,
    write_motion_coordinate_sidecar,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--sort", action="store_true")
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument(
        "--motion-sidecar",
        action="store_true",
        help="Run the rigid DREDGE sidecar (already default with --prepare/--sort).",
    )
    motion_group.add_argument(
        "--no-motion-sidecar",
        action="store_true",
        help="Explicitly disable the default rigid DREDGE sidecar.",
    )
    parser.add_argument("--recompute-motion", action="store_true")
    parser.add_argument("--motion-strict", action="store_true")
    parser.add_argument("--motion-chunk-duration", default="2s")
    parser.add_argument(
        "--motion-split-half",
        action="store_true",
        help="Run the optional diagnostic split-half DREDGE audit.",
    )
    parser.add_argument("--artifact-sidecar", action="store_true")
    parser.add_argument(
        "--motion-field",
        type=Path,
        help="Qualified motion-field NPZ; writes a post-sort coordinate sidecar.",
    )
    parser.add_argument("--motion-gain", type=float, default=1.0)
    parser.add_argument("--motion-min-support", type=float, default=1.0)
    parser.add_argument("--motion-min-confidence", type=float, default=0.5)
    parser.add_argument("--motion-coordinate-chunk-spikes", type=int, default=1_000_000)
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
            "rigid_dredge_motion_sidecar_on_estimator_view",
            "exact_identity_route_to_sorter",
            "single_internal_kilosort_car_highpass_whitening",
        ],
        "disabled": [
            "external_sorter_filter",
            "external_sorter_reference",
            "external_voltage_motion_correction",
            "kilosort_internal_motion_correction",
            "cross_peel_claim_mask",
            "kilosort_batch_artifact_rejection",
            "nonrigid_dredge_estimation",
            "legacy_motion_cache_export",
        ],
        "motion_sidecar": {
            "enabled_by_default_with_prepare_or_sort": True,
            "explicitly_disabled": args.no_motion_sidecar,
            "config": MotionSidecarConfig(split_half=args.motion_split_half).as_dict(),
            "job_config": JobConfig(
                n_jobs=args.n_jobs,
                chunk_duration=args.motion_chunk_duration,
                progress_bar=True,
            ).as_kwargs(),
            "voltage_modified": False,
            "correction_policy_validated": False,
        },
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
    if not any(
        (
            args.prepare,
            args.sort,
            args.motion_sidecar,
            args.artifact_sidecar,
            args.motion_field,
        )
    ):
        if not args.plan:
            raise SystemExit(
                "Choose --plan, --prepare, --sort, --motion-sidecar, "
                "--artifact-sidecar, or --motion-field"
            )
        return
    recording_dir = output_dir / "recording"
    raw = None
    start_frame = end_frame = None
    bad_ids = None
    if any((args.prepare, args.artifact_sidecar)):
        raw = load_raw(args)
        start_frame, end_frame = requested_frames(raw, args)
        bad_ids = physical_channel_ids(raw, args.bad_channel)
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
    run_default_sidecar = (args.prepare or args.sort) and not args.no_motion_sidecar
    if args.motion_sidecar or run_default_sidecar:
        from spikeinterface.core import load_extractor

        accepted_recording = load_extractor(recording_dir)
        accepted_manifest = json.loads(
            (recording_dir / "rescue_recording_manifest.json").read_text()
        )
        validate_accepted_recording(recording_dir, accepted_manifest)
        try:
            estimator_recording = build_motion_estimator_input(
                accepted_recording,
                MotionSidecarConfig().estimator_input,
            )
            motion_backend = None
        except Exception as estimator_input_error:
            # Preserve a normal sidecar failure receipt and identity sorting even
            # when construction of the estimator-only view fails.
            estimator_recording = accepted_recording

            def fail_estimator_input(
                *args, _error=estimator_input_error, **kwargs
            ):
                raise RuntimeError(
                    f"Motion estimator input construction failed: {_error}"
                ) from _error

            motion_backend = MotionBackend(
                fail_estimator_input,
                fail_estimator_input,
                fail_estimator_input,
                {"estimator_input": "construction-failed"},
            )
        motion_result = run_motion_sidecar_safely(
            estimator_recording,
            recording_for_sorting=accepted_recording,
            cache_dir=output_dir / "motion",
            config=MotionSidecarConfig(split_half=args.motion_split_half),
            job_config=JobConfig(
                n_jobs=args.n_jobs,
                chunk_duration=args.motion_chunk_duration,
                progress_bar=True,
            ),
            recompute=args.recompute_motion,
            strict=args.motion_strict,
            backend=motion_backend,
            accepted_recording_manifest=accepted_manifest,
        )
        print(
            json.dumps(
                {
                    "motion_sidecar_status": motion_result.status,
                    "request_digest": motion_result.request_digest,
                    "cache_lineage": motion_result.cache_lineage,
                    "qc_status": motion_result.qc.status,
                    "recording_for_sorting": str(recording_dir),
                    "voltage_modified": False,
                },
                indent=2,
            )
        )
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
    if args.motion_field:
        result = write_motion_coordinate_sidecar(
            output_dir / "kilosort4",
            args.motion_field,
            output_dir / "motion_coordinates",
            gain=args.motion_gain,
            min_support=args.motion_min_support,
            min_confidence=args.motion_min_confidence,
            chunk_spikes=args.motion_coordinate_chunk_spikes,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
