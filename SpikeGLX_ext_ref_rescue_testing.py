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
import hashlib
import json
import os
from pathlib import Path

from pipeline import (
    JobConfig,
    MotionBackend,
    MotionSidecarConfig,
    PIPELINE_VERSION,
    PRODUCTION_UV_PREFIX,
    RescueConfig,
    build_motion_estimator_input,
    materialize_rescue_recording,
    phase_correct,
    production_environment_contract,
    rescue_kilosort4_overrides,
    run_kilosort4,
    run_motion_sidecar_safely,
    validate_accepted_recording,
    validate_production_environment,
    write_artifact_sidecar,
    write_motion_coordinate_sidecar,
)


RUN_CONFIG_SCHEMA = "rescue-run-config-v1"


def _load_run_config(path: Path) -> tuple[dict, dict]:
    path = Path(path).resolve()
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load run config {path}: {error}") from error
    if payload.get("schema_version") != RUN_CONFIG_SCHEMA:
        raise ValueError(f"Unsupported run config schema in {path}")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"Run config {path} must contain an arguments object")
    receipt = {
        "schema_version": RUN_CONFIG_SCHEMA,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "description": payload.get("description"),
    }
    return dict(arguments), receipt


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    defaults = {} if defaults is None else dict(defaults)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        dest="run_config",
        type=Path,
        help="Versioned JSON run configuration; explicit CLI flags override it.",
    )
    parser.add_argument("--data-dir", type=Path, required="data_dir" not in defaults)
    parser.add_argument("--stream-id", required="stream_id" not in defaults)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sort", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument(
        "--motion-strict", action=argparse.BooleanOptionalAction, default=False
    )
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
    valid_destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - valid_destinations)
    if unknown:
        parser.error(f"Run config contains unknown arguments: {unknown}")
    parser.set_defaults(**defaults)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", dest="run_config", type=Path)
    preliminary, _ = config_parser.parse_known_args(argv)
    defaults = {}
    receipt = None
    if preliminary.run_config is not None:
        try:
            defaults, receipt = _load_run_config(preliminary.run_config)
        except ValueError as error:
            config_parser.error(str(error))
    args = build_parser(defaults).parse_args(argv)
    for name in ("data_dir", "output_dir", "motion_field"):
        value = getattr(args, name, None)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(value))
    args.run_config_receipt = receipt
    return args


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

    print(
        f"Loading SpikeGLX source {args.data_dir} stream {args.stream_id}; "
        "large sessions can take tens of seconds to scan...",
        flush=True,
    )
    raw = si.read_spikeglx(
        folder_path=args.data_dir,
        load_sync_channel=False,
        stream_id=args.stream_id,
    )
    if args.start_s < 0 or (args.duration_s is not None and args.duration_s <= 0):
        raise ValueError("Requested time range must be positive")
    print(
        "Loaded SpikeGLX source: "
        f"{raw.get_num_channels()} channels, "
        f"{raw.get_total_duration():.2f} s, "
        f"{raw.get_num_samples()} samples.",
        flush=True,
    )
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
        "run_config": getattr(args, "run_config_receipt", None),
        "production_environment": {
            **production_environment_contract(),
            "canonical_prefix": PRODUCTION_UV_PREFIX,
            "lock_required": True,
        },
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
        return
    if not any(
        (
            args.prepare,
            args.sort,
            args.motion_sidecar,
            args.artifact_sidecar,
            args.motion_field,
        )
    ):
        raise SystemExit(
            "Choose --plan, --prepare, --sort, --motion-sidecar, "
            "--artifact-sidecar, or --motion-field"
        )
    environment_receipt = validate_production_environment(require_cuda=args.sort)
    print(json.dumps({"production_environment": environment_receipt}, indent=2))
    recording_dir = output_dir / "recording"
    raw = None
    start_frame = end_frame = None
    bad_ids = None
    if any((args.prepare, args.artifact_sidecar)):
        raw = load_raw(args)
        start_frame, end_frame = requested_frames(raw, args)
        bad_ids = physical_channel_ids(raw, args.bad_channel)
    if args.prepare:
        print(
            "Preparing accepted recording. Bad-channel metrics sample "
            f"{config.channel_metric_batches} x "
            f"{config.channel_metric_batch_duration_s:g}-s batches across the "
            "source session before the requested interval is materialized.",
            flush=True,
        )
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
        from spikeinterface.core import load

        accepted_recording = load(recording_dir)
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
