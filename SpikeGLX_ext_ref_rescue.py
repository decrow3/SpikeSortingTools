"""Human-readable production run sheet for the external-reference rescue pipeline.

Edit the paths and switches in the first section, activate the locked uv
environment, and run this file with ``python SpikeGLX_ext_ref_rescue.py``.
Every expensive stage is cache-aware and safe to restart with its switch left on.
"""

# %% ------------------------------------------------------------------------
# EDIT THIS SECTION FOR EACH RECORDING
# ---------------------------------------------------------------------------
from pathlib import Path

DATA_DIR = Path("/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0")
STREAM_ID = "imec0.ap"
OUTPUT_DIR = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0"
)

# Use DURATION_S = None for the complete recording.
START_S = 0.0
DURATION_S = None

# Leave completed stages enabled: exact accepted caches are reused. Set a stage
# to False only when deliberately skipping it.
RUN_PREPARE = True
RUN_MOTION_SIDECAR = True
RUN_KILOSORT = True

# Safety and restart controls.
MOTION_STRICT = True
RECOMPUTE_MOTION = False
RECOMPUTE_CHANNEL_METRICS = False
PHYSICAL_BAD_CHANNELS = None  # Example: [191]; None uses frozen metrics.

# Compute settings.
N_JOBS = 20
MATERIALIZE_CHUNK_DURATION = "10s"
MOTION_CHUNK_DURATION = "2s"
RUN_MOTION_SPLIT_HALF = False

# Optional diagnostic exports, normally left off.
WRITE_RAW_ARTIFACT_SIDECAR = False
QUALIFIED_MOTION_FIELD = None  # Example: Path("/path/to/qualified_motion.npz")
MOTION_COORDINATE_GAIN = 1.0
MOTION_COORDINATE_MIN_SUPPORT = 1.0
MOTION_COORDINATE_MIN_CONFIDENCE = 0.5
MOTION_COORDINATE_CHUNK_SPIKES = 1_000_000

# Set True to print the complete run plan without loading data or writing files.
PLAN_ONLY = False


# %% ------------------------------------------------------------------------
# PIPELINE IMPORTS AND SMALL OPERATOR HELPERS
# ---------------------------------------------------------------------------
import json
import os

from pipeline import (
    JobConfig,
    MotionSidecarConfig,
    PIPELINE_VERSION,
    PRODUCTION_UV_SETUP,
    RescueConfig,
    materialize_rescue_recording,
    phase_correct,
    production_environment_contract,
    rescue_kilosort4_overrides,
    run_kilosort4,
    run_motion_sidecar_for_accepted_recording,
    validate_production_environment,
    write_artifact_sidecar,
    write_motion_coordinate_sidecar,
)


def _stage(number: int, title: str) -> None:
    print(f"\n{'=' * 72}\nSTEP {number}: {title}\n{'=' * 72}", flush=True)


def _load_raw():
    import spikeinterface.full as si

    print(f"Reading {DATA_DIR} [{STREAM_ID}] ...", flush=True)
    recording = si.read_spikeglx(
        folder_path=DATA_DIR,
        load_sync_channel=False,
        stream_id=STREAM_ID,
    )
    print(
        f"Loaded {recording.get_num_channels()} channels, "
        f"{recording.get_total_duration():.2f} s.",
        flush=True,
    )
    return recording


def _selected_frames(recording) -> tuple[int, int]:
    if START_S < 0 or (DURATION_S is not None and DURATION_S <= 0):
        raise ValueError("START_S must be nonnegative and DURATION_S positive")
    sampling_frequency = float(recording.get_sampling_frequency())
    start = int(round(START_S * sampling_frequency))
    stop = (
        int(recording.get_num_samples())
        if DURATION_S is None
        else start + int(round(DURATION_S * sampling_frequency))
    )
    if start < 0 or stop > recording.get_num_samples() or start >= stop:
        raise ValueError("The selected time range falls outside the recording")
    return start, stop


def physical_channel_ids(recording, requested=PHYSICAL_BAD_CHANNELS):
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
                f"Physical channel {physical} resolved to {matches}; "
                "the channel identity is not unambiguous"
            )
        resolved.append(matches[0])
    return resolved


def _sorter_overrides_for_json() -> dict:
    return {
        key: ("Infinity" if value == float("inf") else value)
        for key, value in rescue_kilosort4_overrides().items()
    }


def build_run_plan() -> dict:
    """Return the visible run-sheet settings and frozen processing contract."""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "production_environment": {
            **production_environment_contract(),
            "canonical_setup": list(PRODUCTION_UV_SETUP),
            "lock_required": True,
        },
        "source_folder": str(DATA_DIR.resolve()),
        "stream_id": STREAM_ID,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "time_range": {"start_s": START_S, "duration_s": DURATION_S},
        "stages": {
            "prepare": RUN_PREPARE,
            "motion_sidecar": RUN_MOTION_SIDECAR,
            "sort_kilosort4": RUN_KILOSORT,
            "artifact_sidecar": WRITE_RAW_ARTIFACT_SIDECAR,
            "motion_coordinates": QUALIFIED_MOTION_FIELD is not None,
        },
        "restart_policy": "reuse_only_exact_validated_cache",
        "physical_bad_channels": PHYSICAL_BAD_CHANNELS,
        "motion_strict": MOTION_STRICT,
        "recompute_motion": RECOMPUTE_MOTION,
        "job_settings": {
            "n_jobs": N_JOBS,
            "materialize_chunk_duration": MATERIALIZE_CHUNK_DURATION,
            "motion_chunk_duration": MOTION_CHUNK_DURATION,
        },
        "motion_sidecar": MotionSidecarConfig(
            split_half=RUN_MOTION_SPLIT_HALF
        ).as_dict(),
        "sorter_overrides": _sorter_overrides_for_json(),
        "voltage_motion_correction": False,
    }


# %% ------------------------------------------------------------------------
# SEQUENTIAL, RESTARTABLE PRODUCTION RUN
# ---------------------------------------------------------------------------
def main() -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spikeglx-rescue-numba-cache")
    plan = build_run_plan()
    print(json.dumps({"run_plan": plan}, indent=2), flush=True)
    if PLAN_ONLY:
        return
    if not any(
        (
            RUN_PREPARE,
            RUN_MOTION_SIDECAR,
            RUN_KILOSORT,
            WRITE_RAW_ARTIFACT_SIDECAR,
            QUALIFIED_MOTION_FIELD is not None,
        )
    ):
        raise RuntimeError("No pipeline stage is enabled in the editable section")

    _stage(0, "VERIFY LOCKED PRODUCTION ENVIRONMENT")
    environment = validate_production_environment(require_cuda=RUN_KILOSORT)
    print(json.dumps(environment, indent=2), flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rescue_run_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )

    recording_dir = OUTPUT_DIR / "recording"
    raw = None
    start_frame = end_frame = None
    bad_channel_ids = None

    # STEP 1 is idempotent. An exact accepted recording is validated and reused.
    if RUN_PREPARE or WRITE_RAW_ARTIFACT_SIDECAR:
        _stage(1, "LOAD SPIKEGLX SOURCE")
        raw = _load_raw()
        start_frame, end_frame = _selected_frames(raw)
        bad_channel_ids = physical_channel_ids(raw)

    if RUN_PREPARE:
        _stage(2, "PREPARE AND MATERIALIZE ACCEPTED RECORDING")
        rescue_config = RescueConfig(
            materialize_n_jobs=N_JOBS,
            materialize_chunk_duration=MATERIALIZE_CHUNK_DURATION,
        )
        _, recording_manifest = materialize_rescue_recording(
            raw,
            recording_dir,
            source_folder=DATA_DIR,
            stream_id=STREAM_ID,
            config=rescue_config,
            bad_channel_ids=bad_channel_ids,
            recompute_channel_metrics=RECOMPUTE_CHANNEL_METRICS,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        print(json.dumps(recording_manifest, indent=2), flush=True)
        if bad_channel_ids is None:
            recorded_bad_ids = set(recording_manifest["bad_channel_ids"])
            bad_channel_ids = [
                channel_id
                for channel_id in raw.get_channel_ids().tolist()
                if str(channel_id) in recorded_bad_ids
            ]

    # STEP 3 is observational: it never changes the accepted sorter voltage.
    if RUN_MOTION_SIDECAR:
        _stage(3, "ESTIMATE RIGID DREDGE MOTION SIDECAR")
        motion_result = run_motion_sidecar_for_accepted_recording(
            recording_dir,
            cache_dir=OUTPUT_DIR / "motion",
            config=MotionSidecarConfig(split_half=RUN_MOTION_SPLIT_HALF),
            job_config=JobConfig(
                n_jobs=N_JOBS,
                chunk_duration=MOTION_CHUNK_DURATION,
                progress_bar=True,
            ),
            recompute=RECOMPUTE_MOTION,
            strict=MOTION_STRICT,
        )
        print(
            json.dumps(
                {
                    "status": motion_result.status,
                    "qc_status": motion_result.qc.status,
                    "cache_lineage": motion_result.cache_lineage,
                    "voltage_modified": False,
                },
                indent=2,
            ),
            flush=True,
        )

    # STEP 4 reuses a complete matching sort. A declared failed partial is
    # archived automatically before a clean retry.
    if RUN_KILOSORT:
        _stage(4, "RUN KILOSORT 4 ON THE EXACT ACCEPTED RECORDING")
        sort_manifest = run_kilosort4(recording_dir, OUTPUT_DIR / "kilosort4")
        print(json.dumps(sort_manifest, indent=2), flush=True)

    if WRITE_RAW_ARTIFACT_SIDECAR:
        _stage(5, "WRITE RAW OVER-500-uV ARTIFACT SIDECAR")
        if bad_channel_ids is None:
            recording_manifest = json.loads(
                (recording_dir / "rescue_recording_manifest.json").read_text()
            )
            recorded_bad_ids = set(recording_manifest["bad_channel_ids"])
            bad_channel_ids = [
                channel_id
                for channel_id in raw.get_channel_ids().tolist()
                if str(channel_id) in recorded_bad_ids
            ]
        phase_recording = phase_correct(raw)
        if start_frame != 0 or end_frame != raw.get_num_samples():
            phase_recording = phase_recording.frame_slice(
                start_frame=start_frame,
                end_frame=end_frame,
            )
        artifact_result = write_artifact_sidecar(
            phase_recording,
            OUTPUT_DIR / "artifacts/raw_over_500uv.h5",
            threshold_uv=500.0,
            excluded_channel_ids=bad_channel_ids,
            chunk_duration_s=MATERIALIZE_CHUNK_DURATION,
            n_jobs=N_JOBS,
        )
        print(json.dumps(artifact_result, indent=2), flush=True)

    if QUALIFIED_MOTION_FIELD is not None:
        _stage(6, "WRITE QUALIFIED POST-SORT MOTION COORDINATES")
        coordinate_result = write_motion_coordinate_sidecar(
            OUTPUT_DIR / "kilosort4",
            QUALIFIED_MOTION_FIELD,
            OUTPUT_DIR / "motion_coordinates",
            gain=MOTION_COORDINATE_GAIN,
            min_support=MOTION_COORDINATE_MIN_SUPPORT,
            min_confidence=MOTION_COORDINATE_MIN_CONFIDENCE,
            chunk_spikes=MOTION_COORDINATE_CHUNK_SPIKES,
        )
        print(json.dumps(coordinate_result, indent=2), flush=True)

    print(f"\nFinished: {DATA_DIR.name} {STREAM_ID}\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
