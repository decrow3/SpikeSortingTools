#!/usr/bin/env python3
"""Bounded EM.1 voltage-equivalence check for huklaban5.

This reads three deterministically selected 3-second snippets from the accepted
DD parent and materialized S_L recording.  It does not materialize voltage,
launch a sort, or write sample values to its compact output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

from npx_preprocessing.motion import ExactLatticeRemapRecording, exact_coordinate_mapping
from npx_preprocessing.motion.lattice_remap_si import sample_stepwise_shifts


DEFAULT_BASE = Path(
    "/media/huklaban5/writable/DARTsort_motion_experiments/"
    "luke0804-imec1-deployable-motion-benchmark-v1/windows/W2/shared/recording"
)
DEFAULT_DD = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/dd_lattice_w2_20260928/huklaban5_v2"
)
TARGET_IDS = [f"imec1.ap#AP{k}" for k in range(202, 384)]
DT = 0.25
SNIPPET_S = 3.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def load_relative_recording(si_module: object, recording_json: Path):
    """Load a portable SI recording relative to its own JSON directory."""

    return si_module.load(recording_json, base_folder=recording_json.parent)


def dd_source_time_origin(parent: object, receipt: dict[str, object]) -> float:
    """Validate and return DD's absolute source-frame clock for the crop."""

    window = receipt.get("source_window")
    if not isinstance(window, dict):
        raise ValueError("DD receipt lacks source_window")
    required = {"start_frame", "end_frame_exclusive", "fs", "start_s", "end_s"}
    if not required.issubset(window):
        raise ValueError(f"DD source_window lacks {sorted(required - set(window))}")
    start_frame = int(window["start_frame"])
    end_frame = int(window["end_frame_exclusive"])
    fs = float(window["fs"])
    start_s = float(window["start_s"])
    end_s = float(window["end_s"])
    if end_frame <= start_frame:
        raise ValueError("DD source window is empty")
    if int(parent.get_num_samples()) != end_frame - start_frame:
        raise ValueError("parent frame count differs from DD source window")
    if not np.isclose(float(parent.get_sampling_frequency()), fs, rtol=0, atol=1e-9):
        raise ValueError("parent sampling rate differs from DD source window")
    if not np.isclose(start_s, start_frame / fs, rtol=0, atol=1e-9):
        raise ValueError("DD source start time is inconsistent with frame and rate")
    if not np.isclose(end_s, end_frame / fs, rtol=0, atol=1e-9):
        raise ValueError("DD source end time is inconsistent with frame and rate")
    return start_s


def get_traces_with_aligned_parent_chunks(
    recording: object,
    start_frame: int,
    end_frame: int,
    *,
    chunk_frames: int,
) -> np.ndarray:
    """Read complete DD materialization chunks, then slice the requested range."""

    start = int(start_frame)
    end = int(end_frame)
    chunk = int(chunk_frames)
    total = int(recording.get_num_samples())
    if chunk <= 0:
        raise ValueError("chunk_frames must be positive")
    if start < 0 or end <= start or end > total:
        raise ValueError("requested trace range is outside recording")
    cover_start = (start // chunk) * chunk
    cover_end = min(total, ((end + chunk - 1) // chunk) * chunk)
    blocks = [
        recording.get_traces(start_frame=left, end_frame=min(left + chunk, total))
        for left in range(cover_start, cover_end, chunk)
    ]
    covered = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)
    offset = start - cover_start
    return covered[offset : offset + end - start]


def preserve_failure(error: BaseException) -> None:
    """Persist an execution failure when an output directory was allocated."""

    if "--output-dir" not in sys.argv:
        return
    index = sys.argv.index("--output-dir") + 1
    if index >= len(sys.argv):
        return
    output_dir = Path(sys.argv[index])
    if not output_dir.is_dir() or (output_dir / "COMPLETE.json").exists():
        return
    atomic_json(
        output_dir / "FAILURE.json",
        {
            "schema": "em-huklaban5-voltage-equivalence-failure-v1",
            "status": "failed",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "sort_launched": False,
        },
    )


def choose_windows(knots: pd.DataFrame) -> list[dict[str, object]]:
    """Choose flat, episode-core, and transition windows without voltage."""

    required = {
        "time_s",
        "field_um",
        "reference_median_um",
        "inside_canonical_mask",
        "q_um",
    }
    if not required.issubset(knots.columns):
        raise ValueError(f"knot table lacks {sorted(required - set(knots.columns))}")
    times = knots.time_s.to_numpy(float)
    q = knots.q_um.to_numpy(float)
    inside = knots.inside_canonical_mask.to_numpy(bool)
    deviation = np.abs(
        knots.field_um.to_numpy(float) - knots.reference_median_um.to_numpy(float)
    )
    cells = int(round(SNIPPET_S / DT))
    candidates: list[int] = []
    for start in range(len(knots) - cells + 1):
        section = times[start : start + cells]
        if np.allclose(np.diff(section), DT, rtol=0, atol=1e-9):
            candidates.append(start)

    rest = [
        start
        for start in candidates
        if not inside[start : start + cells].any()
        and np.all(q[start : start + cells] == 0)
    ]
    if not rest:
        raise ValueError("no deterministic 3-second flat candidate")
    rest_start = min(
        rest,
        key=lambda start: (
            float(np.max(deviation[start : start + cells])),
            float(times[start]),
        ),
    )

    cores = [
        start
        for start in candidates
        if inside[start : start + cells].all()
        and np.any(q[start : start + cells] != 0)
    ]
    if not cores:
        raise ValueError("no deterministic 3-second episode-core candidate")
    core_start = min(
        cores,
        key=lambda start: (
            -float(np.mean(np.abs(q[start : start + cells]))),
            float(times[start]),
        ),
    )

    changes = np.flatnonzero(np.diff(q) != 0) + 1
    if changes.size == 0:
        raise ValueError("no q transition")
    transition_index = min(
        changes.tolist(),
        key=lambda index: (-abs(float(q[index] - q[index - 1])), float(times[index])),
    )
    transition_boundary = (times[transition_index - 1] + times[transition_index]) / 2.0

    def row(label: str, start_s: float, rule: str) -> dict[str, object]:
        return {"label": label, "nominal_start_s": start_s, "duration_s": SNIPPET_S, "rule": rule}

    return [
        row(
            "low_deviation_flat",
            float(times[rest_start] - DT / 2),
            "outside mask, q=0; minimize maximum abs(field-reference), then earliest",
        ),
        row(
            "episode_core",
            float(times[core_start] - DT / 2),
            "inside mask with nonzero q; maximize mean abs(q), then earliest",
        ),
        row(
            "transition",
            float(transition_boundary - SNIPPET_S / 2),
            "largest abs adjacent-q change, then earliest; center 3 s on boundary",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-recording", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dd-root", type=Path, default=DEFAULT_DD)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    import spikeinterface as si
    from spikeinterface.preprocessing import get_spatial_interpolation_kernel

    started = time.monotonic()
    provenance_path = args.base_recording / "provenance.json"
    knots_path = args.dd_root / "lattice_W2_pair.csv"
    if not knots_path.exists():
        knots_path = args.dd_root / "receipts/lattice_W2_pair.csv"
    dd_recording_path = args.dd_root / "inputs/S_L/recording.json"
    dd_receipt_path = args.dd_root / "INPUT_AND_SOURCE_RECEIPT.json"
    provenance = json.loads(provenance_path.read_text())
    parent = si.load(provenance["kwargs"]["parent_recording"], base_folder=args.base_recording)
    expected = load_relative_recording(si, dd_recording_path)
    dd_receipt = json.loads(dd_receipt_path.read_text())
    source_time_origin = dd_source_time_origin(parent, dd_receipt)
    knots = pd.read_csv(knots_path)

    if si.__version__ != "0.104.7":
        raise RuntimeError(f"expected SpikeInterface 0.104.7, got {si.__version__}")
    if parent.get_num_channels() != 383 or expected.get_num_channels() != 182:
        raise RuntimeError("unexpected parent or DD target channel count")
    if list(map(str, expected.channel_ids)) != TARGET_IDS:
        raise RuntimeError("DD target channel order mismatch")
    if "imec1.ap#AP191" in set(map(str, parent.channel_ids)):
        raise RuntimeError("AP191 unexpectedly present")

    centers = knots.time_s.to_numpy(float)
    shifts = knots.q_um.to_numpy(float)
    corrected_parent = ExactLatticeRemapRecording(
        parent,
        centers,
        shifts,
        cell_width_s=DT,
        time_origins_s=[source_time_origin],
    )
    corrected = corrected_parent.select_channels(TARGET_IDS)
    selections = choose_windows(knots)
    fs = float(parent.get_sampling_frequency())
    dd_chunk_frames = round(fs)
    origin = source_time_origin
    frames = int(round(SNIPPET_S * fs))
    for selection in selections:
        start = int(np.ceil((float(selection["nominal_start_s"]) - origin) * fs - 1e-9))
        end = start + frames
        if start < 0 or end > parent.get_num_samples():
            raise ValueError(f"selection outside parent: {selection}")
        selection["start_frame"] = start
        selection["end_frame_exclusive"] = end
        selection["actual_start_s"] = origin + start / fs
        selection["actual_end_s"] = origin + end / fs
    atomic_json(args.output_dir / "SELECTIONS.json", selections)

    geometry = np.asarray(parent.get_channel_locations(), float)
    mapping_hashes = {}
    for shift in np.unique(shifts):
        mapping = exact_coordinate_mapping(geometry, float(shift))
        mapping_hashes[str(int(shift))] = hashlib.sha256(mapping.tobytes()).hexdigest()

    results = []
    all_pass = True
    for selection in selections:
        start = int(selection["start_frame"])
        end = int(selection["end_frame_exclusive"])
        actual = get_traces_with_aligned_parent_chunks(
            corrected,
            start,
            end,
            chunk_frames=dd_chunk_frames,
        )
        reference = expected.get_traces(start_frame=start, end_frame=end)
        equal = bool(np.array_equal(actual, reference))
        zero_equal = bool(np.array_equal(actual == 0, reference == 0))
        max_error = float(np.max(np.abs(actual.astype(np.float64) - reference)))
        sample_times = origin + np.arange(start, end) / fs
        q_by_frame = sample_stepwise_shifts(
            centers, shifts, sample_times, cell_width_s=DT
        )
        difference = np.argwhere(actual != reference)
        first_difference = None
        if difference.size:
            first_difference = {
                "relative_frame": int(difference[0, 0]),
                "channel_index": int(difference[0, 1]),
                "q_um": float(q_by_frame[difference[0, 0]]),
            }
        passed = equal and zero_equal
        all_pass &= passed
        results.append(
            {
                "label": selection["label"],
                "shape": list(actual.shape),
                "dtype_actual": str(actual.dtype),
                "dtype_reference": str(reference.dtype),
                "bytewise_equal": equal,
                "zero_positions_equal": zero_equal,
                "max_abs_error": max_error,
                "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "q_by_frame_sha256": hashlib.sha256(q_by_frame.tobytes()).hexdigest(),
                "first_difference": first_difference,
            }
        )

    source_files = {
        "adapter": Path(inspect.getsourcefile(ExactLatticeRemapRecording)),
        "si_kernel": Path(inspect.getsourcefile(get_spatial_interpolation_kernel)),
        "runner": Path(__file__).resolve(),
        "dd_knots": knots_path,
        "dd_input_receipt": dd_receipt_path,
        "dd_materializer": args.dd_root / "source/dd_lattice_inputs.py",
    }
    receipt = {
        "schema": "em-huklaban5-voltage-equivalence-v1",
        "status": "pass" if all_pass else "fail",
        "spikeinterface_version": si.__version__,
        "parent_read_chunk_frames": dd_chunk_frames,
        "voltage_exported": False,
        "sort_launched": False,
        "total_voltage_duration_s": len(selections) * SNIPPET_S,
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "mapping_sha256_by_q": mapping_hashes,
        "results": results,
        "wall_s": time.monotonic() - started,
    }
    atomic_json(args.output_dir / "RESULT.json", receipt)
    if not all_pass:
        raise RuntimeError("EM exact-adapter voltage equivalence failed; evidence preserved")
    atomic_json(
        args.output_dir / "COMPLETE.json",
        {"status": "complete", "result_sha256": sha256(args.output_dir / "RESULT.json")},
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        preserve_failure(error)
        raise
