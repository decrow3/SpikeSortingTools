#!/usr/bin/env python3
"""Exercise DH's exact motion object through the installed DARTsort consumer."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from testing.luke_dh_exact_chunk_motion import ExactChunkLatticeMotion


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packet", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from dartsort.util.motion import MotionInfo
    from dartsort.peel.matching import ObjectiveUpdateTemplateMatchingPeeler

    with np.load(args.packet / "EXACT_QUERY_MOTION.npz", allow_pickle=False) as z:
        states = np.asarray(z["states_um"], float)
        centers = np.asarray(z["chunk_center_samples"], np.int64)
        fs = float(z["sampling_frequency_hz"])
        chunk = int(z["chunk_length_samples"])
        n_samples = int(z["n_samples"])
    with np.load(
        Path("testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/bc_attempt5")
        / "full_probe_final_label_templates.npz",
        allow_pickle=False,
    ) as z:
        geom = np.asarray(z["geometry_um"], float)[202:384]

    est = ExactChunkLatticeMotion(
        states,
        sampling_frequency_hz=fs,
        chunk_length_samples=chunk,
        n_samples=n_samples,
    )
    consumer = MotionInfo.from_motion_est(geom=geom, dredge_motion_est=est)
    query_times = centers / fs  # exactly matching.py's local sample_index_to_time path
    got = consumer.disp_at_s(query_times, geom[:, 1], grid=True)
    expected = np.broadcast_to(states[None, :], (geom.shape[0], states.size))
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise RuntimeError("installed DARTsort MotionInfo changed exact queried states")
    one = consumer.disp_at_s(np.array([query_times[-1]]), geom[:, 1], grid=True)
    if one.shape != (geom.shape[0], 1) or not np.all(one[:, 0] == states[-1]):
        raise RuntimeError("installed DARTsort scalar matching query shape/value mismatch")

    consumer_source = Path(inspect.getsourcefile(MotionInfo)).resolve()
    matching_source = Path(inspect.getsourcefile(ObjectiveUpdateTemplateMatchingPeeler)).resolve()
    peel_lines, peel_start = inspect.getsourcelines(ObjectiveUpdateTemplateMatchingPeeler.peel_chunk)
    query_line_offsets = [i for i, line in enumerate(peel_lines) if "chunk_center_samples" in line or "sample_index_to_time" in line or "data_at_time" in line]
    result = {
        "status": "pass",
        "consumer": "installed dartsort.util.motion.MotionInfo.from_motion_est/disp_at_s",
        "consumer_source": str(consumer_source),
        "consumer_source_sha256": digest(consumer_source),
        "matching_source": str(matching_source),
        "matching_source_sha256": digest(matching_source),
        "matching_query_source_lines_1based": [peel_start + i for i in query_line_offsets],
        "recording_time_origin_s": 0.0,
        "sampling_frequency_hz": fs,
        "matching_chunk_length_samples": chunk,
        "query_rule": "(chunk_start_sample + 7500//2) / sampling_frequency_hz",
        "query_count": int(states.size),
        "query_first_s": float(query_times[0]),
        "query_last_s": float(query_times[-1]),
        "motion_grid_shape": list(got.shape),
        "motion_grid_dtype": str(got.dtype),
        "scalar_time_grid_shape": list(one.shape),
        "registered_geometry_shape": list(consumer.rgeom.shape),
        "final_padding_samples": est.final_padding_samples,
        "query_support_samples_half_open": list(est.query_support_samples),
        "local_origin_and_support_enforced": True,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
