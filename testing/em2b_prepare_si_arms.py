#!/usr/bin/env python3
"""Prepare the two frozen EM.2b stage-1 SpikeInterface recording descriptors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from npx_preprocessing.motion import InterpolatedMotionRecording
from testing.em_huklaban5_voltage_equivalence import DT, TARGET_IDS, dd_source_time_origin
from testing.em_huklaban5_voltage_equivalence_arms import DD, preflight, sha256
from testing.em2a_operator_screen import EXPECTED_FIELD_SHA256, FIELD


ARM_SPECS = {
    "unrounded_kriging": {"field": "unrounded", "method": "kriging"},
    "rounded_kriging_bridge": {"field": "rounded", "method": "kriging"},
}


def descriptor_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_recording(state: dict[str, object], field: str) -> InterpolatedMotionRecording:
    parent = state["parent"]
    if field == "unrounded":
        with np.load(FIELD, allow_pickle=False) as saved:
            centers = np.asarray(saved["time_s"], dtype=np.float64)
            shifts = np.asarray(saved["displacement_um"], dtype=np.float64)[:, 0]
            if str(saved["sign_convention"]) != "corrected = observed - displacement":
                raise RuntimeError("unrounded field sign convention differs")
    elif field == "rounded":
        knots = state["knots"]
        centers = knots.time_s.to_numpy(np.float64)
        shifts = knots.q_um.to_numpy(np.float64)
    else:
        raise ValueError(field)
    source_origin = dd_source_time_origin(parent, state["dd_receipt"])
    return InterpolatedMotionRecording(
        parent,
        centers,
        shifts,
        full_channel_ids=state["full_ids"],
        full_channel_locations=state["full_geometry"],
        target_channel_ids=TARGET_IDS,
        bad_channel_id="imec1.ap#AP191",
        cell_width_s=DT,
        method="kriging",
        sigma_um=20.0,
        p=1.0,
        bad_channel_sigma_um=20.0,
        bad_channel_p=1.3,
        time_origins_s=[source_origin],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import spikeinterface as si

    if si.__version__ != "0.104.7":
        raise RuntimeError(f"descriptor preparation requires SI 0.104.7, got {si.__version__}")
    if sha256(FIELD) != EXPECTED_FIELD_SHA256:
        raise RuntimeError("unrounded source field hash differs")
    state = preflight(si)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    arm_rows = {}
    for arm, spec in ARM_SPECS.items():
        path = args.output / f"{arm}_recording.json"
        if path.exists():
            raise FileExistsError(path)
        recording = build_recording(state, spec["field"])
        if (
            recording.get_num_samples() != 10_199_918
            or recording.get_num_channels() != 182
            or recording.get_dtype() != np.dtype("float32")
            or list(map(str, recording.get_channel_ids())) != TARGET_IDS
        ):
            raise RuntimeError(f"unexpected recording signature for {arm}")
        recording.dump_to_json(path)
        loaded = si.load(path)
        if (
            loaded.get_num_samples() != recording.get_num_samples()
            or loaded.get_num_channels() != recording.get_num_channels()
            or loaded.get_dtype() != recording.get_dtype()
        ):
            raise RuntimeError(f"descriptor reload differs for {arm}")
        files[path.name] = {"bytes": path.stat().st_size, "sha256": descriptor_sha256(path)}
        arm_rows[arm] = {
            "field": spec["field"],
            "method": "kriging",
            "sigma_um": 20.0,
            "p": 1.0,
            "bad_channel": "imec1.ap#AP191",
            "bad_channel_sigma_um": 20.0,
            "bad_channel_p": 1.3,
            "frames": recording.get_num_samples(),
            "channels": recording.get_num_channels(),
            "dtype": str(recording.get_dtype()),
            "sampling_frequency_hz": float(recording.get_sampling_frequency()),
            "time_origin_s": float(recording._kwargs["time_origins_s"][0]),
        }
    receipt = {
        "schema": "em2b-stage1-recordings-v1",
        "status": "complete",
        "producer": {"python": sys.executable, "spikeinterface": si.__version__},
        "voltage_read": False,
        "sort_launched": False,
        "source_field_sha256": EXPECTED_FIELD_SHA256,
        "rounded_knots_sha256": state["compact_hashes"]["knots"],
        "source_geometry_sha256": state["compact_hashes"]["source_geometry"],
        "target_geometry_sha256": state["compact_hashes"]["target_geometry"],
        "arms": arm_rows,
        "files": files,
    }
    receipt_path = args.output / "stage1_recordings.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
