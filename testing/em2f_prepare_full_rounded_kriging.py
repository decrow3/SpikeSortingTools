#!/usr/bin/env python3
"""Prepare the hash-bound full-session rounded-kriging recording descriptor."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npx_preprocessing.motion import InterpolatedMotionRecording
from testing.em_huklaban5_voltage_equivalence import TARGET_IDS
from testing.em_huklaban5_voltage_equivalence_arms import find_channel_slice


BASE = Path(
    "/media/huklaban5/writable/DARTsort_motion_experiments/"
    "luke0804-imec1-deployable-motion-benchmark-v1/windows/W2/shared/recording"
)
PROVENANCE = BASE / "provenance.json"
LATTICE = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/"
    "dd_lattice_w2_20260928/huklaban5_v2/lattice_full.csv"
)
FIELD = Path(
    "/media/huklaban5/writable/DARTsort_motion_experiments/incoming/"
    "luke0804_imec1_two_layer_v1/luke0804_imec1_two_layer_motion.npz"
)
MASK = Path(
    "/home/huklaban5/Documents/DARTsort/experiments/luke0804_imec1/"
    "canonical_censor_mask_hub_v1/censor_mask_v1.csv"
)
EXPECTED = {
    "provenance": "0ea45d85a15b99e52d66e24691aef3e6a136cff1bcb0d8d5db64a5ea62281105",
    "lattice": "92d7a28ccc808f76924a6ba392eb71c1a9b0b867796615a15307d1c59bc7df4f",
    "field": "85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9",
    "mask": "86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55",
    "source_geometry": "c01d267823426cf3cd96609984105f9a5833890dec9801bbb15f5d14b54d8f71",
    "target_geometry": "cb5a508f62f02ac6cd0774472e19cea99b56a3696ac6fc76c9610337437a8607",
}
EXPECTED_FRAMES = 314_204_094
EXPECTED_FS = 29_999.759166666667


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def remove_unique_frame_slice(node: dict[str, object]) -> tuple[dict[str, object], dict[str, int]]:
    """Return a copied SI descriptor with its one FrameSlice node removed."""
    result = copy.deepcopy(node)
    removed: list[dict[str, int]] = []

    def visit(value: object) -> object:
        if isinstance(value, dict):
            class_name = str(value.get("class", ""))
            if class_name.endswith("FrameSliceRecording"):
                kwargs = value["kwargs"]
                assert isinstance(kwargs, dict)
                removed.append(
                    {
                        "start_frame": int(kwargs["start_frame"]),
                        "end_frame": int(kwargs["end_frame"]),
                    }
                )
                return visit(kwargs["parent_recording"])
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    result = visit(result)
    if len(removed) != 1:
        raise RuntimeError(f"expected one FrameSliceRecording, found {len(removed)}")
    assert isinstance(result, dict)
    return result, removed[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import spikeinterface as si

    for name, path in (
        ("provenance", PROVENANCE),
        ("lattice", LATTICE),
        ("field", FIELD),
        ("mask", MASK),
    ):
        actual = sha256(path)
        if actual != EXPECTED[name]:
            raise RuntimeError(f"{name} hash differs: {actual}")

    provenance = json.loads(PROVENANCE.read_text())
    parent_descriptor, removed_slice = remove_unique_frame_slice(
        provenance["kwargs"]["parent_recording"]
    )
    parent = si.load(parent_descriptor, base_folder=BASE)
    parent_ids = list(map(str, parent.get_channel_ids()))
    if parent.get_num_samples() != EXPECTED_FRAMES:
        raise RuntimeError(f"unexpected full-session frames: {parent.get_num_samples()}")
    if not np.isclose(parent.get_sampling_frequency(), EXPECTED_FS, rtol=0, atol=1e-9):
        raise RuntimeError("unexpected full-session sampling frequency")
    if len(parent_ids) != 383 or "imec1.ap#AP191" in parent_ids:
        raise RuntimeError("parent is not the accepted 383-channel AP191-excluded source")

    source_geometry = np.asarray(parent.get_channel_locations(), dtype=np.float64)[:, :2]
    target_indices = np.asarray([parent_ids.index(channel) for channel in TARGET_IDS])
    target_geometry = source_geometry[target_indices]
    if array_sha256(source_geometry) != EXPECTED["source_geometry"]:
        raise RuntimeError("source geometry hash differs")
    if array_sha256(target_geometry) != EXPECTED["target_geometry"]:
        raise RuntimeError("target geometry hash differs")

    full_slice = find_channel_slice(provenance, 384)
    full_ids = list(map(str, full_slice["kwargs"]["channel_ids"]))
    full_geometry = np.asarray(full_slice["properties"]["location"], dtype=np.float64)[:, :2]
    if full_ids[191] != "imec1.ap#AP191":
        raise RuntimeError("full geometry does not place AP191 at index 191")

    knots = pd.read_csv(LATTICE)
    required_columns = {
        "knot_index",
        "time_s",
        "field_um",
        "reference_median_um",
        "inside_canonical_mask",
        "q_um",
    }
    if set(knots.columns) != required_columns or len(knots) != 41_895:
        raise RuntimeError("unexpected full-session lattice schema")
    if not np.allclose(knots.time_s.to_numpy(), np.arange(len(knots)) * 0.25):
        raise RuntimeError("full-session lattice is not on the 0.25-s grid")
    references = np.unique(knots.reference_median_um.to_numpy(np.float64))
    if references.size != 1 or not np.isclose(references[0], -4.69558824159, atol=1e-11):
        raise RuntimeError(f"unexpected reference offset: {references}")
    states = sorted(map(int, knots.q_um.unique()))
    expected_states = [-280, -240, -200, -160, -120, -80, -40, 0, 40, 80]
    if states != expected_states:
        raise RuntimeError(f"unexpected full-session rounded states: {states}")

    args.output.mkdir(parents=True)
    frozen_lattice = args.output / "lattice_full.csv"
    shutil.copy2(LATTICE, frozen_lattice)
    recording = InterpolatedMotionRecording(
        parent,
        knots.time_s.to_numpy(np.float64),
        knots.q_um.to_numpy(np.float64),
        full_channel_ids=full_ids,
        full_channel_locations=full_geometry,
        target_channel_ids=TARGET_IDS,
        bad_channel_id="imec1.ap#AP191",
        cell_width_s=0.25,
        method="kriging",
        sigma_um=20.0,
        p=1.0,
        bad_channel_sigma_um=20.0,
        bad_channel_p=1.3,
        time_origins_s=[0.0],
    )
    if recording.get_num_samples() != EXPECTED_FRAMES or recording.get_num_channels() != 182:
        raise RuntimeError("unexpected full-session output signature")
    if recording.get_dtype() != np.dtype("float32"):
        raise RuntimeError("unexpected full-session output dtype")

    descriptor = args.output / "rounded_kriging_recording.json"
    recording.dump_to_json(descriptor)
    loaded = si.load(descriptor)
    if loaded.get_num_samples() != EXPECTED_FRAMES or loaded.get_num_channels() != 182:
        raise RuntimeError("reloaded full-session descriptor differs")

    counts = {str(int(key)): int(value) for key, value in knots.q_um.value_counts().items()}
    result = {
        "schema": "em2f-full-rounded-kriging-recording-v1",
        "status": "complete",
        "producer": {"python": sys.executable, "spikeinterface": si.__version__},
        "voltage_read": False,
        "sort_launched": False,
        "frame_slice_removed": removed_slice,
        "preprocessing_change": "removed only the W2 FrameSliceRecording; all surrounding wrappers retained",
        "rounded_states_um": states,
        "state_counts": counts,
        "recording": {
            "path": str(descriptor.resolve().relative_to(ROOT)),
            "bytes": descriptor.stat().st_size,
            "sha256": sha256(descriptor),
            "frames": EXPECTED_FRAMES,
            "channels": 182,
            "dtype": "float32",
            "sampling_frequency_hz": EXPECTED_FS,
            "duration_seconds": EXPECTED_FRAMES / EXPECTED_FS,
        },
        "inputs": {
            "provenance_path": str(PROVENANCE),
            "provenance_sha256": EXPECTED["provenance"],
            "lattice_path": str(frozen_lattice.resolve().relative_to(ROOT)),
            "lattice_sha256": EXPECTED["lattice"],
            "field_path": str(FIELD),
            "field_sha256": EXPECTED["field"],
            "mask_path": str(MASK),
            "mask_sha256": EXPECTED["mask"],
            "source_geometry_sha256": EXPECTED["source_geometry"],
            "target_geometry_sha256": EXPECTED["target_geometry"],
        },
    }
    (args.output / "PREPARATION.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
