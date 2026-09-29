#!/usr/bin/env python3
"""Prepare the hash-bound W3 rounded-kriging recording descriptor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
    "luke0804-imec1-deployable-motion-benchmark-v1/windows/W3/shared/recording"
)
DD = Path("/home/huklaban5/DARTsort_experiment_scratch/ef_w3_lattice_20260928")
PROVENANCE = BASE / "provenance.json"
KNOTS = DD / "lattice_W3_pair.csv"
INPUT_RECEIPT = DD / "INPUT_AND_SOURCE_RECEIPT.json"
MATERIALIZATION = DD / "MATERIALIZATION_RECEIPT.json"
EXACT_SORT = DD / "ej_v1/runs/SL/dartsort_sorting.npz"
EXPECTED = {
    "provenance": "83c6aebb2c9c602c706d484a03bca666bd61ea18ef9302bda16cee668f471fe4",
    "knots": "0a2ea9368ee47452177edc6e6ad91f8d0861c4d2ecb4c70c35ace8a6a5fa311b",
    "source_geometry": "c01d267823426cf3cd96609984105f9a5833890dec9801bbb15f5d14b54d8f71",
    "target_geometry": "cb5a508f62f02ac6cd0774472e19cea99b56a3696ac6fc76c9610337437a8607",
    "exact_sort": "901de0ec769c62939b06b6472c36500afadd6c515a4ad8e38ad81b91e9b42190",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import spikeinterface as si

    if si.__version__ != "0.104.7":
        raise RuntimeError(f"requires SpikeInterface 0.104.7, got {si.__version__}")
    for name, path in (("provenance", PROVENANCE), ("knots", KNOTS), ("exact_sort", EXACT_SORT)):
        actual = sha256(path)
        if actual != EXPECTED[name]:
            raise RuntimeError(f"{name} hash differs: {actual}")

    provenance = json.loads(PROVENANCE.read_text())
    parent = si.load(provenance["kwargs"]["parent_recording"], base_folder=BASE)
    parent_ids = list(map(str, parent.get_channel_ids()))
    if len(parent_ids) != 383 or "imec1.ap#AP191" in parent_ids:
        raise RuntimeError("W3 parent is not the accepted 383-channel AP191-excluded source")
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

    receipt = json.loads(INPUT_RECEIPT.read_text())
    window = receipt["window"]
    frames = int(window["end_frame_exclusive"]) - int(window["start_frame"])
    if parent.get_num_samples() != frames:
        raise RuntimeError("W3 parent frame count differs from receipt")
    if not np.isclose(parent.get_sampling_frequency(), window["fs"], rtol=0, atol=1e-9):
        raise RuntimeError("W3 parent sampling frequency differs from receipt")
    materialization = json.loads(MATERIALIZATION.read_text())
    if materialization.get("status") != "complete" or not materialization.get(
        "q0_byte_equal_to_accepted_cache"
    ):
        raise RuntimeError("W3 exact lattice materialization lacks accepted q0 identity")

    knots = pd.read_csv(KNOTS)
    states = sorted(map(int, knots.q_um.unique()))
    if states != [-280, -240, -200, -160, -120, -80, -40, 0]:
        raise RuntimeError(f"unexpected W3 rounded states: {states}")
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
        time_origins_s=[float(window["start_s"])],
    )
    if recording.get_num_samples() != frames or recording.get_num_channels() != 182:
        raise RuntimeError("unexpected W3 output signature")
    if recording.get_dtype() != np.dtype("float32"):
        raise RuntimeError("unexpected W3 output dtype")

    args.output.mkdir(parents=True)
    descriptor = args.output / "rounded_kriging_recording.json"
    recording.dump_to_json(descriptor)
    loaded = si.load(descriptor)
    if loaded.get_num_samples() != frames or loaded.get_num_channels() != 182:
        raise RuntimeError("reloaded W3 descriptor differs")
    result = {
        "schema": "em2e-w3-rounded-kriging-recording-v1",
        "status": "complete",
        "producer": {"python": sys.executable, "spikeinterface": si.__version__},
        "voltage_read": False,
        "sort_launched": False,
        "window": window,
        "rounded_states_um": states,
        "recording": {
            "path": str(descriptor.resolve().relative_to(ROOT)),
            "bytes": descriptor.stat().st_size,
            "sha256": sha256(descriptor),
            "frames": frames,
            "channels": 182,
            "dtype": "float32",
        },
        "inputs": {
            "provenance_sha256": EXPECTED["provenance"],
            "knots_sha256": EXPECTED["knots"],
            "source_geometry_sha256": EXPECTED["source_geometry"],
            "target_geometry_sha256": EXPECTED["target_geometry"],
            "exact_lattice_sort_sha256": EXPECTED["exact_sort"],
            "q0_byte_equal_to_accepted_cache": True,
        },
    }
    result_path = args.output / "PREPARATION.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
