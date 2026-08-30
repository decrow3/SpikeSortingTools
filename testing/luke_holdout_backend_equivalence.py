"""Gate the CUDA local-reference implementation against sealed NumPy semantics.

The representative subset is chosen without voltage access: one of the two
sealed windows in every probe x time-third cell is selected by SHA256 rank.
The rule happens to yield three quiet and three high-motion chunks.  Acceptance
requires identical candidate coordinates, strata, and SHA-selected identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from testing.luke_draw_prospective_holdout_events import (
    MANIFEST,
    METHOD,
    OUTPUT,
    PARENT_SEAL,
    PROBE_SAMPLE_RATES_HZ,
    SEALED_WINDOWS,
    SEED,
    apply_local_reference,
    scan_window,
    select_events,
    sha256_file,
    validate_parent_seal,
)
from testing.luke_yates_raw_voltage_audit import load_specs, spatial_neighbors


PROTOCOL = OUTPUT / "backend_equivalence_protocol_v2.json"
RESULT = OUTPUT / "backend_equivalence_result_v2.json"
CORE_S = 5.0
SAMPLE_CHECK_S = 0.25
MAX_ABS_REFERENCE_DIFFERENCE_UV = 1e-3


def select_representative_windows(windows: list[dict]) -> list[dict]:
    selected = []
    for probe in sorted(PROBE_SAMPLE_RATES_HZ):
        for time_third in (1, 2, 3):
            choices = [
                row
                for row in windows
                if row["probe"] == probe and row["time_third"] == time_third
            ]
            if len(choices) != 2:
                raise RuntimeError(f"Expected two sealed choices for {probe} third {time_third}")
            selected.append(
                min(
                    choices,
                    key=lambda row: hashlib.sha256(
                        f"{SEED}|backend-equivalence|{probe}|{time_third}|{row['window_id']}".encode()
                    ).hexdigest(),
                )
            )
    return selected


def protocol_payload(manifest_path: Path = MANIFEST) -> dict:
    manifest = json.loads(manifest_path.read_text())
    selected = select_representative_windows(manifest["windows"])
    return {
        "version": 2,
        "purpose": "Accept or reject torch_cuda as an implementation-equivalent backend for the sealed NumPy local-median operation.",
        "manifest_sha256": sha256_file(manifest_path),
        "selection_rule": "For each probe x time-third, take the lower SHA256(seed|backend-equivalence|probe|time_third|window_id); test the first 5 s core.",
        "chunks": [
            {
                "probe": row["probe"],
                "time_third": row["time_third"],
                "window_id": row["window_id"],
                "motion_stratum": row["motion_stratum"],
                "start_s": row["start_s"],
                "stop_s": row["start_s"] + CORE_S,
            }
            for row in selected
        ],
        "acceptance": {
            "max_abs_reference_difference_uv": MAX_ABS_REFERENCE_DIFFERENCE_UV,
            "candidate_coordinate_sets_exact": True,
            "candidate_strata_exact": True,
            "sha_selected_identity_sets_exact": True,
        },
    }


def seal_protocol(path: Path = PROTOCOL) -> dict:
    payload = protocol_payload()
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        raise RuntimeError("Refusing to replace a different backend-equivalence protocol")
    if not path.exists():
        path.write_bytes(encoded)
    seal = {"protocol": str(path), "sha256": sha256_file(path), "sealed": True}
    seal_path = path.with_name("backend_equivalence_protocol_seal_v2.json")
    if seal_path.exists() and json.loads(seal_path.read_text()) != seal:
        raise RuntimeError("Existing backend-equivalence protocol seal differs")
    if not seal_path.exists():
        seal_path.write_text(json.dumps(seal, indent=2) + "\n")
    return seal


def reference_difference(spec, window: dict) -> float:
    probe = window["probe"]
    fs = PROBE_SAMPLE_RATES_HZ[probe]
    pad = int(round(0.1 * fs))
    start = int(round(float(window["start_s"]) * fs))
    stop = start + int(round(SAMPLE_CHECK_S * fs))
    raw = np.memmap(
        spec.binary,
        dtype="int16",
        mode="r",
        shape=(spec.n_frames, spec.n_channels_file),
    )
    values = np.asarray(raw[start - pad : stop + pad, : spec.neural_channels], dtype=np.float32)
    values *= float(spec.gain_uv_per_count)
    sos = butter(3, (300.0, 6000.0), btype="bandpass", fs=30000.0, output="sos")
    filtered = sosfiltfilt(sos, values, axis=0).astype(np.float32)[pad:-pad]
    neighbors = spatial_neighbors(spec.locations_um, spec.shanks, 100.0)
    numpy_values = apply_local_reference(filtered, neighbors, "numpy")
    cuda_values = apply_local_reference(filtered, neighbors, "torch_cuda")
    return float(np.max(np.abs(numpy_values - cuda_values)))


def compare_candidates(numpy_frame, cuda_frame, window: dict) -> dict:
    coordinate = ["sample_index", "physical_channel", "polarity"]
    strata = coordinate + ["depth_third", "amplitude_stratum"]
    left = numpy_frame.sort_values(coordinate).reset_index(drop=True)
    right = cuda_frame.sort_values(coordinate).reset_index(drop=True)
    coordinates_equal = left[coordinate].equals(right[coordinate])
    strata_equal = coordinates_equal and left[strata].equals(right[strata])
    if coordinates_equal:
        amplitude_max = float(np.max(np.abs(left.amplitude_uv - right.amplitude_uv))) if len(left) else 0.0
    else:
        amplitude_max = None
    left_selected, _ = select_events(numpy_frame, [window])
    right_selected, _ = select_events(cuda_frame, [window])
    selected_equal = set(left_selected.rank_sha256) == set(right_selected.rank_sha256)
    return {
        "numpy_candidate_count": len(left),
        "cuda_candidate_count": len(right),
        "candidate_coordinate_sets_exact": bool(coordinates_equal),
        "candidate_strata_exact": bool(strata_equal),
        "candidate_amplitude_max_abs_difference_uv": amplitude_max,
        "sha_selected_identity_sets_exact": bool(selected_equal),
    }


def run(protocol_path: Path = PROTOCOL, result_path: Path = RESULT) -> dict:
    validate_parent_seal(MANIFEST, PARENT_SEAL, SEALED_WINDOWS)
    expected = protocol_payload()
    if json.loads(protocol_path.read_text()) != expected:
        raise RuntimeError("Backend-equivalence protocol differs from its fixed rule")
    seal = json.loads(protocol_path.with_name("backend_equivalence_protocol_seal_v2.json").read_text())
    if seal.get("sealed") is not True or seal.get("sha256") != sha256_file(protocol_path):
        raise RuntimeError("Backend-equivalence protocol does not match its seal")
    specs = {
        spec.name.split()[1]: spec
        for spec in load_specs()
        if spec.name.startswith("Luke") and spec.name.endswith("session")
    }
    rows = []
    for chunk in expected["chunks"]:
        window = dict(chunk)
        spec = specs[chunk["probe"]]
        print(f"Comparing {chunk['probe']} {chunk['window_id']}", flush=True)
        numpy_frame = scan_window(spec, window, core_s=CORE_S, reference_backend="numpy")
        cuda_frame = scan_window(spec, window, core_s=CORE_S, reference_backend="torch_cuda")
        rows.append(
            {
                **chunk,
                "reference_max_abs_difference_uv": reference_difference(spec, window),
                **compare_candidates(numpy_frame, cuda_frame, window),
            }
        )
    accepted = all(
        row["reference_max_abs_difference_uv"] <= MAX_ABS_REFERENCE_DIFFERENCE_UV
        and row["candidate_coordinate_sets_exact"]
        and row["candidate_strata_exact"]
        and row["sha_selected_identity_sets_exact"]
        for row in rows
    )
    result = {
        "accepted": accepted,
        "protocol_sha256": sha256_file(protocol_path),
        "method_sha256": sha256_file(METHOD),
        "chunks": rows,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    if not accepted:
        raise RuntimeError("CUDA local-reference backend failed equivalence gate")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("seal-protocol", "run"))
    args = parser.parse_args()
    result = seal_protocol() if args.mode == "seal-protocol" else run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
