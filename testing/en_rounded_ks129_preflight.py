#!/usr/bin/env python3
"""EN preflight: frozen inputs, lattice operator, q=0, and zero-fill audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from npx_preprocessing.motion.lattice_remap_si import audit_spikeinterface_nearest_kernel
from testing.en_rounded_field import build_adapter, rounded_rigid_field, zero_fill_channel_seconds


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_arm(path: Path, expected: str, time_key: str, displacement_key: str, arm: str):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{arm} field hash differs: {actual}")
    with np.load(path, allow_pickle=False) as data:
        time_s = np.asarray(data[time_key], dtype=np.float64)
        displacement = np.asarray(data[displacement_key], dtype=np.float64)
    if arm == "A":
        if displacement.ndim != 2 or displacement.shape[0] != time_s.size:
            raise ValueError("A field is not time-by-depth")
        displacement = np.median(displacement, axis=1)
    if displacement.shape != time_s.shape or not np.all(np.isfinite(displacement)):
        raise ValueError(f"{arm} rigid projection has invalid shape or values")
    if not np.all(np.diff(time_s) > 0):
        raise ValueError(f"{arm} time grid is not strictly increasing")
    return time_s, displacement, actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/en_rounded_field_ks129.v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-binary-hash-receipt", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    config = json.loads(args.config.read_text())

    import kilosort
    import spikeinterface as si
    ref = config["reference"]
    recording_dir = Path(ref["recording_dir"])
    manifest_path = recording_dir / "rescue_recording_manifest.json"
    manifest_hash = sha256(manifest_path)
    if manifest_hash != ref["recording_manifest_sha256"]:
        raise RuntimeError(f"reference manifest hash differs: {manifest_hash}")
    manifest = json.loads(manifest_path.read_text())
    required = {
        "recording_content_sha256": ref["recording_content_sha256"],
        "num_samples": ref["num_samples"],
        "num_channels": ref["num_channels"],
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"reference {key} differs: {manifest.get(key)}")
    expected_bytes = int(ref["num_samples"] * ref["num_channels"] * np.dtype("int16").itemsize)
    binary_path = recording_dir / "traces_cached_seg0.raw"
    if binary_path.stat().st_size != expected_bytes:
        raise RuntimeError("reference binary byte count differs")
    full_hash = None
    if args.full_binary_hash_receipt is not None:
        full_hash = json.loads(args.full_binary_hash_receipt.read_text())
        if full_hash.get("status") != "pass" or full_hash.get("actual_sha256") != ref["binary_sha256"]:
            raise RuntimeError("full binary hash receipt is absent, failed, or names a different digest")
    source = si.load(recording_dir)
    fs = float(source.get_sampling_frequency())
    if not np.isclose(fs, ref["sampling_frequency_hz"], rtol=0, atol=1e-9):
        raise RuntimeError("reference sampling frequency differs")
    if manifest.get("bad_channel_ids") != ["imec0.ap#AP191"]:
        raise RuntimeError("accepted source does not identify only imec0 AP191 as bad")

    duration_s = source.get_num_samples() / fs
    q0 = build_adapter(
        source,
        np.asarray([0.0, duration_s / 2, duration_s]),
        np.asarray([0.0, 0.0, 0.0]),
    )
    snippet_frames = int(round(3 * fs))
    starts = [0, source.get_num_samples() // 2 - snippet_frames // 2, source.get_num_samples() - snippet_frames]
    q0_windows = []
    bytes_read = 0
    for start in starts:
        stop = start + snippet_frames
        raw = source.get_traces(start_frame=start, end_frame=stop)
        adapted = q0.get_traces(start_frame=start, end_frame=stop)
        equal = raw.tobytes() == adapted.tobytes()
        if not equal:
            differing = np.argwhere(raw != adapted)[0]
            raise RuntimeError(f"q0 differs at absolute frame {start + int(differing[0])}, channel {int(differing[1])}")
        bytes_read += raw.nbytes * 2
        q0_windows.append({"start_frame": start, "end_frame_exclusive": stop, "byte_equal": True})

    geometry = np.asarray(source.get_channel_locations(), dtype=np.float64)[:, :2]
    identity_audit = audit_spikeinterface_nearest_kernel(geometry, [0.0])
    if not all(row["stock_matches_exact"] and row["stock_nonzero_weights"] == 1 for row in identity_audit):
        raise RuntimeError("installed SI q0 nearest kernel is not an exact identity")

    arms = {}
    for arm_key, label in (("arm_b", "B"), ("arm_a", "A")):
        spec = config[arm_key]
        path = Path(spec["field_path"])
        if not path.is_file():
            arms[label] = {"status": "awaiting_input", "path": str(path), "expected_sha256": spec["field_sha256"]}
            continue
        time_s, displacement, field_hash = load_arm(
            path, spec["field_sha256"], spec["time_key"], spec["displacement_key"], label
        )
        rounded, reference_um = rounded_rigid_field(displacement, config["adapter"]["period_um"])
        channel_seconds, fraction = zero_fill_channel_seconds(time_s, rounded, duration_s, geometry)
        values, counts = np.unique(rounded, return_counts=True)
        mapping = audit_spikeinterface_nearest_kernel(geometry, values)
        mapping_failures = [row for row in mapping if not row["stock_matches_exact"]]
        arms[label] = {
            "status": "pass",
            "path": str(path),
            "sha256": field_hash,
            "time_bins": int(time_s.size),
            "support_s": [float(time_s[0]), float(time_s[-1])],
            "reference_median_um": reference_um,
            "rounded_states_um": values.tolist(),
            "rounded_state_bin_counts": dict(zip(map(str, values.tolist()), map(int, counts.tolist()))),
            "si_mapping_pairs": len(mapping),
            "si_mapping_failures": len(mapping_failures),
            "zero_filled_channel_seconds": channel_seconds,
            "zero_filled_fraction": fraction,
        }
        if mapping_failures:
            arms[label]["first_mapping_failure"] = mapping_failures[0]
            raise RuntimeError(f"{label} installed SI nearest mapping differs on a rounded full-probe state")

    payload = {
        "schema_version": "en-rounded-field-ks129-preflight-v1",
        "status": "pass_waiting_a" if arms["A"]["status"] == "awaiting_input" else "pass",
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "producer": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "spikeinterface": si.__version__,
            "kilosort": getattr(kilosort, "__version__", "unknown"),
        },
        "reference": {
            "manifest_sha256": manifest_hash,
            **required,
            "binary_expected_sha256": ref["binary_sha256"],
            "binary_size_bytes": expected_bytes,
            "full_binary_hash_receipt": None if args.full_binary_hash_receipt is None else str(args.full_binary_hash_receipt),
        },
        "checks": {
            "bad_channel_is_imec0_ap191": True,
            "sign_fixture": "covered by test_en_rounded_field: +40 um maps [10,20,30,40] to [20,30,40,0]",
            "q0_si_kernel_identity_all_384_channels": True,
            "q0_sample_windows": q0_windows,
            "q0_full_binary_hash": "required separately before smoke or sort",
            "temporal_assignment": config["adapter"]["temporal_assignment"],
        },
        "arms": arms,
        "resources": {
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "voltage_bytes_read": bytes_read,
        },
    }
    args.output.mkdir(parents=True)
    atomic_json(args.output / "PREFLIGHT.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
