#!/usr/bin/env python3
"""Execution-disabled, bounded capture for the frozen A four-unit selections.

The ``preflight`` phase is deliberately incapable of opening either recording
binary.  The ``run`` phase is separately gated by three explicit booleans and
uses direct, counted reads so the 41-call/33-MiB contract is enforceable.
"""

from __future__ import annotations

import argparse
import builtins
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


SCHEMA = "en-a-four-unit-physical-signal-capture-v1"
CHANNELS = 384
DTYPE_BYTES = 2
PAIR_INPUT_SAMPLES = 1054       # 512 + 1 + frozen max lag 29 + 512
SINGLE_INPUT_SAMPLES = 1025     # 512 + 1 + 512
PAIR_RETAINED_SAMPLES = 150     # 60 + 1 + frozen max lag 29 + 60
SINGLE_RETAINED_SAMPLES = 121   # 60 + 1 + 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)


@dataclass
class RecordingOpenGuard:
    """Reject and count recording opens during no-read preflight."""

    recording_paths: set[str]
    attempts: int = 0

    def __enter__(self) -> "RecordingOpenGuard":
        self._open = builtins.open
        self._os_open = os.open
        self._memmap = np.memmap
        guard = self

        def checked_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if os.path.abspath(os.fspath(file)) in guard.recording_paths:
                guard.attempts += 1
                raise RuntimeError("recording open forbidden during preflight")
            return guard._open(file, *args, **kwargs)

        def checked_os_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            if os.path.abspath(os.fspath(file)) in guard.recording_paths:
                guard.attempts += 1
                raise RuntimeError("recording os.open forbidden during preflight")
            return guard._os_open(file, *args, **kwargs)

        def checked_memmap(filename: Any, *args: Any, **kwargs: Any) -> Any:
            if os.path.abspath(os.fspath(filename)) in guard.recording_paths:
                guard.attempts += 1
                raise RuntimeError("recording memmap forbidden during preflight")
            return guard._memmap(filename, *args, **kwargs)

        builtins.open = checked_open
        os.open = checked_os_open
        np.memmap = checked_memmap
        return self

    def __exit__(self, *_: Any) -> None:
        builtins.open = self._open
        os.open = self._os_open
        np.memmap = self._memmap


def load_selections(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20 or len({row["slot_id"] for row in rows}) != 20:
        raise RuntimeError("selection receipt must contain 20 unique slots")
    roles = [row["role"] for row in rows]
    if roles.count("short_pair") != 8 or roles.count("ordinary_event") != 8 or roles.count("q0_identity_control") != 4:
        raise RuntimeError("selection role counts differ from frozen 8-pair/12-single plan")
    for row in rows:
        if row["status"] != "resolved" or row["support_1"] != "supported":
            raise RuntimeError(f"unresolved/unsupported slot: {row['slot_id']}")
        if row["role"] == "short_pair":
            lag = int(row["lag_samples"])
            if not (1 <= lag <= 29):
                raise RuntimeError(f"pair lag outside frozen envelope: {row['slot_id']}")
            if int(row["frame_2"]) - int(row["frame_1"]) != lag:
                raise RuntimeError(f"pair endpoints/lag inconsistent: {row['slot_id']}")
    return rows


def window_bounds(row: dict[str, str]) -> dict[str, int]:
    """Return exact half-open input/retained bounds for one frozen slot."""
    first = int(row["frame_1"])
    if row["role"] == "short_pair":
        second = int(row["frame_2"])
        lag = int(row["lag_samples"])
        if second - first != lag:
            raise RuntimeError(f"pair endpoints/lag inconsistent: {row['slot_id']}")
        return {"input_start": first - 512, "input_stop": second + 513,
                "retained_start": first - 60, "retained_stop": second + 61}
    return {"input_start": first - 512, "input_stop": first + 513,
            "retained_start": first - 60, "retained_stop": first + 61}


def expected_arithmetic(rows: list[dict[str, str]]) -> dict[str, int]:
    pairs = sum(row["role"] == "short_pair" for row in rows)
    singles = len(rows) - pairs
    bounds = [window_bounds(row) for row in rows]
    per_rep_input = sum(item["input_stop"] - item["input_start"] for item in bounds) * CHANNELS * DTYPE_BYTES
    per_rep_retained = sum(item["retained_stop"] - item["retained_start"] for item in bounds) * CHANNELS * DTYPE_BYTES
    first_pair = next((item for row, item in zip(rows, bounds) if row["role"] == "short_pair"), None)
    if first_pair is None:
        raise RuntimeError("duplicate-control source absent")
    duplicate_bytes = (first_pair["input_stop"] - first_pair["input_start"]) * CHANNELS * DTYPE_BYTES
    return {
        "selection_slots": len(rows),
        "pair_windows": pairs,
        "single_event_windows": singles,
        "base_window_reads": len(rows) * 2,
        "duplicate_positive_control_reads": 1,
        "raw_read_calls_max": len(rows) * 2 + 1,
        "planned_base_input_bytes": per_rep_input * 2,
        "planned_with_duplicate_input_bytes": per_rep_input * 2 + duplicate_bytes,
        "planned_retained_int16_bytes": per_rep_retained * 2,
        "cap_base_input_bytes": 31844352,
        "cap_with_one_max_pair_duplicate_bytes": 32653824,
        "cap_retained_int16_bytes": 4073472,
    }


def validate_manifest(binding: dict[str, Any]) -> dict[str, Any]:
    path = Path(binding["manifest_path"])
    if sha256(path) != binding["manifest_sha256"]:
        raise RuntimeError(f"manifest hash differs: {path}")
    value = json.loads(path.read_text())
    observed = {
        "num_samples": int(value["num_samples"]),
        "num_channels": int(value["num_channels"]),
        "sampling_frequency_hz": float(value["sampling_frequency_hz"]),
        "dtype": value["dtype"],
        "selected_start_frame": int(value["selected_start_frame"]),
        "selected_end_frame": int(value["selected_end_frame"]),
        "recording_content_sha256": value["recording_content_sha256"],
        "binary_size_bytes": int(value["recording_binary_files"][0]["size_bytes"]),
        "manifest_recorded_binary_sha256": value["recording_binary_files"][0]["sha256"],
    }
    if observed != binding["expected_manifest_fields"]:
        raise RuntimeError(f"manifest fields differ: {path}")
    if value["recording_binary_files"][0]["name"] != Path(binding["binary_path"]).name:
        raise RuntimeError(f"manifest/binary basename differs: {path}")
    return observed


def validate_service(config: dict[str, Any]) -> None:
    path = Path(config["service"]["path"])
    if sha256(path) != config["service"]["sha256"]:
        raise RuntimeError("service hash differs")
    text = path.read_text()
    required = (
        "CPUQuota=400%", "MemoryMax=2G", "MemorySwapMax=0",
        "RuntimeMaxSec=300", "--phase run", str(Path(config["config_path"])),
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"service limits/binding missing: {missing}")


def validate(config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if config["schema"] != SCHEMA or config["status"] not in {
        "execution_disabled_pending_review_and_human_approval", "approved_for_bounded_capture"
    }:
        raise RuntimeError("config schema/status differs")
    if any(config["gates"].values()) and config["status"] != "approved_for_bounded_capture":
        raise RuntimeError("non-disabled gates require approved status")
    if sha256(Path(__file__).resolve()) != config["source_sha256"]:
        raise RuntimeError("capture source hash differs")
    validate_service(config)
    selection = Path(config["selection_receipt"]["path"])
    if sha256(selection) != config["selection_receipt"]["sha256"]:
        raise RuntimeError("selection receipt hash differs")
    rows = load_selections(selection)
    arithmetic = expected_arithmetic(rows)
    if arithmetic != config["arithmetic"]:
        raise RuntimeError(f"byte/read arithmetic differs: {arithmetic}")
    limits = config["limits"]
    if limits != {"cpu_threads_max": 4, "logical_input_bytes_max": 34603008,
                  "ram_bytes_max": 2147483648, "retained_numeric_bytes_max": 6291456,
                  "wall_seconds_max": 300}:
        raise RuntimeError("resource limits differ")
    if arithmetic["planned_with_duplicate_input_bytes"] > limits["logical_input_bytes_max"]:
        raise RuntimeError("logical input byte cap exceeded")
    if arithmetic["planned_retained_int16_bytes"] > limits["retained_numeric_bytes_max"]:
        raise RuntimeError("retained byte cap exceeded")
    manifests = {name: validate_manifest(binding) for name, binding in config["representations"].items()}
    for item in config["frozen_metadata"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen metadata hash differs: {item['path']}")
    output = Path(config["output"])
    if output.exists():
        raise RuntimeError(f"fresh output required; path already exists: {output}")
    return rows, {"arithmetic": arithmetic, "manifest_fields": manifests}


def preflight(config: dict[str, Any], config_path: Path, expected_config_sha256: str) -> dict[str, Any]:
    if sha256(config_path) != expected_config_sha256:
        raise RuntimeError("config hash differs from command-line binding")
    recording_paths = {os.path.abspath(value["binary_path"]) for value in config["representations"].values()}
    with RecordingOpenGuard(recording_paths) as guard:
        _, evidence = validate(config)
    if guard.attempts != 0:
        raise RuntimeError("preflight attempted a recording open")
    return {
        "schema": "en-a-four-unit-physical-signal-capture-preflight-v1",
        "status": "pass_execution_disabled",
        "config_path": str(config_path),
        "config_sha256": expected_config_sha256,
        "recording_open_attempts": guard.attempts,
        "recording_reads": 0,
        "recording_bytes_read": 0,
        "gates": config["gates"],
        **evidence,
    }


def read_window(path: Path, start: int, stop: int, counters: dict[str, int], limits: dict[str, int]) -> np.ndarray:
    count = stop - start
    nbytes = count * CHANNELS * DTYPE_BYTES
    if counters["read_calls"] + 1 > 41 or counters["logical_bytes"] + nbytes > limits["logical_input_bytes_max"]:
        raise RuntimeError("read contract would be exceeded")
    with path.open("rb", buffering=0) as stream:
        stream.seek(start * CHANNELS * DTYPE_BYTES)
        payload = stream.read(nbytes)
    counters["read_calls"] += 1
    counters["logical_bytes"] += nbytes
    if len(payload) != nbytes:
        raise RuntimeError("short recording read")
    return np.frombuffer(payload, dtype=np.int16).reshape(count, CHANNELS).T.copy()


def native_stage_summary(raw: np.ndarray, ops: dict[str, Any]) -> dict[str, float]:
    import torch
    from kilosort.io import BinaryFiltered

    tensor = torch.from_numpy(raw.astype(np.float32, copy=False))
    common = {"chan_map": None, "invert_sign": bool(ops["invert_sign"]),
              "artifact_threshold": float(ops["artifact_threshold"]),
              "whiten_mat": None, "dshift": None, "device": torch.device("cpu")}
    summaries: dict[str, float] = {}
    for name, do_car, hp_filter in (("centered", False, None), ("car", True, None), ("highpass", True, ops["fwav"])):
        if hp_filter is not None:
            hp_filter = torch.as_tensor(hp_filter, dtype=torch.float32)
        target = SimpleNamespace(**common, do_CAR=do_car, hp_filter=hp_filter)
        values = BinaryFiltered.filter(target, tensor.clone()).numpy()
        summaries[f"{name}_rms"] = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        summaries[f"{name}_peak_abs"] = float(np.max(np.abs(values)))
        del values
    return summaries


def execute(config: dict[str, Any]) -> None:
    gates = config["gates"]
    if gates != {"execution_enabled": True, "human_approved": True, "voltage_read_authorized": True}:
        raise RuntimeError("run refused: all three explicit gates must be true")
    rows, _ = validate(config)
    os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    counters = {"read_calls": 0, "logical_bytes": 0, "retained_bytes": 0}
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        ops = np.load(config["ops_path"], allow_pickle=True).item()
        if int(ops["n_chan_bin"]) != CHANNELS or float(ops["artifact_threshold"]) != float("inf"):
            raise RuntimeError("ops channel/artifact settings differ")
        first_pair_full: np.ndarray | None = None
        for representation, binding in config["representations"].items():
            retained: dict[str, np.ndarray] = {}
            for row in rows:
                if time.monotonic() - started > config["limits"]["wall_seconds_max"]:
                    raise RuntimeError("wall-time cap exceeded")
                pair = row["role"] == "short_pair"
                frame = int(row["frame_1"])
                bounds = window_bounds(row)
                full = read_window(Path(binding["binary_path"]), bounds["input_start"], bounds["input_stop"],
                                   counters, config["limits"])
                if first_pair_full is None and pair and representation == "original":
                    first_pair_full = full.copy()
                keep_width = bounds["retained_stop"] - bounds["retained_start"]
                keep_offset = bounds["retained_start"] - bounds["input_start"]
                keep = full[:, keep_offset:keep_offset + keep_width].copy()
                counters["retained_bytes"] += keep.nbytes
                if counters["retained_bytes"] > config["limits"]["retained_numeric_bytes_max"]:
                    raise RuntimeError("retained numeric byte cap exceeded")
                retained[row["slot_id"]] = keep
                summaries.append({"representation": representation, "slot_id": row["slot_id"],
                                  "input_start_frame": bounds["input_start"], "input_stop_frame_exclusive": bounds["input_stop"],
                                  "retained_samples": keep_width, **native_stage_summary(full, ops)})
                del full
            np.savez(output / f"{representation}_retained_int16.npz", **retained)
        if first_pair_full is None:
            raise RuntimeError("duplicate-control source absent")
        first_pair = next(row for row in rows if row["role"] == "short_pair")
        duplicate_bounds = window_bounds(first_pair)
        duplicate = read_window(Path(config["representations"]["original"]["binary_path"]),
                                duplicate_bounds["input_start"], duplicate_bounds["input_stop"],
                                counters, config["limits"])
        if not np.array_equal(first_pair_full, duplicate):
            raise RuntimeError("duplicate positive control differs")
        with (output / "STAGE_SUMMARIES.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader(); writer.writerows(summaries)
        atomic_json(output / "COUNTERS.json", counters)
        atomic_json(output / "COMPLETE.json", {"schema": SCHEMA, "status": "complete", "counters": counters})
    except Exception as exc:
        atomic_json(output / "FAILED.json", {"schema": SCHEMA, "status": "failed", "error": repr(exc), "counters": counters})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("preflight", "run"), required=True)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if Path(config["config_path"]).resolve() != args.config.resolve():
        raise RuntimeError("config path self-binding differs")
    if args.phase == "preflight":
        if not args.expected_config_sha256 or not args.receipt:
            raise RuntimeError("preflight requires expected config hash and receipt path")
        result = preflight(config, args.config, args.expected_config_sha256)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.receipt, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        execute(config)


if __name__ == "__main__":
    main()
