#!/usr/bin/env python3
"""Bounded H5 real-window equality runner for Candidate 3.

Real execution is disabled unless ``--execute-real`` and the exact reviewed
contract hash are both supplied.  ``--self-test`` uses synthetic in-memory data
and never opens contract input paths.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import resource
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def sha256_bytes(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_write(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(memoryview(contiguous).cast("B"))


def source_path(obj: Any) -> Path:
    path = inspect.getsourcefile(obj)
    if path is None:
        raise RuntimeError(f"no source file for {obj!r}")
    return Path(path).resolve()


def imports_and_sources(contract: dict[str, Any]) -> dict[str, Any]:
    import dartsort
    import kilosort
    from dartsort.util import main_util, preprocess_util
    from kilosort.io import BinaryFiltered
    from npx_preprocessing.motion import KilosortPrewhiteningRecording

    paths = {
        "wrapper": source_path(KilosortPrewhiteningRecording),
        "kilosort_io": source_path(BinaryFiltered),
        "dartsort_main": Path(dartsort.__file__).resolve().parent / "main.py",
        "dartsort_main_util": source_path(main_util.ds_all_to_workdir),
        "dartsort_preprocess_util": source_path(preprocess_util.preprocess),
    }
    expected = contract["source"]
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    required = {
        "wrapper": expected["wrapper_sha256"],
        "kilosort_io": expected["kilosort_io_sha256"],
        "dartsort_main": expected["dartsort_main_sha256"],
        "dartsort_main_util": expected["dartsort_main_util_sha256"],
        "dartsort_preprocess_util": expected["dartsort_preprocess_util_sha256"],
    }
    if hashes != required:
        raise RuntimeError(f"imported source hash mismatch: observed={hashes} expected={required}")
    if kilosort.__version__ != expected["kilosort_version"]:
        raise RuntimeError(f"unexpected Kilosort version {kilosort.__version__}")
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "kilosort_version": kilosort.__version__,
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
    }


def eager_filter(values: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    import torch
    from kilosort.io import BinaryFiltered
    from kilosort.preprocessing import get_highpass_filter

    filt = contract["filter"]
    producer = object.__new__(BinaryFiltered)
    producer.chan_map = torch.arange(values.shape[1], dtype=torch.int64)
    producer.invert_sign = bool(filt["invert_sign"])
    producer.do_CAR = bool(filt["global_lower_median_car"])
    producer.hp_filter = get_highpass_filter(
        fs=contract["input"]["sampling_frequency_hz"],
        cutoff=filt["highpass_cutoff_hz"],
        device=torch.device("cpu"),
    )
    producer.artifact_threshold = np.inf
    producer.whiten_mat = None
    producer.dshift = None
    producer.device = torch.device("cpu")
    tensor = torch.from_numpy(np.ascontiguousarray(values.T)).float()
    return BinaryFiltered.filter(producer, tensor).numpy(force=True).T


def compare_request(
    arm: str,
    name: str,
    raw: np.ndarray,
    wrapped: Any,
    relative_start: int,
    relative_end: int,
    padded_global_start: int,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    lazy = wrapped.get_traces(start_frame=relative_start, end_frame=relative_end)
    eager = eager_filter(raw[relative_start:relative_end], contract)
    mismatch = lazy != eager
    count = int(np.count_nonzero(mismatch))
    first = None
    if count:
        first = [int(v) for v in np.argwhere(mismatch)[0]]
    error = np.abs(lazy.astype(np.float64) - eager.astype(np.float64))
    result = {
        "arm": arm,
        "request": name,
        "global_frames": [
            padded_global_start + relative_start,
            padded_global_start + relative_end,
        ],
        "shape": list(lazy.shape),
        "dtype": str(lazy.dtype),
        "lazy_sha256": array_sha256(lazy),
        "eager_sha256": array_sha256(eager),
        "finite": bool(np.isfinite(lazy).all() and np.isfinite(eager).all()),
        "mismatch_count": count,
        "max_abs_error": float(error.max(initial=0.0)),
        "first_mismatch_index": first,
    }
    expected_shape = [relative_end - relative_start, contract["input"]["channels"]]
    if (
        result["shape"] != expected_shape
        or result["dtype"] != "float32"
        or not result["finite"]
        or count
        or result["lazy_sha256"] != result["eager_sha256"]
    ):
        raise RuntimeError(f"primary equality failed: {result}")
    return result, lazy


def check_dartsort_boundary(wrapped: Any, output: Path) -> dict[str, Any]:
    import dartsort
    from dartsort.util.main_util import ds_all_to_workdir, ds_will_copy_recording
    from dartsort.util.preprocess_util import preprocess

    cfg = dartsort.DARTsortInternalConfig(
        preprocessing="none", copy_recording_to_tmpdir="no", work_in_tmpdir=False
    )
    copy_flag = ds_will_copy_recording(cfg)
    preprocessed = preprocess(wrapped, cfg.preprocessing, cfg.preprocessing_dtype)
    prepared, work_dir = ds_all_to_workdir(
        internal_cfg=cfg,
        output_dir=output,
        work_dir=None,
        recording=preprocessed,
        overwrite=False,
    )
    result = {
        "copy_flag": bool(copy_flag),
        "preprocess_same_object": preprocessed is wrapped,
        "workdir_same_object": prepared is wrapped,
        "work_dir": work_dir,
    }
    if result != {
        "copy_flag": False,
        "preprocess_same_object": True,
        "workdir_same_object": True,
        "work_dir": None,
    }:
        raise RuntimeError(f"DARTsort no-copy boundary failed: {result}")
    return result


def run_arrays(
    contract: dict[str, Any],
    output: Path,
    arrays: Iterable[tuple[dict[str, Any], np.ndarray]],
    geometry: np.ndarray,
    *,
    real_mode: bool,
) -> dict[str, Any]:
    from npx_preprocessing.motion import KilosortPrewhiteningRecording
    from spikeinterface.core import NumpyRecording

    rows = []
    positive_controls = []
    dartsort_checks = []
    array_count = 0
    cached_input_bytes = 0
    fs = float(contract["input"]["sampling_frequency_hz"])
    for window, raw in arrays:
        array_count += 1
        cached_input_bytes += raw.nbytes
        padded_start, padded_end = window["padded_frames"]
        if raw.shape != (padded_end - padded_start, contract["input"]["channels"]):
            raise RuntimeError(f"wrong cached shape for {window['name']}: {raw.shape}")
        rec = NumpyRecording(
            raw,
            sampling_frequency=fs,
            t_starts=[padded_start / fs],
            channel_ids=np.arange(raw.shape[1]),
        )
        rec.set_channel_locations(geometry)
        wrapped = KilosortPrewhiteningRecording(
            rec,
            channel_ids=np.arange(raw.shape[1]),
            highpass_cutoff_hz=contract["filter"]["highpass_cutoff_hz"],
            do_car=contract["filter"]["global_lower_median_car"],
            invert_sign=contract["filter"]["invert_sign"],
            scale=contract["input"]["kilosort_scale"],
            shift=contract["input"]["kilosort_shift"],
            device="cpu",
        )
        if wrapped.sample_index_to_time(0) != padded_start / fs:
            raise RuntimeError("local acquisition origin mismatch")
        if not np.array_equal(wrapped.channel_ids, np.arange(raw.shape[1])):
            raise RuntimeError("channel ID/order mismatch")
        if not np.array_equal(wrapped.get_channel_locations(), geometry):
            raise RuntimeError("geometry mismatch")
        dartsort_checks.append(check_dartsort_boundary(wrapped, output))
        padded_lazy = None
        central_lazy = None
        for request in contract["requests"]:
            row, lazy = compare_request(
                window["name"],
                request["name"],
                raw,
                wrapped,
                request["relative_start"],
                request["relative_end"],
                padded_start,
                contract,
            )
            rows.append(row)
            if request["name"] == "padded_full":
                padded_lazy = lazy
            if request["name"] == "central_full":
                central_lazy = lazy
        if window["name"] == "transition":
            request = contract["transition_extra_request"]
            row, _ = compare_request(
                window["name"],
                request["name"],
                raw,
                wrapped,
                request["relative_start"],
                request["relative_end"],
                padded_start,
                contract,
            )
            rows.append(row)
        assert padded_lazy is not None and central_lazy is not None
        cropped = padded_lazy[60000:359998]
        diff = cropped != central_lazy
        positive = {
            "arm": window["name"],
            "comparison": "padded_filter_cropped_core_vs_core_filtered_alone",
            "mismatch_count": int(np.count_nonzero(diff)),
            "max_abs_error": float(
                np.abs(cropped.astype(np.float64) - central_lazy.astype(np.float64)).max(
                    initial=0.0
                )
            ),
        }
        if positive["mismatch_count"] == 0:
            raise RuntimeError(f"positive control did not differ: {positive}")
        positive_controls.append(positive)
    if array_count != len(contract["windows"]):
        raise RuntimeError(f"expected {len(contract['windows'])} windows, got {array_count}")
    return {
        "verdict": "PASS_REAL_EQUALITY" if real_mode else "PASS_SYNTHETIC_SELF_TEST",
        "requests": rows,
        "positive_controls": positive_controls,
        "dartsort_boundary": dartsort_checks,
        "cached_input_bytes": cached_input_bytes,
    }


def durable_output_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.iterdir() if path.is_file())


def seal_terminal(
    output: Path,
    terminal_name: str,
    terminal_value: Any,
    status: str,
    max_bytes: int,
) -> None:
    if terminal_name not in {"RESULT.json", "FAILURE.json"}:
        raise ValueError("invalid terminal member")
    if any((output / name).exists() for name in ("RESULT.json", "FAILURE.json")):
        raise RuntimeError("terminal member already exists")
    terminal_bytes = canonical_bytes(terminal_value)
    members = {}
    for path in sorted(output.iterdir()):
        if path.name in {"MANIFEST.json", "COMPLETE.json"} or not path.is_file():
            continue
        members[path.name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    members[terminal_name] = {
        "size": len(terminal_bytes),
        "sha256": sha256_bytes(terminal_bytes),
    }
    manifest_bytes = canonical_bytes(
        {"schema": "immutable-evidence-manifest-v1", "status": status, "members": members}
    )
    complete_bytes = canonical_bytes(
        {
            "schema": "immutable-evidence-complete-v1",
            "status": status,
            "manifest_sha256": sha256_bytes(manifest_bytes),
        }
    )
    projected = (
        durable_output_bytes(output)
        + len(terminal_bytes)
        + len(manifest_bytes)
        + len(complete_bytes)
    )
    if projected > max_bytes:
        raise RuntimeError(f"sealed persistent output would exceed bound: {projected}")
    (output / terminal_name).write_bytes(terminal_bytes)
    (output / "MANIFEST.json").write_bytes(manifest_bytes)
    (output / "COMPLETE.json").write_bytes(complete_bytes)


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


class TemporaryBudget:
    def __init__(self, root: Path, max_bytes: int) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self.stop.wait(0.1):
            if directory_bytes(self.root) > self.max_bytes:
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def __enter__(self) -> "TemporaryBudget":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()


def install_resource_guards(contract: dict[str, Any], temp_root: Path) -> TemporaryBudget:
    resources = contract["resources"]
    memory = int(resources["memory_max_bytes"])
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    hard = memory if current_hard == resource.RLIM_INFINITY else min(current_hard, memory)
    resource.setrlimit(resource.RLIMIT_AS, (min(memory, hard), hard))
    file_max = int(resources["temporary_bytes_max"])
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_max, file_max))

    def fail_limit(_signum: int, _frame: object) -> None:
        raise RuntimeError("active wall or temporary-byte resource limit exceeded")

    signal.signal(signal.SIGALRM, fail_limit)
    signal.signal(signal.SIGUSR1, fail_limit)
    signal.setitimer(signal.ITIMER_REAL, float(resources["wall_seconds_max"]))
    return TemporaryBudget(temp_root, file_max)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extra-kilosort-site", required=True)
    parser.add_argument("--review-complete")
    parser.add_argument("--expected-review-complete-sha256")
    parser.add_argument("--dispatch-receipt")
    parser.add_argument("--expected-dispatch-receipt-sha256")
    parser.add_argument("--resource-receipt")
    parser.add_argument("--expected-resource-receipt-sha256")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--execute-real", action="store_true")
    args = parser.parse_args()

    kilosort_site = Path(args.extra_kilosort_site).resolve()
    if not kilosort_site.is_dir():
        raise SystemExit("extra Kilosort site-packages directory is absent")
    sys.path.append(str(kilosort_site))

    contract_path = Path(args.contract).resolve()
    if sha256_file(contract_path) != args.expected_contract_sha256:
        raise SystemExit("contract hash mismatch before output creation")
    contract = json.loads(contract_path.read_text())
    runner_path = Path(__file__).resolve()
    output = Path(args.output).resolve()
    if sha256_file(runner_path) != contract["source"]["runner_sha256"]:
        raise SystemExit("runner hash mismatch before output creation")
    release = None
    if args.execute_real:
        source_contract = contract["source"]
        if Path(sys.executable).resolve() != Path(source_contract["expected_h5_interpreter"]):
            raise SystemExit("real mode interpreter path mismatch")
        if REPOSITORY_ROOT != Path(source_contract["expected_h5_repository"]):
            raise SystemExit("real mode repository path mismatch")
        if runner_path != Path(source_contract["expected_h5_runner_path"]):
            raise SystemExit("real mode runner path mismatch")
        if kilosort_site != Path(source_contract["expected_h5_kilosort_site"]):
            raise SystemExit("real mode Kilosort site-packages path mismatch")
        if contract_path != Path(source_contract["expected_h5_contract_path"]):
            raise SystemExit("real mode contract path mismatch")
        if output != Path(source_contract["expected_h5_output_path"]):
            raise SystemExit("real mode output path mismatch")
        required_gate_args = (
            args.review_complete,
            args.expected_review_complete_sha256,
            args.dispatch_receipt,
            args.expected_dispatch_receipt_sha256,
            args.resource_receipt,
            args.expected_resource_receipt_sha256,
        )
        if not all(required_gate_args):
            raise SystemExit("real mode requires exact review and coordinator-dispatch receipts")
        review_path = Path(args.review_complete).resolve()
        dispatch_path = Path(args.dispatch_receipt).resolve()
        resource_path = Path(args.resource_receipt).resolve()
        if sha256_file(review_path) != args.expected_review_complete_sha256:
            raise SystemExit("focused-review COMPLETE hash mismatch")
        if sha256_file(dispatch_path) != args.expected_dispatch_receipt_sha256:
            raise SystemExit("coordinator-dispatch receipt hash mismatch")
        if sha256_file(resource_path) != args.expected_resource_receipt_sha256:
            raise SystemExit("managed-resource receipt hash mismatch")
        review = json.loads(review_path.read_text())
        dispatch = json.loads(dispatch_path.read_text())
        resource_receipt = json.loads(resource_path.read_text())
        release_contract = contract["release"]
        if review.get("schema") != release_contract["review_schema"]:
            raise SystemExit("focused review schema mismatch")
        if review.get("status") != release_contract["review_status"]:
            raise SystemExit("focused review status mismatch")
        if not review.get("manifest_sha256"):
            raise SystemExit("focused review COMPLETE lacks manifest_sha256")
        if review.get("reviewed_contract_sha256") != args.expected_contract_sha256:
            raise SystemExit("focused review contract binding mismatch")
        if review.get("reviewed_runner_sha256") != contract["source"]["runner_sha256"]:
            raise SystemExit("focused review runner binding mismatch")
        if dispatch.get("task") != release_contract["dispatch_task"]:
            raise SystemExit("coordinator dispatch task mismatch")
        if dispatch.get("status") != release_contract["dispatch_status"]:
            raise SystemExit("coordinator dispatch status mismatch")
        if dispatch.get("contract_sha256") != args.expected_contract_sha256:
            raise SystemExit("coordinator dispatch contract hash mismatch")
        if dispatch.get("runner_sha256") != contract["source"]["runner_sha256"]:
            raise SystemExit("coordinator dispatch runner hash mismatch")
        if dispatch.get("review_complete_sha256") != args.expected_review_complete_sha256:
            raise SystemExit("coordinator dispatch review binding mismatch")
        if dispatch.get("resource_receipt_sha256") != args.expected_resource_receipt_sha256:
            raise SystemExit("coordinator dispatch resource binding mismatch")
        expected_dispatch_values = {
            "interpreter_path": str(Path(sys.executable).resolve()),
            "runner_path": str(runner_path),
            "contract_path": str(contract_path),
            "output_path": str(output),
            "extra_kilosort_site": str(kilosort_site),
            "review_complete_path": str(review_path),
            "resource_receipt_path": str(resource_path),
            "execution_mode": "execute-real",
        }
        for key, value in expected_dispatch_values.items():
            if dispatch.get(key) != value:
                raise SystemExit(f"coordinator dispatch invocation mismatch for {key}")
        expected_resource_receipt = {
            "schema": release_contract["resource_receipt_schema"],
            "status": release_contract["resource_receipt_status"],
            "contract_sha256": args.expected_contract_sha256,
            "runner_sha256": contract["source"]["runner_sha256"],
            "memory_max_bytes": contract["resources"]["memory_max_bytes"],
            "wall_seconds_max": contract["resources"]["wall_seconds_max"],
            "cpu_threads_max": contract["resources"]["cpu_threads_max"],
            "temporary_bytes_max": contract["resources"]["temporary_bytes_max"],
            "persistent_output_bytes_max": contract["resources"]["persistent_output_bytes_max"],
            "gpu": contract["resources"]["gpu"],
            "interpreter_path": str(Path(sys.executable).resolve()),
            "runner_path": str(runner_path),
            "contract_path": str(contract_path),
            "output_path": str(output),
            "extra_kilosort_site": str(kilosort_site),
            "outer_receipt_path": source_contract["expected_h5_outer_receipt_path"],
        }
        for key, value in expected_resource_receipt.items():
            if resource_receipt.get(key) != value:
                raise SystemExit(f"managed-resource receipt mismatch for {key}")
        if not resource_receipt.get("manager_job_id"):
            raise SystemExit("managed-resource receipt lacks manager_job_id")
        release = {
            "review_complete": str(review_path),
            "review_complete_sha256": args.expected_review_complete_sha256,
            "dispatch_receipt": str(dispatch_path),
            "dispatch_receipt_sha256": args.expected_dispatch_receipt_sha256,
            "resource_receipt": str(resource_path),
            "resource_receipt_sha256": args.expected_resource_receipt_sha256,
        }
    output.mkdir(parents=True, exist_ok=False)
    canonical_write(
        output / "INTERPRETATION_FREEZE.json",
        {
            "decision": contract["decision"],
            "acceptance": contract["acceptance"],
            "mismatch_interpretation": contract["mismatch_interpretation"],
            "contract_sha256": args.expected_contract_sha256,
            "mode": "self_test" if args.self_test else "real",
        },
    )
    (output / "CONTRACT_COPY.json").write_bytes(contract_path.read_bytes())
    started = time.monotonic()
    temp_root = Path(tempfile.mkdtemp(prefix="candidate3_h5_equality_v2_"))
    os.environ["TMPDIR"] = str(temp_root)
    os.environ["NUMBA_CACHE_DIR"] = str(temp_root / "numba")
    tempfile.tempdir = str(temp_root)
    budget = install_resource_guards(contract, temp_root)
    try:
        with budget:
            provenance = imports_and_sources(contract)
            provenance["runner_sha256"] = sha256_file(runner_path)
            provenance["mode"] = "self_test" if args.self_test else "real"
            provenance["release"] = release
            provenance["argv"] = sys.argv
            provenance["temporary_root"] = str(temp_root)
            canonical_write(output / "PROVENANCE.json", provenance)
            import torch

            torch.set_num_threads(contract["resources"]["cpu_threads_max"])
            torch.set_num_interop_threads(1)
            if args.self_test:
                rng = np.random.default_rng(81731)
                n_channels = 384

                def arrays() -> Iterable[tuple[dict[str, Any], np.ndarray]]:
                    for window in contract["windows"]:
                        n = window["padded_request_length"]
                        raw = rng.integers(-600, 601, size=(n, n_channels), dtype=np.int16)
                        yield window, raw

                geometry = np.c_[32.0 * (np.arange(n_channels) % 2), 20.0 * np.arange(n_channels)]
            else:
                source = Path(contract["input"]["path"])
                if not source.is_file() or source.stat().st_size != contract["input"]["bytes"]:
                    raise RuntimeError("accepted binary stat/size mismatch")
                bindings = Path(contract["input"]["input_bindings_path"])
                recording_manifest = Path(contract["input"]["recording_manifest_path"])
                geometry_path = Path(contract["input"]["geometry_path"])
                for path, expected in (
                    (bindings, contract["input"]["input_bindings_sha256"]),
                    (recording_manifest, contract["input"]["recording_manifest_sha256"]),
                    (geometry_path, contract["input"]["geometry_sha256"]),
                ):
                    if sha256_file(path) != expected:
                        raise RuntimeError(f"compact input hash mismatch: {path}")
                geometry = np.load(geometry_path, allow_pickle=False)
                if list(geometry.shape) != contract["input"]["geometry_shape"]:
                    raise RuntimeError("geometry shape mismatch")
                mm = np.memmap(
                    source,
                    mode="r",
                    dtype=np.int16,
                    shape=(contract["input"]["frames_half_open"][1], contract["input"]["channels"]),
                )

                def arrays() -> Iterable[tuple[dict[str, Any], np.ndarray]]:
                    for window in contract["windows"]:
                        start, end = window["padded_frames"]
                        yield window, np.array(mm[start:end], copy=True)

            result = run_arrays(
                contract,
                output,
                arrays(),
                geometry,
                real_mode=args.execute_real,
            )
            result["temporary_bytes_final"] = directory_bytes(temp_root)
        result["wall_seconds"] = time.monotonic() - started
        result["binary_read_bytes"] = 0 if args.self_test else result["cached_input_bytes"]
        if result["wall_seconds"] > contract["resources"]["wall_seconds_max"]:
            raise RuntimeError("wall limit exceeded")
        if result["binary_read_bytes"] > contract["resources"]["binary_read_bytes_max"]:
            raise RuntimeError("binary read bound exceeded")
        seal_terminal(
            output,
            "RESULT.json",
            result,
            result["verdict"],
            contract["resources"]["persistent_output_bytes_max"],
        )
    except Exception as exc:
        seal_terminal(
            output,
            "FAILURE.json",
            {
                "status": "FAILED_PRESERVED_NO_RETRY",
                "type": type(exc).__name__,
                "message": str(exc),
                "wall_seconds": time.monotonic() - started,
            },
            "FAILED_PRESERVED_NO_RETRY",
            contract["resources"]["persistent_output_bytes_max"],
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
