"""Frozen field and input validation for the imec1 Part 2b full-session run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    partial.replace(path)


def round_half_away(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def mask_membership(times: np.ndarray, mask_path: Path) -> np.ndarray:
    inside = np.zeros(times.size, dtype=bool)
    with Path(mask_path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            inside |= (times >= float(row["start_s"])) & (times < float(row["end_s"]))
    return inside


def load_and_verify_lattice(config: dict, repo_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    spec = config["field"]
    lattice_path = repo_root / spec["lattice_csv"]
    if sha256(lattice_path) != spec["lattice_sha256"]:
        raise RuntimeError("frozen lattice hash differs")
    if sha256(Path(spec["npz_path"])) != spec["npz_sha256"]:
        raise RuntimeError("motion-field hash differs")
    if sha256(Path(spec["canonical_mask_path"])) != spec["canonical_mask_sha256"]:
        raise RuntimeError("canonical-mask hash differs")
    table = np.genfromtxt(lattice_path, delimiter=",", names=True, dtype=None, encoding="utf8")
    required = {"knot_index", "time_s", "field_um", "reference_median_um", "inside_canonical_mask", "q_um"}
    if set(table.dtype.names or ()) != required or table.size != 41895:
        raise RuntimeError("unexpected lattice schema or length")
    time_s = np.asarray(table["time_s"], dtype=np.float64)
    q_um = np.asarray(table["q_um"], dtype=np.float64)
    if not np.array_equal(np.asarray(table["knot_index"], dtype=np.int64), np.arange(time_s.size)):
        raise RuntimeError("lattice knot order differs")
    if not np.allclose(time_s, np.arange(time_s.size) * spec["cell_width_s"], rtol=0, atol=1e-12):
        raise RuntimeError("lattice time grid differs")
    references = np.unique(np.asarray(table["reference_median_um"], dtype=np.float64))
    # The authoritative CSV was deliberately serialized at nine decimal places;
    # retain the full-precision producer reference separately in the contract.
    if references.size != 1 or not np.isclose(references[0], spec["reference_median_um"], rtol=0, atol=5e-10):
        raise RuntimeError("frozen reference offset differs")
    with np.load(spec["npz_path"], allow_pickle=False) as field:
        source_time = np.asarray(field["time_s"], dtype=np.float64)
        displacement = np.asarray(field["displacement_um"], dtype=np.float64)
        sign = str(field["sign_convention"])
    if displacement.shape != (time_s.size, 1) or not np.array_equal(source_time, time_s):
        raise RuntimeError("field axes differ from frozen lattice")
    if sign != "corrected = observed - displacement":
        raise RuntimeError("motion-field sign differs")
    inside = mask_membership(time_s, Path(spec["canonical_mask_path"]))
    frozen_inside = np.asarray(table["inside_canonical_mask"], dtype=bool)
    if not np.array_equal(inside, frozen_inside):
        raise RuntimeError("canonical-mask membership differs")
    expected_q = np.zeros(time_s.size, dtype=np.float64)
    expected_q[inside] = spec["period_um"] * round_half_away(
        (displacement[:, 0][inside] - spec["reference_median_um"]) / spec["period_um"]
    )
    if not np.array_equal(expected_q, q_um):
        raise RuntimeError("frozen production-order rounding does not reproduce lattice")
    values, counts = np.unique(q_um, return_counts=True)
    audit = {
        "reference_median_um": float(spec["reference_median_um"]),
        "serialized_reference_median_um": float(references[0]),
        "time_bins": int(time_s.size),
        "masked_bins": int(inside.sum()),
        "states_um": values.tolist(),
        "state_bin_counts": dict(zip(map(str, values.tolist()), map(int, counts.tolist()))),
        "field_lattice_reproduced": True,
    }
    return time_s, q_um, audit


def verify_source(config: dict):
    import spikeinterface as si

    spec = config["source"]
    recording_dir = Path(spec["recording_dir"])
    manifest_path = recording_dir / "rescue_recording_manifest.json"
    if sha256(manifest_path) != spec["manifest_sha256"]:
        raise RuntimeError("source manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    for key in ("num_samples", "num_channels", "sampling_frequency_hz", "dtype", "graph", "explicit_bad_channel_ids"):
        if manifest.get(key) != spec[key]:
            raise RuntimeError(f"source manifest {key} differs")
    binary = recording_dir / "traces_cached_seg0.raw"
    if binary.stat().st_size != spec["binary_bytes"]:
        raise RuntimeError("source binary byte count differs")
    recording = si.load(recording_dir)
    if recording.get_num_samples() != spec["num_samples"] or recording.get_num_channels() != spec["num_channels"]:
        raise RuntimeError("loaded source signature differs")
    if str(recording.get_dtype()) != spec["dtype"]:
        raise RuntimeError("loaded source dtype differs")
    ids = list(map(str, recording.get_channel_ids()))
    if ids != [f"imec1.ap#AP{i}" for i in range(384)]:
        raise RuntimeError("source channels are not exact imec1 production order")
    return recording, manifest


def verify_lineage_bytes(entry: dict, release_bytes: bytes | None, prepared_bytes: bytes, runtime_bytes: bytes) -> dict:
    expected_release = entry.get("release_sha256")
    actual_release = None if release_bytes is None else sha256_bytes(release_bytes)
    if actual_release != expected_release:
        raise RuntimeError(f"release source hash differs for {entry.get('path')}")
    actual_prepared = sha256_bytes(prepared_bytes)
    if actual_prepared != entry.get("prepared_sha256"):
        raise RuntimeError(f"prepared source hash differs for {entry.get('path')}")
    actual_runtime = sha256_bytes(runtime_bytes)
    if actual_runtime != entry.get("runtime_sha256"):
        raise RuntimeError(f"installed runtime source hash differs for {entry.get('path')}")
    return {
        "path": entry["path"], "release_sha256": actual_release,
        "prepared_sha256": actual_prepared, "runtime_sha256": actual_runtime,
        "transformation": entry["transformation"],
    }


def verify_part2a_receipts(config: dict, result_path: Path, service_result_path: Path) -> dict:
    spec = config["part2a_gate"]
    expected_paths = (Path(spec["result_path"]), Path(spec["service_result_path"]))
    actual_paths = (Path(result_path), Path(service_result_path))
    if actual_paths != expected_paths:
        raise RuntimeError("Part 2a receipt path differs from frozen contract")
    for path, size_key, hash_key in (
        (actual_paths[0], "result_bytes", "result_sha256"),
        (actual_paths[1], "service_result_bytes", "service_result_sha256"),
    ):
        if not path.is_file():
            raise RuntimeError(f"missing sealed Part 2a receipt: {path}")
        if path.stat().st_size != spec[size_key] or sha256(path) != spec[hash_key]:
            raise RuntimeError(f"sealed Part 2a receipt bytes/hash differ: {path}")
    result = json.loads(actual_paths[0].read_text())
    service = json.loads(actual_paths[1].read_text())
    required_result = {
        "schema": spec["result_schema"], "status": spec["result_status"],
        "recording_sha256": config["source"]["binary_sha256"],
    }
    if any(result.get(key) != value for key, value in required_result.items()):
        raise RuntimeError("Part 2a main result completion semantics differ")
    required_service = {
        "schema": spec["service_result_schema"], "success": spec["service_success"],
        "main_exec_status": spec["main_exec_status"],
        "main_result_status": spec["main_result_status"],
        "systemd_overall_result": spec["preserved_systemd_overall_result"],
        "post_stop_exec_status": spec["preserved_post_stop_exec_status"],
    }
    if any(service.get(key) != value for key, value in required_service.items()):
        raise RuntimeError("Part 2a service recovery/failure semantics differ")
    return {
        "result_path": str(actual_paths[0]), "result_bytes": spec["result_bytes"],
        "result_sha256": spec["result_sha256"], "result_status": result["status"],
        "service_result_path": str(actual_paths[1]), "service_result_bytes": spec["service_result_bytes"],
        "service_result_sha256": spec["service_result_sha256"], "service_success": service["success"],
        "main_exec_status": service["main_exec_status"],
        "preserved_systemd_overall_result": service["systemd_overall_result"],
        "preserved_post_stop_exec_status": service["post_stop_exec_status"],
        "accepted_semantics": spec["accepted_semantics"],
        "preserved_publisher_state": spec["preserved_publisher_state"],
    }


def verify_reviewed_runtime(
    config_path: Path, review_path: Path, installed_unit: Path, *, unit_key: str
) -> dict:
    """Fail unless runtime source and installed units are exactly H1-reviewed."""
    repo = Path(__file__).resolve().parents[1]
    review = json.loads(Path(review_path).read_text())
    if review.get("status") != "PASS_FOR_SINGLE_PRODUCTION_LAUNCH":
        raise RuntimeError("H1 review does not authorize the single production launch")
    if review.get("config_sha256") != sha256(Path(config_path)):
        raise RuntimeError("H1 review config binding differs")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if review.get("source_commit") != head:
        raise RuntimeError("runtime git HEAD differs from H1-reviewed commit")
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"], text=True
    )
    if status:
        raise RuntimeError("runtime worktree is not clean")
    base = "32d04e7d5d8f15e9bd895ff3a4e633f75088b156"
    ancestry = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", base, head])
    if ancestry.returncode != 0:
        raise RuntimeError("reviewed runtime commit is not based on the frozen release")
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-status", f"{base}..{head}"], text=True
    )
    actual_delta = []
    for line in diff.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M", "D"}:
            raise RuntimeError(f"unsupported release delta entry: {line}")
        actual_delta.append({"status": fields[0], "path": fields[1]})
    lineage = review.get("changed_file_lineage")
    if not isinstance(lineage, list):
        raise RuntimeError("H1 receipt omits complete changed-file lineage")
    expected_delta = [{"status": row.get("status"), "path": row.get("path")} for row in lineage]
    if actual_delta != expected_delta:
        raise RuntimeError("runtime release delta differs from H1-reviewed inventory")
    checked_lineage = []
    service_runtime = {
        "systemd/full-session-part2b-imec1-q0-20261004-v1.service": Path("/home/huklaban5/.config/systemd/user/full-session-part2b-imec1-q0-20261004-v1.service"),
        "systemd/full-session-part2b-imec1-production-20261004-v1.service": Path("/home/huklaban5/.config/systemd/user/full-session-part2b-imec1-production-20261004-v1.service"),
    }
    for entry in lineage:
        path = entry["path"]
        release = subprocess.run(
            ["git", "-C", str(repo), "show", f"{base}:{path}"], capture_output=True
        )
        release_bytes = release.stdout if release.returncode == 0 else None
        prepared = subprocess.check_output(["git", "-C", str(repo), "show", f"{head}:{path}"])
        runtime_path = service_runtime.get(path, repo / path)
        if not runtime_path.is_file():
            raise RuntimeError(f"missing installed runtime source: {runtime_path}")
        checked_lineage.append(verify_lineage_bytes(entry, release_bytes, prepared, runtime_path.read_bytes()))
    expected_unit_hash = review.get(unit_key)
    if not isinstance(expected_unit_hash, str) or sha256(Path(installed_unit)) != expected_unit_hash:
        raise RuntimeError("installed service unit differs from H1-reviewed bytes")
    return {
        "source_commit": head,
        "clean_worktree": True,
        "config_sha256": review["config_sha256"],
        "installed_unit": str(Path(installed_unit)),
        "installed_unit_sha256": expected_unit_hash,
        "review_sha256": sha256(Path(review_path)),
        "changed_file_lineage": checked_lineage,
    }
