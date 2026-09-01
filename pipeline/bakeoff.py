"""Guarded sorter-architecture bake-off on accepted unwarped recordings."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PIPELINE_VERSION, fingerprint
from .preprocess import MANIFEST_NAME
from .sorting import SORT_MANIFEST, run_kilosort4


BAKEOFF_SCHEMA = "sorter-architecture-bakeoff-v1"
BAKEOFF_MANIFEST = "bakeoff_sort_manifest.json"


@dataclass(frozen=True)
class SorterCandidate:
    name: str
    architecture: str
    runner: str
    maturity: str
    motion_source: str
    raw_voltage_warp: bool
    pipeline_status: str
    requirement: str | None = None


@dataclass(frozen=True)
class BakeoffWindow:
    name: str
    start_frame: int
    end_frame: int
    source_start_frame: int
    source_end_frame: int
    start_s: float
    duration_s: float
    sampling_frequency_hz: float
    request_digest: str

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def directory_name(self) -> str:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", self.name).strip("-._")
        if not slug:
            slug = "window"
        return f"{slug}-{self.request_digest[:10]}"


CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        SorterCandidate(
            name="ks4_no_motion",
            architecture="static_templates_on_unwarped_voltage",
            runner="accepted_ks4_reference",
            maturity="production_reference",
            motion_source="none",
            raw_voltage_warp=False,
            pipeline_status="runnable",
        ),
        SorterCandidate(
            name="dartsort_native",
            architecture="registered_spike_features_and_drift_aware_model_tracking",
            runner="dartsort_python_api",
            maturity="experimental_challenger",
            motion_source="dartsort_native",
            raw_voltage_warp=False,
            pipeline_status="runnable_when_dependency_isolated",
            requirement=(
                "A dedicated environment with the dartsort package and compatible "
                "CUDA/PyTorch; upstream marks it WIP and not production-recommended"
            ),
        ),
        SorterCandidate(
            name="kiasort",
            architecture="per_neuron_waveform_and_location_tracking",
            runner="kiasort_repository_spikeinterface_wrapper",
            maturity="experimental_challenger",
            motion_source="neuron_specific_tracking",
            raw_voltage_warp=False,
            pipeline_status="runnable_when_repository_is_configured",
            requirement="KIASORT checkout, MATLAB >=2021b, and required MATLAB toolboxes",
        ),
        SorterCandidate(
            name="si_motion_aware_peeler",
            architecture="supplied_motion_moves_templates_during_matching",
            runner="spikeinterface_sorting_component",
            maturity="prototype",
            motion_source="qualified_external_field",
            raw_voltage_warp=False,
            pipeline_status="requires_newer_spikeinterface_and_pipeline_adapter",
            requirement="SpikeInterface TridesclousPeeler with motion_aware and a qualified Motion object",
        ),
        SorterCandidate(
            name="ironclust",
            architecture="anatomical_state_linked_chunk_clustering",
            runner="spikeinterface_external_sorter",
            maturity="optional_independent_control",
            motion_source="ironclust_native",
            raw_voltage_warp=False,
            pipeline_status="optional_not_installed",
            requirement="IronClust, MATLAB, and its required toolboxes",
        ),
    )
}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_kiasort_installation(path: Path | None) -> dict[str, Any]:
    """Resolve the complete upstream KIASORT GUI and SpikeInterface installation."""
    if path is None:
        return {"configured": False, "reason": "KIASORT_PATH is unset"}
    root = Path(path).expanduser().resolve()
    wrapper = root / "SpikeInterface_wrapper" / "kiasort_spikeinterface.py"
    if not wrapper.exists():
        return {
            "configured": False,
            "root": str(root),
            "reason": f"Missing upstream wrapper: {wrapper}",
        }
    gui = root / "kiaSort.m"
    if not gui.exists():
        return {
            "configured": False,
            "root": str(root),
            "wrapper": str(wrapper),
            "reason": f"Missing graphical MATLAB entrypoint: {gui}",
        }
    # Current upstream contains a root entrypoint and a mirrored No_GUI copy.
    # Prefer the root copy because adding the complete checkout to MATLAB's
    # path makes that the documented, deterministic invocation target.
    preferred = root / "run_kiasort_nogui.m"
    entrypoints = sorted(root.rglob("run_kiasort_nogui.m"))
    entrypoint = preferred if preferred.exists() else (
        entrypoints[0] if len(entrypoints) == 1 else None
    )
    if entrypoint is None:
        return {
            "configured": False,
            "root": str(root),
            "wrapper": str(wrapper),
            "gui": str(gui),
            "reason": f"Could not choose run_kiasort_nogui.m from {len(entrypoints)} copies",
        }
    git_commit = None
    tracked_diff_sha256 = None
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        diff_result = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "--no-ext-diff"],
            check=True,
            capture_output=True,
        )
        git_commit = commit_result.stdout.strip()
        tracked_diff_sha256 = hashlib.sha256(diff_result.stdout).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "configured": True,
        "root": str(root),
        "wrapper": str(wrapper),
        "gui": str(gui),
        "nogui_dir": str(entrypoint.parent),
        "wrapper_sha256": _sha256_file(wrapper),
        "gui_entrypoint_sha256": _sha256_file(gui),
        "nogui_entrypoint_sha256": _sha256_file(entrypoint),
        "nogui_entrypoint_candidates": [str(item) for item in entrypoints],
        "git_commit": git_commit,
        "tracked_diff_sha256": tracked_diff_sha256,
    }


def inspect_bakeoff_environment() -> dict[str, Any]:
    """Feature-detect candidate availability without treating plans as runs."""
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spikeglx-bakeoff-numba-cache")
    si_version = _package_version("spikeinterface")
    tdc_motion_aware = False
    tdc_error = None
    if si_version is not None:
        try:
            from spikeinterface.sortingcomponents.matching.tdc_peeler import (
                TridesclousPeeler,
            )

            tdc_motion_aware = (
                "motion_aware"
                in inspect.signature(TridesclousPeeler.__init__).parameters
            )
        except Exception as exc:  # environment audit must remain reportable
            tdc_error = f"{type(exc).__name__}: {exc}"
    ironclust_path = os.environ.get("IRONCLUST_PATH")
    kiasort = resolve_kiasort_installation(os.environ.get("KIASORT_PATH"))
    return {
        "spikeinterface_version": si_version,
        "dartsort_version": _package_version("dartsort"),
        "dartsort_importable": importlib.util.find_spec("dartsort") is not None,
        "kiasort": kiasort,
        "matlab_executable": shutil.which("matlab"),
        "ironclust_path": ironclust_path,
        "ironclust_configured": bool(ironclust_path and Path(ironclust_path).exists()),
        "tdc_motion_aware_available": tdc_motion_aware,
        "tdc_feature_detection_error": tdc_error,
    }


def _load_recording_manifest(recording_dir: Path) -> dict[str, Any]:
    path = Path(recording_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing accepted recording manifest: {path}")
    manifest = json.loads(path.read_text())
    if not manifest.get("complete"):
        raise RuntimeError("Recording manifest is not marked complete")
    return manifest


def resolve_bakeoff_window(
    manifest: dict[str, Any],
    *,
    name: str = "full_recording",
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> BakeoffWindow:
    """Resolve a reproducible interval relative to the accepted recording."""
    sampling_frequency = float(manifest["sampling_frequency_hz"])
    total_frames = int(manifest["selected_end_frame"] - manifest["selected_start_frame"])
    if not np.isfinite(start_s) or start_s < 0:
        raise ValueError("Window start_s must be finite and nonnegative")
    start_frame = int(round(float(start_s) * sampling_frequency))
    if duration_s is None:
        end_frame = total_frames
    else:
        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Window duration_s must be finite and positive")
        end_frame = start_frame + int(round(float(duration_s) * sampling_frequency))
    if start_frame >= total_frames or end_frame > total_frames or end_frame <= start_frame:
        raise ValueError(
            f"Window [{start_frame}, {end_frame}) is outside accepted recording "
            f"[0, {total_frames})"
        )
    selected_start = int(manifest["selected_start_frame"])
    request = {
        "recording_request_digest": manifest["request_digest"],
        "name": str(name),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "sampling_frequency_hz": sampling_frequency,
    }
    return BakeoffWindow(
        name=str(name),
        start_frame=start_frame,
        end_frame=end_frame,
        source_start_frame=selected_start + start_frame,
        source_end_frame=selected_start + end_frame,
        start_s=start_frame / sampling_frequency,
        duration_s=(end_frame - start_frame) / sampling_frequency,
        sampling_frequency_hz=sampling_frequency,
        request_digest=fingerprint(request),
    )


def _window_dict(window: BakeoffWindow) -> dict[str, Any]:
    result = asdict(window)
    result["frame_count"] = window.frame_count
    result["directory_name"] = window.directory_name
    return result


def _slice_recording(recording, window: BakeoffWindow):
    if recording.get_num_samples() < window.end_frame:
        raise RuntimeError("Loaded recording is shorter than the requested bake-off window")
    if window.start_frame == 0 and window.end_frame == recording.get_num_samples():
        return recording
    return recording.frame_slice(
        start_frame=window.start_frame,
        end_frame=window.end_frame,
    )


def _load_si_extractor(path: Path):
    """Load an extractor across the SI 0.102 production and 0.104 adapter APIs."""
    try:
        from spikeinterface.core import load
    except ImportError:
        from spikeinterface.core import load_extractor as load
    return load(path)


def build_bakeoff_plan(
    recording_dir: Path,
    candidates: Iterable[str] = (
        "ks4_no_motion",
        "dartsort_native",
        "kiasort",
        "si_motion_aware_peeler",
    ),
    *,
    window_name: str = "full_recording",
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Describe a controlled architecture comparison without running sorters."""
    manifest = _load_recording_manifest(Path(recording_dir))
    window = resolve_bakeoff_window(
        manifest, name=window_name, start_s=start_s, duration_s=duration_s
    )
    names = list(candidates)
    unknown = sorted(set(names) - CANDIDATES.keys())
    if unknown:
        raise ValueError(f"Unknown bake-off candidates: {unknown}")
    environment = inspect_bakeoff_environment()
    rows = []
    for name in names:
        row = asdict(CANDIDATES[name])
        if name == "dartsort_native":
            row["runnable_now"] = bool(environment["dartsort_importable"])
        elif name == "kiasort":
            row["environment_ready"] = bool(
                environment["kiasort"]["configured"]
                and environment["matlab_executable"]
            )
            row["runnable_now"] = row["environment_ready"]
        elif name == "si_motion_aware_peeler":
            row["component_available"] = bool(environment["tdc_motion_aware_available"])
            row["runnable_now"] = False
        elif name == "ironclust":
            row["environment_ready"] = bool(
                environment["ironclust_configured"] and environment["matlab_executable"]
            )
            row["runnable_now"] = False
        else:
            row["runnable_now"] = True
        rows.append(row)
    plan = {
        "schema_version": BAKEOFF_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "recording_request_digest": manifest["request_digest"],
        "recording_dir": str(Path(recording_dir).resolve()),
        "selected_frame_count": window.frame_count,
        "sampling_frequency_hz": float(manifest["sampling_frequency_hz"]),
        "window": _window_dict(window),
        "shared_input_policy": {
            "source": "accepted materialized rescue recording",
            "voltage_motion_resampling_allowed": False,
            "sorter_native_frontend_allowed": True,
            "frontend_must_be_recorded": True,
            "interpretation": (
                "This compares complete sorting architectures. Native filtering, "
                "referencing, standardization, and whitening may differ and are audited."
            ),
        },
        "candidates": rows,
        "environment": environment,
        "common_endpoints": [
            "reviewed_event_recovery",
            "refractory_violation_burden",
            "near_coincident_duplicate_burden",
            "longitudinal_waveform_family_continuity",
            "early_late_template_alignment",
            "unit_presence_and_fragmentation",
            "runtime_and_peak_memory",
        ],
        "advancement_rule": (
            "No challenger advances on unit count alone; event recovery and duplicate/"
            "refractory guardrails must pass before continuity gains are interpreted."
        ),
    }
    plan["plan_digest"] = fingerprint(plan)
    return plan


def _validate_recording_bytes(recording_dir: Path, manifest: dict[str, Any]) -> None:
    binaries = list(Path(recording_dir).glob("*.raw")) + list(
        Path(recording_dir).glob("*.bin")
    )
    actual = sum(path.stat().st_size for path in binaries)
    if actual != manifest["expected_binary_bytes"]:
        raise RuntimeError(
            f"Recording bytes changed after acceptance: {actual} != "
            f"{manifest['expected_binary_bytes']}"
        )


def validate_dartsort_output(native_output: Path) -> dict[str, Any]:
    """Validate the documented DARTsort final spike-train artifact."""
    sorting_path = Path(native_output) / "dartsort_sorting.npz"
    if not sorting_path.exists():
        raise RuntimeError(f"DARTsort ended without {sorting_path}")
    with np.load(sorting_path, allow_pickle=False) as values:
        required = {"times_samples", "channels", "labels"}
        missing = required - set(values.files)
        if missing:
            raise RuntimeError(f"DARTsort output is missing arrays: {sorted(missing)}")
        times = np.asarray(values["times_samples"]).reshape(-1)
        channels = np.asarray(values["channels"]).reshape(-1)
        labels = np.asarray(values["labels"]).reshape(-1)
    if not (times.size == channels.size == labels.size):
        raise RuntimeError("DARTsort final per-spike arrays have inconsistent lengths")
    assigned = labels >= 0
    return {
        "final_spike_count": int(times.size),
        "assigned_spike_count": int(np.sum(assigned)),
        "unit_count": int(np.unique(labels[assigned]).size),
    }


def normalize_dartsort_output(
    native_output: Path, output_dir: Path, num_samples: int
) -> dict[str, Any]:
    """Write assigned DARTsort spikes in the common bake-off array contract."""
    with np.load(
        Path(native_output) / "dartsort_sorting.npz", allow_pickle=False
    ) as values:
        raw_times = np.asarray(values["times_samples"]).reshape(-1)
        labels = np.asarray(values["labels"], dtype=np.int64).reshape(-1)
    times = np.asarray(raw_times, dtype=np.int64)
    if not np.all(raw_times == times):
        raise RuntimeError("DARTsort times_samples contains non-integer values")
    assigned = labels >= 0
    times = times[assigned]
    labels = labels[assigned]
    if np.any(times < 0) or np.any(times >= num_samples):
        raise RuntimeError(
            "DARTsort returned spike times outside the accepted recording"
        )
    order = np.argsort(times, kind="stable")
    np.save(Path(output_dir) / "spike_times.npy", times[order])
    np.save(Path(output_dir) / "spike_labels.npy", labels[order])
    return {
        "normalized_spike_count": int(times.size),
        "normalized_arrays": ["spike_times.npy", "spike_labels.npy"],
    }


def _load_upstream_kiasort_wrapper(wrapper_path: Path):
    spec = importlib.util.spec_from_file_location(
        "upstream_kiasort_spikeinterface", wrapper_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load KIASORT wrapper from {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_kiasort"):
        raise RuntimeError("Upstream KIASORT wrapper has no run_kiasort function")
    return module


def _load_kiasort_native_sorting(native_output: Path, sampling_frequency: float):
    """Load the two native arrays used by KIASORT's upstream SI wrapper."""
    import h5py
    import spikeinterface as si

    result_dir = Path(native_output) / "RES_Sorted"
    spike_path = result_dir / "spike_idx.h5"
    label_path = result_dir / "unifiedLabels.h5"
    if not spike_path.exists() or not label_path.exists():
        raise RuntimeError(f"KIASORT ended without native result arrays in {result_dir}")
    with h5py.File(spike_path, "r") as values:
        spike_indices = np.asarray(values["/spike_idx"]).reshape(-1).astype(np.int64)
    with h5py.File(label_path, "r") as values:
        labels = np.asarray(values["/unifiedLabels"]).reshape(-1).astype(np.int64)
    if spike_indices.size != labels.size:
        raise RuntimeError("KIASORT native spike and label arrays have different lengths")
    assigned = labels >= 0
    return si.NumpySorting.from_samples_and_labels(
        samples_list=spike_indices[assigned],
        labels_list=labels[assigned],
        sampling_frequency=float(sampling_frequency),
    )


def _resume_kiasort_matlab(
    native_output: Path,
    matlab_executable: str,
    python_executable: str,
    numba_threads: int,
) -> None:
    """Resume a guarded partial after SI already exported its binary input."""
    script_path = Path(native_output) / "_run_kiasort.m"
    recording_path = Path(native_output) / "recording.dat"
    channel_map_path = Path(native_output) / "channel_map.mat"
    missing = [
        str(path)
        for path in (script_path, recording_path, channel_map_path)
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"KIASORT partial cannot be resumed; missing: {missing}")
    matlab_env = os.environ.copy()
    python_lib = str(Path(python_executable).resolve().parent.parent / "lib")
    existing_library_path = matlab_env.get("LD_LIBRARY_PATH", "")
    matlab_env["LD_LIBRARY_PATH"] = (
        python_lib
        if not existing_library_path
        else python_lib + os.pathsep + existing_library_path
    )
    matlab_env["PYTHONNOUSERSITE"] = "1"
    matlab_env["NUMBA_NUM_THREADS"] = str(numba_threads)
    process = subprocess.Popen(
        [matlab_executable, "-batch", script_path.read_text().replace("\n", " ")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=matlab_env,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="", flush=True)
    returncode = process.wait()
    if returncode:
        raise RuntimeError(
            f"MATLAB exited with code {returncode}\n{''.join(output_lines[-80:])}"
        )


def _normalize_sorting(sorting, output_dir: Path, num_samples: int) -> dict[str, Any]:
    vector = sorting.to_spike_vector()
    times = np.asarray(vector["sample_index"], dtype=np.int64)
    unit_indices = np.asarray(vector["unit_index"], dtype=np.int64)
    unit_ids = np.asarray(sorting.unit_ids)
    try:
        labels = np.asarray(unit_ids[unit_indices], dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Sorter unit IDs must be integer-normalizable") from exc
    if np.any(times < 0) or np.any(times >= num_samples):
        raise RuntimeError("Sorter returned spike times outside the accepted recording")
    order = np.argsort(times, kind="stable")
    times = times[order]
    labels = labels[order]
    np.save(output_dir / "spike_times.npy", times)
    np.save(output_dir / "spike_labels.npy", labels)
    return {
        "final_spike_count": int(times.size),
        "assigned_spike_count": int(times.size),
        "unit_count": int(np.unique(labels).size),
        "normalized_arrays": ["spike_times.npy", "spike_labels.npy"],
    }


def run_kiasort_challenger(
    recording_dir: Path,
    output_dir: Path,
    *,
    kiasort_path: Path | None = None,
    python_executable: Path | str | None = None,
    numba_threads: int = 2,
    channel_start_index: int = 0,
    channel_count: int | None = None,
    matlab_bin: str = "matlab",
    config_overrides: dict[str, Any] | None = None,
    keep_intermediate: bool = False,
    window_name: str = "full_recording",
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Run KIASORT through its upstream SpikeInterface/MATLAB wrapper."""
    recording_dir = Path(recording_dir)
    output_dir = Path(output_dir)
    partial = output_dir.with_name(output_dir.name + ".partial")
    recording_manifest = _load_recording_manifest(recording_dir)
    window = resolve_bakeoff_window(
        recording_manifest, name=window_name, start_s=start_s, duration_s=duration_s
    )
    _validate_recording_bytes(recording_dir, recording_manifest)
    configured_path = kiasort_path or os.environ.get("KIASORT_PATH")
    installation = resolve_kiasort_installation(configured_path)
    if not installation["configured"]:
        raise RuntimeError(f"KIASORT is not configured: {installation['reason']}")
    matlab_executable = shutil.which(matlab_bin)
    if matlab_executable is None:
        raise RuntimeError(f"MATLAB executable is unavailable: {matlab_bin}")
    configured_python = python_executable or os.environ.get(
        "KIASORT_PYTHON_EXECUTABLE"
    )
    if configured_python is None:
        raise RuntimeError(
            "KIASORT requires a MATLAB-compatible Python with UMAP; pass "
            "python_executable or set KIASORT_PYTHON_EXECUTABLE"
        )
    configured_python = str(Path(configured_python).resolve())
    if not Path(configured_python).is_file():
        raise RuntimeError(f"KIASORT Python executable is unavailable: {configured_python}")
    if numba_threads < 1:
        raise ValueError("KIASORT Numba thread count must be positive")
    total_channels = int(recording_manifest["num_channels"])
    selected_channel_count = (
        total_channels - channel_start_index
        if channel_count is None
        else int(channel_count)
    )
    channel_end_index = channel_start_index + selected_channel_count
    if (
        channel_start_index < 0
        or selected_channel_count < 1
        or channel_end_index > total_channels
    ):
        raise ValueError("KIASORT channel selection is outside the recording")
    os.environ["KIASORT_PYTHON_EXECUTABLE"] = configured_python
    os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)
    numba_cache_dir = os.environ.get(
        "NUMBA_CACHE_DIR", "/tmp/kiasort-numba-cache"
    )
    Path(numba_cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = numba_cache_dir
    overrides = dict(config_overrides or {})
    request = {
        "schema_version": BAKEOFF_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "candidate": "kiasort",
        "recording_request_digest": recording_manifest["request_digest"],
        "window": _window_dict(window),
        "wrapper_sha256": installation["wrapper_sha256"],
        "gui_entrypoint_sha256": installation["gui_entrypoint_sha256"],
        "nogui_entrypoint_sha256": installation["nogui_entrypoint_sha256"],
        "kiasort_git_commit": installation["git_commit"],
        "kiasort_tracked_diff_sha256": installation["tracked_diff_sha256"],
        "kiasort_python_executable": configured_python,
        "kiasort_numba_threads": int(numba_threads),
        "channel_selection": {
            "start_index": int(channel_start_index),
            "end_index_exclusive": int(channel_end_index),
            "count": int(selected_channel_count),
            "full_probe": selected_channel_count == total_channels,
        },
        "config_overrides": overrides,
        "keep_intermediate": bool(keep_intermediate),
        "raw_voltage_warp": False,
    }
    request_digest = fingerprint(request)
    accepted_manifest = output_dir / BAKEOFF_MANIFEST
    recovering_partial = partial.exists()
    partial_request_path = partial / "bakeoff_partial_request.json"
    if output_dir.exists():
        if not accepted_manifest.exists():
            raise RuntimeError("Existing KIASORT output lacks an accepted manifest")
        existing = json.loads(accepted_manifest.read_text())
        if existing.get("request_digest") != request_digest:
            raise RuntimeError("Existing KIASORT run belongs to another request")
        return existing
    if recovering_partial:
        if not partial_request_path.exists():
            raise RuntimeError(
                "KIASORT partial predates guarded request receipts and cannot be "
                "resumed automatically"
            )
        partial_request = json.loads(partial_request_path.read_text())
        if partial_request.get("request_digest") != request_digest:
            raise RuntimeError("KIASORT partial belongs to another request")

    recording = _load_si_extractor(recording_dir)
    full_num_samples = int(
        recording_manifest["selected_end_frame"]
        - recording_manifest["selected_start_frame"]
    )
    if recording.get_num_samples() != full_num_samples:
        raise RuntimeError("Loaded recording length differs from its accepted manifest")
    recording = _slice_recording(recording, window)
    selected_channel_ids = recording.channel_ids[
        channel_start_index:channel_end_index
    ]
    recording = recording.select_channels(channel_ids=selected_channel_ids)
    num_samples = window.frame_count
    wrapper = _load_upstream_kiasort_wrapper(Path(installation["wrapper"]))
    partial.mkdir(parents=True, exist_ok=recovering_partial)
    partial_request_path.write_text(
        json.dumps({"request_digest": request_digest, "request": request}, indent=2)
        + "\n"
    )
    native_output = partial / "native_output"
    reused_native_results = False
    if recovering_partial:
        result_dir = native_output / "RES_Sorted"
        reused_native_results = (result_dir / "spike_idx.h5").exists() and (
            result_dir / "unifiedLabels.h5"
        ).exists()
        if not reused_native_results:
            recovery_overrides = dict(overrides)
            recovery_overrides.setdefault(
                "samplingFrequency", recording.get_sampling_frequency()
            )
            recovery_overrides.setdefault("numChannels", recording.get_num_channels())
            recovery_overrides.setdefault("dataType", "int16")
            script_path = native_output / "_run_kiasort.m"
            script_path.write_text(
                wrapper._build_matlab_script(
                    Path(installation["nogui_dir"]),
                    native_output / "recording.dat",
                    native_output,
                    native_output / "channel_map.mat",
                    recovery_overrides,
                )
            )
            _resume_kiasort_matlab(
                native_output, matlab_executable, configured_python, numba_threads
            )
        sorting = _load_kiasort_native_sorting(
            native_output, recording.get_sampling_frequency()
        )
    else:
        sorting = wrapper.run_kiasort(
            recording,
            native_output,
            Path(installation["nogui_dir"]),
            matlab_bin=matlab_executable,
            config_overrides=overrides,
            keep_intermediate=keep_intermediate,
            verbose=True,
        )
    summary = _normalize_sorting(sorting, partial, num_samples)
    receipt = {
        **request,
        "request_digest": request_digest,
        "kiasort_root": installation["root"],
        "matlab_executable": matlab_executable,
        "summary": summary,
        "experimental": True,
        "recovered_partial": recovering_partial,
        "reused_native_results": reused_native_results,
        "complete": True,
    }
    (partial / BAKEOFF_MANIFEST).write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(partial, output_dir)
    return receipt


def accept_ks4_reference(
    recording_dir: Path,
    ks4_dir: Path,
    output_dir: Path,
    *,
    window_name: str = "full_recording",
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Register the existing accepted KS4 baseline as one bake-off condition."""
    recording_manifest = _load_recording_manifest(Path(recording_dir))
    window = resolve_bakeoff_window(
        recording_manifest, name=window_name, start_s=start_s, duration_s=duration_s
    )
    sort_manifest_path = Path(ks4_dir) / SORT_MANIFEST
    if not sort_manifest_path.exists():
        run_kilosort4(Path(recording_dir), Path(ks4_dir))
    sort_manifest = json.loads(sort_manifest_path.read_text())
    if (
        sort_manifest.get("recording_request_digest")
        != recording_manifest["request_digest"]
    ):
        raise RuntimeError("KS4 baseline belongs to another recording")
    request = {
        "schema_version": BAKEOFF_SCHEMA,
        "candidate": "ks4_no_motion",
        "recording_request_digest": recording_manifest["request_digest"],
        "native_sort_request_digest": sort_manifest["request_digest"],
        "window": _window_dict(window),
        "reference_method": "accepted_full_sort_window_extraction",
    }
    manifest = {
        **request,
        "request_digest": fingerprint(request),
        "native_sort_dir": str(Path(ks4_dir).resolve()),
        "raw_voltage_warp": False,
        "complete": True,
    }
    output_dir = Path(output_dir)
    path = output_dir / BAKEOFF_MANIFEST
    partial = output_dir.with_name(output_dir.name + ".partial")
    if partial.exists():
        raise RuntimeError(
            f"Incomplete KS4 bake-off receipt requires inspection: {partial}"
        )
    if output_dir.exists():
        if not path.exists():
            raise RuntimeError("Existing KS4 bake-off directory lacks a manifest")
        existing = json.loads(path.read_text())
        if existing.get("request_digest") != manifest["request_digest"]:
            raise RuntimeError(
                "Existing KS4 bake-off receipt belongs to another request"
            )
        return existing
    native_output = Path(ks4_dir) / "sorter_output"
    times_path = native_output / "spike_times.npy"
    labels_path = native_output / "spike_clusters.npy"
    full_window = (
        window.start_frame == 0
        and window.end_frame
        == recording_manifest["selected_end_frame"]
        - recording_manifest["selected_start_frame"]
    )
    partial.mkdir(parents=True)
    if times_path.exists() and labels_path.exists():
        raw_times = np.asarray(np.load(times_path, mmap_mode="r")).reshape(-1)
        raw_labels = np.asarray(np.load(labels_path, mmap_mode="r")).reshape(-1)
        if raw_times.size != raw_labels.size:
            raise RuntimeError("KS4 spike_times and spike_clusters lengths differ")
        keep = (raw_times >= window.start_frame) & (raw_times < window.end_frame)
        times = np.asarray(raw_times[keep], dtype=np.int64) - window.start_frame
        labels = np.asarray(raw_labels[keep], dtype=np.int64)
        order = np.argsort(times, kind="stable")
        np.save(partial / "spike_times.npy", times[order])
        np.save(partial / "spike_labels.npy", labels[order])
        manifest["summary"] = {
            "final_spike_count": int(times.size),
            "unit_count": int(np.unique(labels).size),
            "normalized_arrays": ["spike_times.npy", "spike_labels.npy"],
        }
    elif full_window:
        # Backward compatibility for a legacy accepted receipt whose native
        # arrays have been archived separately.
        manifest["summary"] = sort_manifest["summary"]
        manifest["normalized_arrays_available"] = False
    else:
        raise RuntimeError("Cannot extract a KS4 window without native spike arrays")
    (partial / BAKEOFF_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, output_dir)
    return manifest


def run_dartsort_challenger(
    recording_dir: Path,
    output_dir: Path,
    *,
    preprocessing: str = "ibllikecmr",
    work_in_tmpdir: bool = True,
    window_name: str = "full_recording",
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Run native motion-aware DARTsort with an atomic experimental receipt."""
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spike-sort-challengers-numba-cache")
    recording_dir = Path(recording_dir)
    output_dir = Path(output_dir)
    partial = output_dir.with_name(output_dir.name + ".partial")
    manifest = _load_recording_manifest(recording_dir)
    window = resolve_bakeoff_window(
        manifest, name=window_name, start_s=start_s, duration_s=duration_s
    )
    _validate_recording_bytes(recording_dir, manifest)
    version = _package_version("dartsort")
    if version is None:
        raise RuntimeError(
            "DARTsort is not installed; use a dedicated compatible environment before running"
        )
    config_request = {
        "preprocessing": preprocessing,
        "do_motion_estimation": True,
        "work_in_tmpdir": bool(work_in_tmpdir),
        "copy_recording_to_tmpdir": True,
    }
    request = {
        "schema_version": BAKEOFF_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "candidate": "dartsort_native",
        "candidate_version": version,
        "recording_request_digest": manifest["request_digest"],
        "window": _window_dict(window),
        "config": config_request,
        "raw_voltage_warp": False,
    }
    request_digest = fingerprint(request)
    accepted_manifest = output_dir / BAKEOFF_MANIFEST
    if partial.exists():
        raise RuntimeError(f"Incomplete DARTsort run requires inspection: {partial}")
    if output_dir.exists():
        if not accepted_manifest.exists():
            raise RuntimeError("Existing DARTsort output lacks an accepted manifest")
        existing = json.loads(accepted_manifest.read_text())
        if existing.get("request_digest") != request_digest:
            raise RuntimeError("Existing DARTsort run belongs to another request")
        return existing

    import dartsort
    signature = inspect.signature(dartsort.DARTsortUserConfig)
    missing = set(config_request) - signature.parameters.keys()
    if missing:
        raise RuntimeError(
            f"DARTsort {version} config API lacks required controls: {sorted(missing)}"
        )
    recording = _load_si_extractor(recording_dir)
    expected_samples = manifest["selected_end_frame"] - manifest["selected_start_frame"]
    if recording.get_num_samples() != expected_samples:
        raise RuntimeError("Loaded recording length differs from its accepted manifest")
    recording = _slice_recording(recording, window)
    partial.mkdir(parents=True)
    native_output = partial / "native_output"
    cfg = dartsort.DARTsortUserConfig(**config_request)
    dartsort.dartsort(recording, native_output, cfg=cfg)
    summary = validate_dartsort_output(native_output)
    summary.update(normalize_dartsort_output(native_output, partial, window.frame_count))
    receipt = {
        **request,
        "request_digest": request_digest,
        "summary": summary,
        "experimental": True,
        "production_recommended": False,
        "upstream_status": "work_in_progress_not_recommended_for_production",
        "complete": True,
    }
    (partial / BAKEOFF_MANIFEST).write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(partial, output_dir)
    return receipt
