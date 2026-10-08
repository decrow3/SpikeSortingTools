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
KS4_PEELER_SCHEMA = "ks4-seeded-motion-aware-peeler-v2"


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
            name="kiasort_auto_curated",
            architecture="per_neuron_tracking_plus_native_gui_equivalent_curation",
            runner="kiasort_existing_result_auto_curation",
            maturity="experimental_challenger",
            motion_source="neuron_specific_tracking",
            raw_voltage_warp=False,
            pipeline_status="runnable_from_accepted_kiasort_output",
            requirement="Accepted KIASORT native output and pinned MATLAB checkout",
        ),
        SorterCandidate(
            name="ks4_seeded_static_peeler",
            architecture="accepted_ks4_seeds_with_static_template_matching",
            runner="paired_spikeinterface_tdc_peeler",
            maturity="experimental_control",
            motion_source="none",
            raw_voltage_warp=False,
            pipeline_status="runnable_with_newer_spikeinterface",
            requirement="SpikeInterface TridesclousPeeler and an accepted KS4 baseline",
        ),
        SorterCandidate(
            name="ks4_seeded_motion_native_peeler",
            architecture="accepted_ks4_seeds_with_motion_aware_template_matching",
            runner="paired_spikeinterface_tdc_peeler",
            maturity="experimental_challenger",
            motion_source="ks4_native_rigid_dshift",
            raw_voltage_warp=False,
            pipeline_status="runnable_with_ks4_rigid_motion_source",
            requirement="SpikeInterface motion-aware TridesclousPeeler and rigid KS4 ops.npy",
        ),
        SorterCandidate(
            name="ks4_seeded_motion_stabilized_peeler",
            architecture="accepted_ks4_seeds_with_jump_stabilized_motion_aware_matching",
            runner="paired_spikeinterface_tdc_peeler",
            maturity="experimental_challenger",
            motion_source="jump_stabilized_ks4_rigid_dshift",
            raw_voltage_warp=False,
            pipeline_status="runnable_with_ks4_rigid_motion_source",
            requirement="SpikeInterface motion-aware TridesclousPeeler and rigid KS4 ops.npy",
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
            ["git", "-C", str(root), "diff", "HEAD", "--binary", "--no-ext-diff"],
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
        "ks4_seeded_static_peeler",
        "ks4_seeded_motion_native_peeler",
        "ks4_seeded_motion_stabilized_peeler",
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
        elif name == "kiasort_auto_curated":
            row["environment_ready"] = bool(
                environment["kiasort"]["configured"]
                and environment["matlab_executable"]
            )
            row["runnable_now"] = False
            row["source_output_required"] = True
        elif name.startswith("ks4_seeded_"):
            row["component_available"] = bool(environment["tdc_motion_aware_available"])
            row["runnable_now"] = bool(environment["tdc_motion_aware_available"])
            if "motion_" in name:
                row["motion_source_required"] = True
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


KIASORT_CHANNEL_MAP_ADAPTER_VERSION = "column-vectors-v1"


def _adapt_kiasort_wrapper_channel_map(module) -> None:
    """Keep scipy ``savemat`` vectors compatible with KIASORT's MATLAB loader.

    KIASORT horizontally concatenates xcoords/ycoords.  SciPy otherwise writes
    one-dimensional arrays as MATLAB row vectors, producing a flattened 1x3N
    geometry instead of the required Nx3 channel-location matrix.
    """
    upstream_savemat = module.savemat

    def save_column_channel_map(path, values, *args, **kwargs):
        adapted = dict(values)
        for key in ("chanMap", "connected", "xcoords", "ycoords", "shankInd"):
            if key in adapted:
                adapted[key] = np.asarray(adapted[key]).reshape(-1, 1)
        return upstream_savemat(path, adapted, *args, **kwargs)

    module.savemat = save_column_channel_map


def _validate_kiasort_native_geometry(native_output: Path, expected_channels: int) -> dict[str, Any]:
    from scipy.io import loadmat

    path = Path(native_output) / "RES_Samples/channel_info.mat"
    try:
        channel_info = loadmat(
            path, variable_names=["channel_locations", "channel_mapping"]
        )
        locations = np.asarray(channel_info.get("channel_locations"), dtype=float)
        mapping = np.asarray(channel_info.get("channel_mapping")).reshape(-1)
        mat_format = "classic"
    except (NotImplementedError, ValueError):
        import h5py

        with h5py.File(path) as handle:
            # MATLAB v7.3 stores array dimensions in reverse HDF5 order.
            locations = np.asarray(handle["channel_locations"], dtype=float).T
            mapping = np.asarray(handle["channel_mapping"]).reshape(-1)
        mat_format = "v7.3_hdf5"
    if locations.ndim != 2 or locations.shape[0] != expected_channels or locations.shape[1] < 2:
        raise RuntimeError(
            "KIASORT native geometry is malformed: expected "
            f"{expected_channels}x>=2 channel_locations, got {locations.shape}. "
            "Do not use this run for drift-aware comparison."
        )
    if mapping.size != expected_channels:
        raise RuntimeError(
            f"KIASORT channel mapping has {mapping.size} entries; expected {expected_channels}"
        )
    return {
        "channel_locations_shape": list(locations.shape),
        "channel_mapping_count": int(mapping.size),
        "channel_info_mat_format": mat_format,
    }


def _recover_kiasort_postcurate_rollback(native_output: Path) -> dict[str, Any]:
    """Finish KIASORT's rollback only when its paired backup is provably intact."""
    import h5py

    native_output = Path(native_output)
    results = native_output / "RES_Sorted"
    backup = native_output / "Backup/postcurate"
    labels = results / "unifiedLabels.h5"
    paths = {
        "current_spikes": results / "spike_idx.h5",
        "backup_spikes": backup / "spike_idx.h5",
        "backup_labels": backup / "unifiedLabels.h5",
        "current_samples": native_output / "Sorted_Samples/sorted_samples.mat",
        "backup_samples": backup / "sorted_samples.mat",
        "log": native_output / "KIASort_log.txt",
    }
    if labels.exists():
        rollback_copies = [
            paths["current_spikes"],
            paths["backup_spikes"],
            paths["backup_labels"],
            paths["current_samples"],
            paths["backup_samples"],
        ]
        existing_rollback_copies = [path for path in rollback_copies if path.exists()]
        if not existing_rollback_copies:
            return {"needed": False, "recovered": False}
        if len(existing_rollback_copies) != len(rollback_copies):
            return {
                "needed": True,
                "recovered": False,
                "reason": "rollback inputs missing",
            }
        if _sha256_file(paths["current_spikes"]) != _sha256_file(paths["backup_spikes"]):
            return {"needed": True, "recovered": False, "reason": "spike backup differs"}
        if _sha256_file(paths["current_samples"]) != _sha256_file(paths["backup_samples"]):
            return {"needed": True, "recovered": False, "reason": "sample backup differs"}
        if _sha256_file(labels) == _sha256_file(paths["backup_labels"]):
            return {
                "needed": True,
                "recovered": True,
                "already_restored": True,
                "source": "Backup/postcurate/unifiedLabels.h5",
            }
        with h5py.File(paths["current_spikes"]) as handle:
            spike_count = int(np.asarray(handle["/spike_idx"]).size)
        with h5py.File(paths["backup_labels"]) as handle:
            label_count = int(np.asarray(handle["/unifiedLabels"]).size)
        if spike_count != label_count:
            return {"needed": True, "recovered": False, "reason": "label length differs"}
        shutil.copy2(paths["backup_labels"], labels)
        return {
            "needed": True,
            "recovered": True,
            "source": "Backup/postcurate/unifiedLabels.h5",
            "event_count": spike_count,
        }
    if any(not path.exists() for path in paths.values()):
        return {"needed": True, "recovered": False, "reason": "rollback inputs missing"}
    if "Post-hoc processing finished" not in paths["log"].read_text(errors="replace"):
        return {"needed": True, "recovered": False, "reason": "post-hoc did not finish"}
    if _sha256_file(paths["current_spikes"]) != _sha256_file(paths["backup_spikes"]):
        return {"needed": True, "recovered": False, "reason": "spike backup differs"}
    if _sha256_file(paths["current_samples"]) != _sha256_file(paths["backup_samples"]):
        return {"needed": True, "recovered": False, "reason": "sample backup differs"}
    with h5py.File(paths["current_spikes"]) as handle:
        spike_count = int(np.asarray(handle["/spike_idx"]).size)
    with h5py.File(paths["backup_labels"]) as handle:
        label_count = int(np.asarray(handle["/unifiedLabels"]).size)
    if spike_count != label_count:
        return {"needed": True, "recovered": False, "reason": "label length differs"}
    shutil.copy2(paths["backup_labels"], labels)
    return {
        "needed": True,
        "recovered": True,
        "source": "Backup/postcurate/unifiedLabels.h5",
        "event_count": spike_count,
    }


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
    constructor = getattr(si.NumpySorting, "from_samples_and_labels", None)
    if constructor is not None:
        return constructor(
            samples_list=spike_indices[assigned],
            labels_list=labels[assigned],
            sampling_frequency=float(sampling_frequency),
        )
    return si.NumpySorting.from_times_labels(
        times_list=spike_indices[assigned],
        labels_list=labels[assigned],
        sampling_frequency=float(sampling_frequency),
    )


def normalize_kiasort_curated_output(
    native_output: Path, output_dir: Path, num_samples: int
) -> dict[str, Any]:
    """Normalize KIASORT's non-destructive GUI-equivalent curated arrays."""
    import h5py
    from scipy.io import loadmat

    native_output = Path(native_output)
    result_dir = native_output / "RES_Sorted"
    names = {
        "times": ("spike_idx_curated.h5", "/spike_idx_curated"),
        "labels": ("unifiedLabels_curated.h5", "/unifiedLabels_curated"),
        "channels": ("channelNum_curated.h5", "/channelNum_curated"),
        "inclusion": ("inclusion_curated.h5", "/inclusion_curated"),
    }
    arrays = {}
    for name, (filename, dataset) in names.items():
        path = result_dir / filename
        if not path.exists():
            raise RuntimeError(f"KIASORT auto-curation did not create {path}")
        with h5py.File(path, "r") as values:
            arrays[name] = np.asarray(values[dataset]).reshape(-1)
    lengths = {values.size for values in arrays.values()}
    if len(lengths) != 1:
        raise RuntimeError("KIASORT curated arrays have different lengths")
    times = arrays["times"].astype(np.int64)
    labels = arrays["labels"].astype(np.int64)
    channels = arrays["channels"].astype(np.int64) - 1
    if not np.all(arrays["times"] == times) or not np.all(arrays["labels"] == labels):
        raise RuntimeError("KIASORT curated times or labels are non-integer")
    keep_assigned = labels >= 0
    keep_stable = arrays["inclusion"] > 0
    keep = keep_assigned & keep_stable
    times, labels, channels = times[keep], labels[keep], channels[keep]
    if np.any(times < 0) or np.any(times >= num_samples):
        raise RuntimeError("KIASORT curated times are outside the accepted window")
    channel_map = loadmat(native_output / "channel_map.mat")
    ycoords = np.asarray(channel_map["ycoords"]).reshape(-1)
    if np.any(channels < 0) or np.any(channels >= ycoords.size):
        raise RuntimeError("KIASORT curated channels are outside the selected band")
    order = np.argsort(times, kind="stable")
    times, labels, channels = times[order], labels[order], channels[order]
    output_dir = Path(output_dir)
    np.save(output_dir / "spike_times.npy", times)
    np.save(output_dir / "spike_labels.npy", labels)
    np.save(output_dir / "spike_channels.npy", channels)
    np.save(output_dir / "spike_depths_um.npy", ycoords[channels])
    return {
        "curated_assigned_spike_count_before_stable_interval": int(keep_assigned.sum()),
        "stable_interval_excluded_spike_count": int((keep_assigned & ~keep_stable).sum()),
        "final_spike_count": int(times.size),
        "assigned_spike_count": int(times.size),
        "unit_count": int(np.unique(labels).size),
        "normalized_arrays": [
            "spike_times.npy",
            "spike_labels.npy",
            "spike_channels.npy",
            "spike_depths_um.npy",
        ],
    }


def run_kiasort_auto_curation(
    source_output_dir: Path,
    output_dir: Path,
    *,
    kiasort_path: Path | None = None,
    matlab_bin: str = "matlab",
) -> dict[str, Any]:
    """Replay only KIASORT's non-destructive GUI-equivalent auto-curation."""
    source_output_dir = Path(source_output_dir)
    output_dir = Path(output_dir)
    partial = output_dir.with_name(output_dir.name + ".partial")
    source_manifest_path = source_output_dir / BAKEOFF_MANIFEST
    if not source_manifest_path.exists():
        raise RuntimeError("KIASORT auto-curation source lacks an accepted manifest")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("candidate") != "kiasort" or not source_manifest.get("complete"):
        raise RuntimeError("KIASORT auto-curation source is not an accepted native run")
    configured_path = kiasort_path or os.environ.get("KIASORT_PATH")
    installation = resolve_kiasort_installation(configured_path)
    if not installation["configured"]:
        raise RuntimeError(f"KIASORT is not configured: {installation['reason']}")
    auto_curate_path = Path(installation["root"]) / "No_GUI/kiaSort_auto_curate_nogui.m"
    if not auto_curate_path.exists():
        raise RuntimeError(f"KIASORT auto-curation entrypoint is missing: {auto_curate_path}")
    matlab_executable = shutil.which(matlab_bin)
    if matlab_executable is None:
        raise RuntimeError(f"MATLAB executable is unavailable: {matlab_bin}")
    source_native = source_output_dir / "native_output"
    source_geometry = _validate_kiasort_native_geometry(
        source_native, int(source_manifest["channel_selection"]["count"])
    )
    native_h5_names = (
        "amplitude.h5",
        "channelNum.h5",
        "features.h5",
        "labels.h5",
        "spike_idx.h5",
        "unifiedLabels.h5",
        "upadatedLabels.h5",
    )
    required = [
        source_native / "RES_Samples/channel_info.mat",
        source_native / "Sorted_Samples/sorted_samples.mat",
        source_native / "channel_map.mat",
    ] + [source_native / "RES_Sorted" / name for name in native_h5_names]
    if any(not path.exists() for path in required):
        raise RuntimeError("KIASORT auto-curation source is missing native inputs")
    source_hashes = {str(path.relative_to(source_native)): _sha256_file(path) for path in required}
    request = {
        "schema_version": BAKEOFF_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "candidate": "kiasort_auto_curated",
        "source_candidate_request_digest": source_manifest["request_digest"],
        "recording_request_digest": source_manifest["recording_request_digest"],
        "window": source_manifest["window"],
        "channel_selection": source_manifest["channel_selection"],
        "source_native_hashes": source_hashes,
        "source_geometry": source_geometry,
        "kiasort_git_commit": installation["git_commit"],
        "kiasort_tracked_diff_sha256": installation["tracked_diff_sha256"],
        "auto_curate_entrypoint_sha256": _sha256_file(auto_curate_path),
        "auto_curate_config": {
            "extractWaveforms": False,
            "defaults": "kiaSort_main_configs+kiaSort_extended_configs+kiaSort_hidden_configs",
            "source_config_overrides": source_manifest.get("config_overrides", {}),
        },
        "raw_voltage_warp": False,
    }
    request_digest = fingerprint(request)
    accepted_manifest = output_dir / BAKEOFF_MANIFEST
    if output_dir.exists():
        if not accepted_manifest.exists():
            raise RuntimeError("Existing KIASORT auto-curation output lacks a manifest")
        existing = json.loads(accepted_manifest.read_text())
        if existing.get("request_digest") != request_digest:
            raise RuntimeError("Existing KIASORT auto-curation belongs to another request")
        return existing
    if partial.exists():
        raise RuntimeError(f"Incomplete KIASORT auto-curation requires inspection: {partial}")
    replay_native = partial / "native_output"
    for source in required:
        destination = replay_native / source.relative_to(source_native)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    canonical = [
        replay_native / path.relative_to(source_native)
        for path in required
        if path.parent.name == "RES_Sorted"
    ] + [
        replay_native / "RES_Samples/channel_info.mat",
        replay_native / "Sorted_Samples/sorted_samples.mat",
    ]
    before_hashes = {
        str(path.relative_to(replay_native)): _sha256_file(path) for path in canonical
    }
    overrides = source_manifest.get("config_overrides", {})
    assignments = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            literal = "true" if value else "false"
        elif isinstance(value, (int, float)):
            literal = repr(value)
        elif isinstance(value, str):
            literal = "'" + value.replace("'", "''") + "'"
        else:
            raise RuntimeError(f"Unsupported KIASORT auto-curation override: {key}")
        assignments.append(f"cfg.{key} = {literal};")
    assignments.extend(
        [
            f"cfg.samplingFrequency = {source_manifest['window']['sampling_frequency_hz']!r};",
            f"cfg.numChannels = {source_manifest['channel_selection']['count']};",
        ]
    )
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    report_path = replay_native / "auto_curate_report.json"
    script = " ".join(
        [
            f"addpath(genpath({quote(installation['root'])}));",
            "cfg = kiaSort_main_configs();",
            "cfg = kiaSort_extended_configs(cfg);",
            "cfg = kiaSort_hidden_configs(cfg);",
            *assignments,
            f"cfg.outputFolder = {quote(replay_native)};",
            f"report = kiaSort_auto_curate_nogui({quote(replay_native)}, cfg, 'extractWaveforms', false, 'verbose', true);",
            f"fid = fopen({quote(report_path)}, 'w');",
            "fprintf(fid, '%s', jsonencode(report));",
            "fclose(fid);",
        ]
    )
    process = subprocess.run(
        [matlab_executable, "-batch", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (replay_native / "auto_curate_matlab.log").write_text(process.stdout)
    if process.returncode:
        raise RuntimeError(
            f"KIASORT auto-curation failed with code {process.returncode}\n"
            + "\n".join(process.stdout.splitlines()[-80:])
        )
    after_hashes = {
        str(path.relative_to(replay_native)): _sha256_file(path) for path in canonical
    }
    if after_hashes != before_hashes:
        raise RuntimeError("KIASORT auto-curation modified canonical replay inputs")
    if not report_path.exists():
        raise RuntimeError("KIASORT auto-curation did not write its report")
    matlab_report = json.loads(report_path.read_text())
    if not matlab_report.get("ok"):
        raise RuntimeError("KIASORT auto-curation report is not successful")
    summary = normalize_kiasort_curated_output(
        replay_native, partial, int(source_manifest["window"]["frame_count"])
    )
    receipt = {
        **request,
        "request_digest": request_digest,
        "matlab_executable": matlab_executable,
        "matlab_report": matlab_report,
        "canonical_replay_inputs_unchanged": True,
        "summary": summary,
        "experimental": True,
        "complete": True,
    }
    (partial / BAKEOFF_MANIFEST).write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(partial, output_dir)
    return receipt


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
        "channel_map_adapter_version": KIASORT_CHANNEL_MAP_ADAPTER_VERSION,
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
    _adapt_kiasort_wrapper_channel_map(wrapper)
    partial.mkdir(parents=True, exist_ok=recovering_partial)
    partial_request_path.write_text(
        json.dumps({"request_digest": request_digest, "request": request}, indent=2)
        + "\n"
    )
    native_output = partial / "native_output"
    reused_native_results = False
    postcurate_rollback = {"needed": False, "recovered": False}
    if recovering_partial:
        postcurate_rollback = _recover_kiasort_postcurate_rollback(native_output)
        if postcurate_rollback["needed"] and not postcurate_rollback["recovered"]:
            raise RuntimeError(
                "KIASORT post-curation rollback is incomplete: "
                + postcurate_rollback["reason"]
            )
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
    geometry = _validate_kiasort_native_geometry(
        native_output, selected_channel_count
    )
    summary = _normalize_sorting(sorting, partial, num_samples)
    receipt = {
        **request,
        "request_digest": request_digest,
        "kiasort_root": installation["root"],
        "matlab_executable": matlab_executable,
        "summary": summary,
        "native_geometry": geometry,
        "postcurate_rollback": postcurate_rollback,
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


def stabilize_ks4_rigid_motion(
    displacement_um: np.ndarray,
    *,
    max_step_um: float = 20.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unwrap discontinuous KS4 registration steps using a frozen step gate.

    A rejected increment contributes zero displacement instead of moving the
    trajectory onto a new coarse-registration branch. The operation is causal,
    deterministic, and independent of any sorting endpoint.
    """
    values = np.asarray(displacement_um, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("KS4 rigid motion must contain at least two finite bins")
    if not np.isfinite(max_step_um) or max_step_um <= 0:
        raise ValueError("max_step_um must be finite and positive")
    increments = np.diff(values)
    rejected = np.abs(increments) > float(max_step_um)
    accepted_increments = increments.copy()
    accepted_increments[rejected] = 0.0
    stabilized = np.r_[values[0], values[0] + np.cumsum(accepted_increments)]
    report = {
        "method": "unwrap_reject_large_batch_steps",
        "max_step_um": float(max_step_um),
        "rejected_step_count": int(np.sum(rejected)),
        "rejected_step_indices": (np.flatnonzero(rejected) + 1).astype(int).tolist(),
        "maximum_native_step_um": float(np.max(np.abs(increments))),
        "maximum_stabilized_step_um": float(np.max(np.abs(np.diff(stabilized)))),
    }
    return stabilized, report


def load_ks4_rigid_motion(
    ops_path: Path,
    *,
    window: BakeoffWindow,
    time_reference: str = "window_start",
    max_step_um: float = 20.0,
) -> dict[str, Any]:
    """Load, align, center, and stabilize a rigid KS4 ``dshift`` trajectory."""
    ops_path = Path(ops_path)
    if not ops_path.exists():
        raise FileNotFoundError(f"Missing KS4 motion source: {ops_path}")
    ops = np.load(ops_path, allow_pickle=True).item()
    dshift = ops.get("dshift")
    if dshift is None:
        raise ValueError("KS4 motion source has no dshift; use a rigid correction run")
    dshift = np.asarray(dshift, dtype=np.float64)
    if dshift.ndim == 1:
        dshift = dshift[:, None]
    if dshift.ndim != 2 or dshift.shape[1] != 1:
        raise ValueError(
            f"First benchmark requires rigid KS4 dshift with shape (time, 1), got {dshift.shape}"
        )
    if not np.all(np.isfinite(dshift)):
        raise ValueError("KS4 dshift contains non-finite values")
    fs = float(ops.get("fs", window.sampling_frequency_hz))
    batch_size = int(ops.get("batch_size", 0))
    if not np.isfinite(fs) or fs <= 0 or batch_size <= 0:
        raise ValueError("KS4 ops must contain positive fs and batch_size")
    if not np.isclose(fs, window.sampling_frequency_hz, rtol=1e-5, atol=1e-3):
        raise ValueError("KS4 motion and accepted recording sampling frequencies differ")
    if time_reference not in {"window_start", "selected_recording_start"}:
        raise ValueError(
            "time_reference must be 'window_start' or 'selected_recording_start'"
        )
    centers_s = (np.arange(dshift.shape[0], dtype=np.float64) + 0.5) * batch_size / fs
    if time_reference == "selected_recording_start":
        centers_s = centers_s - window.start_s
    duration_s = window.duration_s
    overlap = (centers_s >= -batch_size / fs) & (
        centers_s <= duration_s + batch_size / fs
    )
    if np.sum(overlap) < 2:
        raise ValueError("KS4 motion source does not overlap the requested window")
    centers_s = centers_s[overlap]
    native = dshift[overlap, 0]
    stabilized, stabilization = stabilize_ks4_rigid_motion(
        native, max_step_um=max_step_um
    )
    native_center = float(np.median(native))
    stabilized_center = float(np.median(stabilized))
    # KS4 dshift is the spatial shift applied to voltage during correction.
    # SI Motion instead represents the observed displacement used to move a
    # template on the stationary voltage, so the conventions have opposite
    # signs. Center first because KS4's registration zero is arbitrary.
    native = -(native - native_center)
    stabilized = -(stabilized - stabilized_center)
    return {
        "time_s": centers_s,
        "native_um": native,
        "stabilized_um": stabilized,
        "source_sha256": _sha256_file(ops_path),
        "source_path": str(ops_path.resolve()),
        "source_time_reference": time_reference,
        "source_bin_size_s": batch_size / fs,
        "ks4_dshift_convention": "voltage_correction_shift_um",
        "spikeinterface_motion_transform": "negative_median_centered_ks4_dshift",
        "native_center_removed_um": native_center,
        "stabilized_center_removed_um": stabilized_center,
        "stabilization": stabilization,
    }


def _select_ks4_seed_spikes(
    ks4_dir: Path,
    window: BakeoffWindow,
    *,
    min_spikes_per_unit: int,
    max_spikes_per_unit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    native = Path(ks4_dir) / "sorter_output"
    times = np.asarray(np.load(native / "spike_times.npy", mmap_mode="r")).reshape(-1)
    labels = np.asarray(np.load(native / "spike_clusters.npy", mmap_mode="r")).reshape(-1)
    if times.size != labels.size:
        raise RuntimeError("KS4 spike_times and spike_clusters lengths differ")
    if min_spikes_per_unit < 1:
        raise ValueError("min_spikes_per_unit must be positive")
    if max_spikes_per_unit is not None and max_spikes_per_unit < min_spikes_per_unit:
        raise ValueError("max_spikes_per_unit must be at least min_spikes_per_unit")
    in_window = (times >= window.start_frame) & (times < window.end_frame)
    local_times = np.asarray(times[in_window], dtype=np.int64) - window.start_frame
    local_labels = np.asarray(labels[in_window], dtype=np.int64)
    unit_ids, counts = np.unique(local_labels, return_counts=True)
    unit_ids = unit_ids[counts >= min_spikes_per_unit]
    if unit_ids.size == 0:
        raise RuntimeError("No KS4 units satisfy the seed-spike minimum in this window")
    keep = np.isin(local_labels, unit_ids)
    local_times = local_times[keep]
    local_labels = local_labels[keep]
    if max_spikes_per_unit is not None:
        selected = []
        for unit_id in unit_ids:
            inds = np.flatnonzero(local_labels == unit_id)
            if inds.size > max_spikes_per_unit:
                positions = np.linspace(0, inds.size - 1, max_spikes_per_unit)
                inds = inds[np.round(positions).astype(np.int64)]
            selected.append(inds)
        selected = np.sort(np.concatenate(selected))
        local_times = local_times[selected]
        local_labels = local_labels[selected]
    order = np.argsort(local_times, kind="stable")
    local_times = local_times[order]
    local_labels = local_labels[order]
    report = {
        "window_ks4_spike_count": int(np.sum(in_window)),
        "training_spike_count": int(local_times.size),
        "training_unit_count": int(unit_ids.size),
        "min_spikes_per_unit": int(min_spikes_per_unit),
        "max_spikes_per_unit": (
            None if max_spikes_per_unit is None else int(max_spikes_per_unit)
        ),
        "selection": "all_eligible_or_evenly_spaced_cap_per_unit",
    }
    return local_times, local_labels, unit_ids, report


def _normalize_peeler_spikes(
    spikes: np.ndarray,
    unit_ids: np.ndarray,
    output_dir: Path,
    num_samples: int,
) -> dict[str, Any]:
    required = {"sample_index", "cluster_index"}
    if spikes.dtype.names is None or not required.issubset(spikes.dtype.names):
        raise RuntimeError("TDC peeler output lacks sample_index/cluster_index fields")
    times = np.asarray(spikes["sample_index"], dtype=np.int64)
    cluster_index = np.asarray(spikes["cluster_index"], dtype=np.int64)
    valid = (
        (times >= 0)
        & (times < num_samples)
        & (cluster_index >= 0)
        & (cluster_index < unit_ids.size)
    )
    times = times[valid]
    cluster_index = cluster_index[valid]
    labels = np.asarray(unit_ids[cluster_index], dtype=np.int64)
    order = np.argsort(times, kind="stable")
    times = times[order]
    labels = labels[order]
    np.save(output_dir / "spike_times.npy", times)
    np.save(output_dir / "spike_labels.npy", labels)
    if "amplitude" in spikes.dtype.names:
        np.save(
            output_dir / "spike_amplitudes.npy",
            np.asarray(spikes["amplitude"][valid], dtype=np.float32)[order],
        )
    if "channel_index" in spikes.dtype.names:
        np.save(
            output_dir / "spike_channels.npy",
            np.asarray(spikes["channel_index"][valid], dtype=np.int32)[order],
        )
    return {
        "native_spike_count": int(spikes.size),
        "normalized_spike_count": int(times.size),
        "invalid_or_out_of_bounds_spike_count": int(spikes.size - times.size),
        "unit_count": int(np.unique(labels).size),
        "seed_unit_count": int(unit_ids.size),
    }


def summarize_ks4_seeded_peeler_arm(
    reference_times: np.ndarray,
    reference_labels: np.ndarray,
    spike_times: np.ndarray,
    spike_labels: np.ndarray,
    *,
    sampling_frequency_hz: float,
    duration_s: float,
    event_tolerance_ms: float = 0.5,
    refractory_ms: float = 1.5,
    duplicate_ms: float = 0.2,
    presence_bin_s: float = 10.0,
) -> dict[str, Any]:
    """Compute label-preserving paired guardrails for one peeler arm."""
    reference_times = np.asarray(reference_times, dtype=np.int64).reshape(-1)
    reference_labels = np.asarray(reference_labels, dtype=np.int64).reshape(-1)
    spike_times = np.asarray(spike_times, dtype=np.int64).reshape(-1)
    spike_labels = np.asarray(spike_labels, dtype=np.int64).reshape(-1)
    if reference_times.size != reference_labels.size or spike_times.size != spike_labels.size:
        raise ValueError("Paired metric spike arrays have inconsistent lengths")
    if sampling_frequency_hz <= 0 or duration_s <= 0 or presence_bin_s <= 0:
        raise ValueError("Metric sampling frequency, duration, and bin size must be positive")
    event_tol = max(1, int(round(event_tolerance_ms * sampling_frequency_hz / 1000.0)))
    refractory = max(1, int(round(refractory_ms * sampling_frequency_hz / 1000.0)))
    duplicate = max(1, int(round(duplicate_ms * sampling_frequency_hz / 1000.0)))
    units = np.unique(reference_labels)
    recovered = 0
    unit_refractory = []
    unit_presence = []
    first_last = []
    num_bins = max(1, int(np.ceil(duration_s / presence_bin_s)))
    endpoint_frames = int(round(min(20.0, duration_s / 2.0) * sampling_frequency_hz))
    duration_frames = int(round(duration_s * sampling_frequency_hz))
    for unit_id in units:
        ref = reference_times[reference_labels == unit_id]
        arm = spike_times[spike_labels == unit_id]
        if arm.size:
            ref_index = 0
            arm_index = 0
            while ref_index < ref.size and arm_index < arm.size:
                delta = int(arm[arm_index]) - int(ref[ref_index])
                if delta < -event_tol:
                    arm_index += 1
                elif delta > event_tol:
                    ref_index += 1
                else:
                    recovered += 1
                    ref_index += 1
                    arm_index += 1
            if arm.size > 1:
                unit_refractory.append(float(np.mean(np.diff(arm) <= refractory)))
            else:
                unit_refractory.append(0.0)
            bins = np.minimum(
                (arm / (presence_bin_s * sampling_frequency_hz)).astype(np.int64),
                num_bins - 1,
            )
            unit_presence.append(float(np.unique(bins).size / num_bins))
            first_last.append(
                bool(np.any(arm < endpoint_frames) and np.any(arm >= duration_frames - endpoint_frames))
            )
        else:
            unit_refractory.append(0.0)
            unit_presence.append(0.0)
            first_last.append(False)
    order = np.argsort(spike_times, kind="stable")
    ordered_times = spike_times[order]
    ordered_labels = spike_labels[order]
    duplicate_pair_count = 0
    window_start = 0
    window_label_counts: dict[int, int] = {}
    for window_end in range(ordered_times.size):
        while ordered_times[window_end] - ordered_times[window_start] > duplicate:
            expired_label = int(ordered_labels[window_start])
            window_label_counts[expired_label] -= 1
            if window_label_counts[expired_label] == 0:
                del window_label_counts[expired_label]
            window_start += 1
        current_label = int(ordered_labels[window_end])
        same_label_count = window_label_counts.get(current_label, 0)
        duplicate_pair_count += window_end - window_start - same_label_count
        window_label_counts[current_label] = same_label_count + 1
    return {
        "reference_spike_count": int(reference_times.size),
        "reference_unit_count": int(units.size),
        "label_preserving_reference_event_recovery": (
            float(recovered / reference_times.size) if reference_times.size else 0.0
        ),
        "median_unit_refractory_fraction": (
            float(np.median(unit_refractory)) if unit_refractory else 0.0
        ),
        "median_unit_presence_fraction": (
            float(np.median(unit_presence)) if unit_presence else 0.0
        ),
        "units_present_in_first_and_last_20s_fraction": (
            float(np.mean(first_last)) if first_last else 0.0
        ),
        "cross_unit_near_coincident_pair_count": duplicate_pair_count,
        "cross_unit_near_coincident_pairs_per_spike": (
            float(duplicate_pair_count / spike_times.size) if spike_times.size else 0.0
        ),
        "event_tolerance_ms": float(event_tolerance_ms),
        "refractory_ms": float(refractory_ms),
        "duplicate_ms": float(duplicate_ms),
        "presence_bin_s": float(presence_bin_s),
    }


def run_ks4_seeded_peeler_pair(
    recording_dir: Path,
    ks4_dir: Path,
    motion_ops_path: Path,
    output_dir: Path,
    *,
    motion_time_reference: str = "window_start",
    motion_max_step_um: float = 20.0,
    highpass_hz: float = 300.0,
    template_radius_um: float = 100.0,
    min_spikes_per_unit: int = 20,
    max_spikes_per_unit: int | None = 2000,
    detect_threshold: float = 5.0,
    interpolation_time_bin_size_s: float = 1.0,
    motion_step_um: float = 5.0,
    n_jobs: int = 1,
    chunk_duration: str = "1s",
    window_name: str = "full_recording",
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Run one static and two motion-aware TDC arms from identical KS4 seeds."""
    recording_dir = Path(recording_dir)
    ks4_dir = Path(ks4_dir)
    motion_ops_path = Path(motion_ops_path)
    output_dir = Path(output_dir)
    partial = output_dir.with_name(output_dir.name + ".partial")
    recording_manifest = _load_recording_manifest(recording_dir)
    window = resolve_bakeoff_window(
        recording_manifest, name=window_name, start_s=start_s, duration_s=duration_s
    )
    _validate_recording_bytes(recording_dir, recording_manifest)
    sort_manifest_path = ks4_dir / SORT_MANIFEST
    if not sort_manifest_path.exists():
        raise FileNotFoundError(f"Missing accepted KS4 manifest: {sort_manifest_path}")
    sort_manifest = json.loads(sort_manifest_path.read_text())
    if not sort_manifest.get("complete"):
        raise RuntimeError("KS4 seed manifest is not marked complete")
    if sort_manifest.get("recording_request_digest") != recording_manifest["request_digest"]:
        raise RuntimeError("KS4 seed sort belongs to another recording")
    motion = load_ks4_rigid_motion(
        motion_ops_path,
        window=window,
        time_reference=motion_time_reference,
        max_step_um=motion_max_step_um,
    )
    config = {
        "frontend": {
            "highpass_hz": float(highpass_hz),
            "reference": "global",
            "reference_operator": "median",
            "return_in_uV": False,
        },
        "template_radius_um": float(template_radius_um),
        "min_spikes_per_unit": int(min_spikes_per_unit),
        "max_spikes_per_unit": max_spikes_per_unit,
        "detect_threshold": float(detect_threshold),
        "interpolation_time_bin_size_s": float(interpolation_time_bin_size_s),
        "motion_step_um": float(motion_step_um),
        "motion_stabilization": motion["stabilization"],
        "motion_transform": "negative_median_centered_ks4_dshift",
        "job": {"n_jobs": int(n_jobs), "chunk_duration": str(chunk_duration)},
    }
    if highpass_hz <= 0 or template_radius_um <= 0 or detect_threshold <= 0:
        raise ValueError("Frontend, template radius, and detection threshold must be positive")
    if interpolation_time_bin_size_s <= 0 or motion_step_um <= 0 or n_jobs < 1:
        raise ValueError("Motion interpolation, step, and n_jobs must be positive")
    if n_jobs != 1:
        raise ValueError(
            "The SI 0.104.8 motion-aware peeler adapter is intentionally serial; "
            "its disabled warm-up call is not safe for parallel Numba compilation"
        )
    request = {
        "schema_version": KS4_PEELER_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "candidate": "ks4_seeded_peeler_pair",
        "recording_request_digest": recording_manifest["request_digest"],
        "ks4_seed_request_digest": sort_manifest["request_digest"],
        "ks4_motion_source_sha256": motion["source_sha256"],
        "ks4_motion_time_reference": motion_time_reference,
        "window": _window_dict(window),
        "config": config,
        "raw_voltage_warp": False,
    }
    request_digest = fingerprint(request)
    accepted_manifest = output_dir / BAKEOFF_MANIFEST
    if partial.exists():
        raise RuntimeError(f"Incomplete KS4-seeded peeler pair requires inspection: {partial}")
    if output_dir.exists():
        if not accepted_manifest.exists():
            raise RuntimeError("Existing peeler-pair output lacks an accepted manifest")
        existing = json.loads(accepted_manifest.read_text())
        if existing.get("request_digest") != request_digest:
            raise RuntimeError("Existing peeler-pair output belongs to another request")
        return existing

    try:
        import spikeinterface
        from spikeinterface.core import NumpySorting, Templates, estimate_templates, get_noise_levels
        from spikeinterface.core.motion import Motion
        from spikeinterface.preprocessing import common_reference, highpass_filter
        from spikeinterface.sortingcomponents.matching import find_spikes_from_templates
        from spikeinterface.sortingcomponents.matching.tdc_peeler import TridesclousPeeler
    except ImportError as exc:
        raise RuntimeError("Use the isolated SpikeInterface 0.104 challenger environment") from exc
    required_api = {"motion_aware", "motion", "interpolation_time_bin_size_s", "motion_step_um"}
    missing_api = required_api - set(inspect.signature(TridesclousPeeler.__init__).parameters)
    if missing_api:
        raise RuntimeError(f"Installed TDC peeler lacks motion API: {sorted(missing_api)}")

    recording = _load_si_extractor(recording_dir)
    expected_samples = recording_manifest["selected_end_frame"] - recording_manifest["selected_start_frame"]
    if recording.get_num_samples() != expected_samples:
        raise RuntimeError("Loaded recording length differs from its accepted manifest")
    recording = _slice_recording(recording, window)
    recording.reset_times()
    recording = highpass_filter(recording, freq_min=highpass_hz, dtype="float32")
    recording = common_reference(
        recording, reference="global", operator="median", dtype="float32"
    )
    seed_times, seed_labels, unit_ids, seed_report = _select_ks4_seed_spikes(
        ks4_dir,
        window,
        min_spikes_per_unit=min_spikes_per_unit,
        max_spikes_per_unit=max_spikes_per_unit,
    )
    reference_times, reference_labels, reference_unit_ids, _ = _select_ks4_seed_spikes(
        ks4_dir,
        window,
        min_spikes_per_unit=min_spikes_per_unit,
        max_spikes_per_unit=None,
    )
    if not np.array_equal(reference_unit_ids, unit_ids):
        raise RuntimeError("Training cap unexpectedly changed the eligible KS4 unit set")
    sorting = NumpySorting.from_samples_and_labels(
        seed_times,
        seed_labels,
        window.sampling_frequency_hz,
        unit_ids=unit_ids,
    )
    spike_vector = sorting.to_spike_vector(concatenated=True)
    native_output = ks4_dir / "sorter_output"
    ops = np.load(native_output / "ops.npy", allow_pickle=True).item()
    nbefore = int(ops.get("nt0min", 20))
    template_samples = int(ops.get("nt", 61))
    nafter = template_samples - nbefore
    if nbefore < 1 or nafter < 1:
        raise ValueError("KS4 ops contain invalid nt/nt0min template alignment")
    job_kwargs = {"n_jobs": int(n_jobs), "chunk_duration": str(chunk_duration)}
    noise_levels = get_noise_levels(
        recording,
        return_in_uV=False,
        random_slices_kwargs={"seed": 1002},
        **job_kwargs,
    )
    dense_array = estimate_templates(
        recording,
        spike_vector,
        unit_ids,
        nbefore,
        nafter,
        return_in_uV=False,
        **job_kwargs,
    )
    if not np.all(np.isfinite(dense_array)):
        raise RuntimeError("Estimated KS4-seeded templates contain non-finite values")
    channel_locations = np.asarray(recording.get_channel_locations(), dtype=np.float64)
    main_channels = np.argmin(np.min(dense_array, axis=1), axis=1)
    distances = np.linalg.norm(
        channel_locations[None, :, :] - channel_locations[main_channels, None, :], axis=2
    )
    sparsity_mask = distances <= float(template_radius_um)
    dense_templates = Templates(
        templates_array=np.asarray(dense_array, dtype=np.float32),
        sampling_frequency=window.sampling_frequency_hz,
        nbefore=nbefore,
        channel_ids=recording.channel_ids,
        unit_ids=unit_ids,
        probe=recording.get_probe(),
        is_in_uV=False,
    )
    templates = dense_templates.to_sparse(sparsity_mask)

    partial.mkdir(parents=True)
    shared = partial / "shared_inputs"
    shared.mkdir()
    np.save(shared / "training_spike_times.npy", seed_times)
    np.save(shared / "training_spike_labels.npy", seed_labels)
    np.save(shared / "reference_spike_times.npy", reference_times)
    np.save(shared / "reference_spike_labels.npy", reference_labels)
    np.save(shared / "unit_ids.npy", unit_ids)
    np.save(shared / "templates.npy", templates.templates_array)
    np.save(shared / "template_sparsity_mask.npy", sparsity_mask)
    np.save(shared / "noise_levels.npy", noise_levels)
    np.savez(
        shared / "ks4_rigid_motion.npz",
        time_s=motion["time_s"],
        native_um=motion["native_um"],
        stabilized_um=motion["stabilized_um"],
    )
    spatial_center = np.array([float(np.median(channel_locations[:, 1]))])
    motion_objects = {
        "ks4_seeded_static_peeler": None,
        "ks4_seeded_motion_native_peeler": Motion(
            displacement=motion["native_um"][:, None],
            temporal_bins_s=motion["time_s"],
            spatial_bins_um=spatial_center,
            direction="y",
            interpolation_method="linear",
        ),
        "ks4_seeded_motion_stabilized_peeler": Motion(
            displacement=motion["stabilized_um"][:, None],
            temporal_bins_s=motion["time_s"],
            spatial_bins_um=spatial_center,
            direction="y",
            interpolation_method="linear",
        ),
    }
    summaries = {}
    paired_metrics = {}
    method_base = {
        "noise_levels": noise_levels,
        "detect_threshold": float(detect_threshold),
    }
    for arm, motion_object in motion_objects.items():
        arm_dir = partial / arm
        arm_dir.mkdir()
        method_kwargs = dict(method_base)
        method_kwargs["motion_aware"] = motion_object is not None
        if motion_object is not None:
            method_kwargs.update(
                motion=motion_object,
                interpolation_time_bin_size_s=float(interpolation_time_bin_size_s),
                motion_step_um=float(motion_step_um),
            )
        # SI 0.104.8's generic warm-up passes a margin-padded trace while the
        # motion branch asserts against the unpadded time span. In serial mode
        # the warm-up is unnecessary, so this version-specific adapter disables
        # it without changing matching behavior.
        original_warmup = TridesclousPeeler.need_first_call_before_pipeline
        TridesclousPeeler.need_first_call_before_pipeline = False
        try:
            spikes = find_spikes_from_templates(
                recording,
                templates,
                method="tdc-peeler",
                method_kwargs=method_kwargs,
                pipeline_kwargs={"gather_mode": "memory"},
                job_kwargs=job_kwargs,
                verbose=True,
            )
        finally:
            TridesclousPeeler.need_first_call_before_pipeline = original_warmup
        summary = _normalize_peeler_spikes(
            spikes, unit_ids, arm_dir, window.frame_count
        )
        summaries[arm] = summary
        paired_metrics[arm] = summarize_ks4_seeded_peeler_arm(
            reference_times,
            reference_labels,
            np.load(arm_dir / "spike_times.npy", mmap_mode="r"),
            np.load(arm_dir / "spike_labels.npy", mmap_mode="r"),
            sampling_frequency_hz=window.sampling_frequency_hz,
            duration_s=window.duration_s,
        )
        arm_receipt = {
            "schema_version": KS4_PEELER_SCHEMA,
            "parent_request_digest": request_digest,
            "candidate": arm,
            "motion_aware": motion_object is not None,
            "motion_variant": (
                "none"
                if motion_object is None
                else ("native" if "native" in arm else "stabilized")
            ),
            "raw_voltage_warp": False,
            "summary": summary,
            "paired_guardrails": paired_metrics[arm],
            "complete": True,
        }
        (arm_dir / BAKEOFF_MANIFEST).write_text(json.dumps(arm_receipt, indent=2) + "\n")
    receipt = {
        **request,
        "request_digest": request_digest,
        "spikeinterface_version": spikeinterface.__version__,
        "ks4_motion_source": {
            key: value
            for key, value in motion.items()
            if key not in {"time_s", "native_um", "stabilized_um"}
        },
        "seed_summary": seed_report,
        "template_summary": {
            "template_count": int(unit_ids.size),
            "template_samples": int(template_samples),
            "nbefore": int(nbefore),
            "maximum_active_channels": int(np.max(np.sum(sparsity_mask, axis=1))),
            "shared_across_all_arms": True,
        },
        "spikeinterface_adapter": {
            "serial_only": True,
            "tdc_first_call_warmup_disabled": True,
            "reason": "SI 0.104.8 motion warm-up mixes padded traces with unpadded times",
        },
        "arms": summaries,
        "paired_guardrails": paired_metrics,
        "interpretation_guardrail": (
            "Counts do not establish benefit. Motion-aware arms must preserve KS4 event "
            "recovery and refractory/near-coincident guardrails before continuity gains "
            "are interpreted; waveform residual and reviewed-event audits remain required."
        ),
        "experimental": True,
        "complete": True,
    }
    (partial / BAKEOFF_MANIFEST).write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(partial, output_dir)
    return receipt


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
