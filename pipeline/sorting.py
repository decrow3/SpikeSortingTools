"""Guarded Kilosort 4 execution for the rescue recording."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import PIPELINE_VERSION, fingerprint
from .preprocess import MANIFEST_NAME, validate_accepted_recording


SORT_MANIFEST = "rescue_sort_manifest.json"


def rescue_kilosort4_overrides() -> dict[str, Any]:
    """Return the frozen settings selected by the rescue experiments.

    This function is deliberately independent of SpikeInterface so ``--plan``
    can report the tested choices without importing a sorter environment.
    """
    return {
        "do_correction": False,
        "do_CAR": True,
        "artifact_threshold": math.inf,
        "save_extra_vars": True,
        "bad_channels": None,
        "Th_universal": 12,
        "Th_learned": 9,
        "duplicate_spike_ms": 0.25,
        "ccg_threshold": 0.75,
        "nearest_chans": 20,
        "nearest_templates": 200,
        "max_channel_distance": 64,
        "clear_cache": True,
        "cross_peel_claim_ms": 0.0,
        "cross_peel_claim_um": 0.0,
    }


def build_kilosort4_params(defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return explicit, tested sorter settings with all rejected features off."""
    if defaults is None:
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spikeglx-rescue-numba-cache")
        from spikeinterface.sorters import get_default_sorter_params

        defaults = get_default_sorter_params("kilosort4")
    params = dict(defaults)
    required = {
        "do_correction",
        "do_CAR",
        "artifact_threshold",
        "save_extra_vars",
        "Th_universal",
        "Th_learned",
        "duplicate_spike_ms",
        "ccg_threshold",
        "nearest_chans",
        "nearest_templates",
        "max_channel_distance",
        "clear_cache",
    }
    missing = required - params.keys()
    if missing:
        raise RuntimeError(f"Kilosort 4 settings are missing: {sorted(missing)}")
    overrides = rescue_kilosort4_overrides()
    # Patched environments expose the claim-mask keys. In an unpatched
    # environment the feature does not exist and is therefore already off.
    for name in ("cross_peel_claim_ms", "cross_peel_claim_um"):
        if name not in params:
            overrides.pop(name)
    params.update(overrides)
    return params


def _json_safe_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: ("Infinity" if isinstance(value, float) and math.isinf(value) else value)
        for key, value in params.items()
    }


def validate_applied_settings(
    applied: Mapping[str, Any], requested: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse a completed sort whose saved critical settings differ."""
    expected = {
        "do_correction": False,
        "do_CAR": True,
        "artifact_threshold": math.inf,
        "cross_peel_claim_ms": 0.0,
        "cross_peel_claim_um": 0.0,
    }
    validated = {}
    for key, value in expected.items():
        if key not in requested:
            continue
        if key in applied:
            observed = applied[key]
        elif key == "do_correction" and "nblocks" in applied:
            # ``do_correction`` is a SpikeInterface wrapper option.  Kilosort
            # does not retain it in ops.npy; its effective native state is
            # nblocks=0 when correction is disabled.
            observed = int(applied["nblocks"]) != 0
        else:
            raise RuntimeError(f"Saved Kilosort ops omit critical setting {key}")
        if math.isinf(value):
            try:
                matches = bool(np.isinf(observed))
            except TypeError:
                matches = False
        else:
            matches = observed == value
        if not matches:
            raise RuntimeError(f"Saved Kilosort setting {key}={observed!r}, expected {value!r}")
        validated[key] = "Infinity" if math.isinf(value) else value
        if key == "do_correction" and "do_correction" not in applied:
            validated["effective_nblocks"] = int(applied["nblocks"])
    return validated


def _sort_summary(sorter_output: Path, params: Mapping[str, Any]) -> dict[str, Any]:
    ops_path = sorter_output / "ops.npy"
    if not ops_path.exists():
        raise RuntimeError(f"Kilosort ended without {ops_path}")
    ops = np.load(ops_path, allow_pickle=True).item()
    # The nested settings are Kilosort's input settings.  Top-level ops values
    # are the effective settings after SpikeInterface wrapper translations
    # such as do_correction=False -> nblocks=0, so they take precedence.
    applied = dict(ops.get("settings", {}))
    applied.update(ops)
    validated = validate_applied_settings(applied, params)
    clusters = np.load(sorter_output / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    label_path = sorter_output / "cluster_KSLabel.tsv"
    good_units = None
    if label_path.exists():
        with label_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        good_units = sum(
            any(str(value).strip().lower() == "good" for value in row.values())
            for row in rows
        )
    return {
        "critical_saved_settings": validated,
        "final_spike_count": int(clusters.size),
        "unit_count": int(np.unique(clusters).size),
        "kilosort_good_unit_count": good_units,
    }


def _validate_saved_sorter_params(partial: Path, params: Mapping[str, Any]) -> None:
    """Require an exported partial sort to match the complete frozen request."""
    params_path = partial / "spikeinterface_params.json"
    if not params_path.exists():
        raise RuntimeError(f"Incomplete sort requires inspection: {partial}")
    saved = json.loads(params_path.read_text())
    if saved.get("sorter_name") != "kilosort4":
        raise RuntimeError(f"Partial sort is not a Kilosort 4 run: {partial}")
    saved_params = saved.get("sorter_params")
    if not isinstance(saved_params, dict):
        raise RuntimeError(f"Partial sort has no saved sorter parameters: {partial}")
    if _json_safe_params(saved_params) != _json_safe_params(params):
        raise RuntimeError("Completed partial sort belongs to another sorter configuration")


def run_kilosort4(recording_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run Kilosort into a partial directory and atomically accept completion."""
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spikeglx-rescue-numba-cache")
    from spikeinterface.core import load_extractor
    from spikeinterface.sorters import run_sorter

    recording_dir = Path(recording_dir)
    output_dir = Path(output_dir)
    partial = output_dir.with_name(output_dir.name + ".partial")
    recording_manifest_path = recording_dir / MANIFEST_NAME
    if not recording_manifest_path.exists():
        raise FileNotFoundError(f"Missing accepted recording manifest: {recording_manifest_path}")
    recording_manifest = json.loads(recording_manifest_path.read_text())
    validate_accepted_recording(recording_dir, recording_manifest)
    params = build_kilosort4_params()
    safe_params = _json_safe_params(params)
    request = {
        "pipeline_version": PIPELINE_VERSION,
        "recording_request_digest": recording_manifest["request_digest"],
        "sorter": "kilosort4",
        "sorter_params": safe_params,
    }
    request_digest = fingerprint(request)
    manifest_path = output_dir / SORT_MANIFEST
    if partial.exists():
        sorter_output = partial / "sorter_output"
        if not (sorter_output / "spike_times.npy").exists():
            raise RuntimeError(f"Incomplete sort requires inspection: {partial}")
        _validate_saved_sorter_params(partial, params)
        summary = _sort_summary(sorter_output, params)
        manifest = {
            **request,
            "request_digest": request_digest,
            "summary": summary,
            "complete": True,
            "recovered_completed_partial": True,
        }
        (partial / SORT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(partial, output_dir)
        return manifest
    if output_dir.exists():
        if not manifest_path.exists():
            raise RuntimeError(f"Existing sort lacks {SORT_MANIFEST}: {output_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request_digest") != request_digest:
            raise RuntimeError("Existing sort belongs to another recording/configuration")
        return manifest
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    recording = load_extractor(recording_dir)
    selected_samples = (
        recording_manifest["selected_end_frame"]
        - recording_manifest["selected_start_frame"]
    )
    if recording.get_num_samples() != selected_samples:
        raise RuntimeError("Loaded recording length differs from its accepted manifest")
    run_sorter(
        "kilosort4",
        recording,
        folder=str(partial),
        verbose=True,
        remove_existing_folder=False,
        **params,
    )
    sorter_output = partial / "sorter_output"
    spike_times = sorter_output / "spike_times.npy"
    if not spike_times.exists():
        raise RuntimeError(f"Kilosort ended without {spike_times}")
    summary = _sort_summary(sorter_output, params)
    manifest = {
        **request,
        "request_digest": request_digest,
        "summary": summary,
        "complete": True,
    }
    (partial / SORT_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, output_dir)
    return manifest
