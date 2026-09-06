"""Thin candidate runner for the first-version delivery contract.

Build instructions: ``docs/luke_two_motion_pipeline_build_instructions.md`` §3.
Contract: ``configs/first_pipeline_candidate.v1.json``. Governing plan:
``docs/pipeline_improvement_plan.md``.

What this runs is a **retained-sort replay**, which is what the contract now
declares. The accepted rescue sort's spike rows are read back, restricted to
one declared interval, regrouped into identity families, exported, and scored
on the endpoint that selected the case -- estimated missing-spike percentage
from truncated amplitude fits. No sorting happens, and no production output is
written to.

The contract, not the command line, decides what runs. Every gate constant,
interval, input identity and motion declaration comes out of
``candidate.settings.value.resolved_configuration``; the CLI flags the build
instructions require still exist, but a value that disagrees with the contract
is **refused** rather than applied. A run that could be steered by a flag is not
a run against a frozen contract.

Two arms in each sense of the word:

* ``--option control`` replays the retained labels unchanged and produces the
  same export and the same completeness table as the candidate, so "baseline"
  is a measured comparator rather than an absence.
* ``--arm case | healthy_control`` selects which declared interval is processed:
  the nominated failure, or the reserved healthy interval on which preservation
  is measured.

CLI (the build instructions' required surface, plus ``--arm``)::

    --option control|external_warp|unwarped_identity
    --arm case|healthy_control
    --mode verify|smoke|l1|l2|l2l
    --out-root PATH
    --snippet-dir PATH        # must agree with the contract's declared input
    --motion-info-dir PATH    # only for a contract-declared motion-aware arm
    --truth PATH              # refused for a retained-sort replay
    --config PATH             # must digest-match the contract's own settings
    --contract PATH
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.motion_coordinates import (
    interpolate_motion_at_spikes,
    load_qualified_motion_field,
)
from testing.first_pipeline_candidate_contract import (
    DEFAULT_CONTRACT,
    ContractRefusal,
    canonical_digest,
    git_worktree_state,
    load_contract,
    reject_unsafe_out_root,
    validate,
)
from testing.ladder_unwarped_identity import (
    MotionDeclaration,
    ReplayInput,
    UnwarpedIdentityConfig,
    refractory_violation_fraction,
    run_unwarped_identity_replay,
)
from testing.luke_amplitude_dropout_audit import load_curated_arrays
from testing.luke_candidate_completeness_qc import (
    CompletenessConfig,
    VERDICT_PASS,
    VERDICT_UNEVALUABLE,
    compare_completeness,
    family_amplitude_scale_check,
    train_completeness,
)

BAKEOFF_SCHEMA = "luke-two-motion-pipeline-bakeoff-v2"
BAKEOFF_MANIFEST = "bakeoff_manifest.json"

OPTIONS = ("control", "external_warp", "unwarped_identity")
ARMS = ("case", "healthy_control")
MODES = ("verify", "smoke", "l1", "l2", "l2l")
EXECUTION_MODES = ("smoke", "l1", "l2", "l2l")

#: Arrays this replay reads that ``load_curated_arrays`` does not already hash.
EXTRA_HASHED_FILES = (
    "spike_positions.npy",
    "spike_templates.npy",
    "templates.npy",
    "whitening_mat_inv.npy",
    "channel_positions.npy",
)

#: Label given to a family assembled from more than one original cluster. It is
#: not `good`: nothing downstream has validated it yet.
UNVALIDATED_LABEL = "unvalidated"


class RunnerRefusal(ValueError):
    """The runner refuses to proceed. Never caught internally."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_hashed_array(path: Path) -> tuple[np.ndarray, str]:
    """Hash and parse the very same bytes, so the digest attests what ran."""
    data = Path(path).read_bytes()
    return np.load(io.BytesIO(data), allow_pickle=False), _sha256_bytes(data)


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, stop in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], stop))
        else:
            out.append((start, stop))
    return out


def _covered_by(interval: tuple[float, float], allowed: list[tuple[float, float]]) -> bool:
    start, stop = interval
    for lo, hi in _merge(allowed):
        if lo <= start and stop <= hi:
            return True
    return False


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _pairs(node: Any) -> list[tuple[float, float]]:
    out = []
    for item in node:
        if isinstance(item, dict):
            out.append((float(item["start_s"]), float(item["stop_s"])))
        else:
            out.append((float(item[0]), float(item[1])))
    return out


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# resolving the contract's own settings
# --------------------------------------------------------------------------- #
def _refuse_conflicting_override(name: str, supplied: Any, declared: Any) -> None:
    raise RunnerRefusal(
        f"refusing {name}={supplied!r}: the contract declares {declared!r}. The contract's "
        "resolved settings and declared inputs are what execute; a command line that disagrees "
        "with them is refused rather than applied."
    )


def resolve_settings(
    contract: dict[str, Any],
    *,
    option: str,
    arm: str,
    snippet_dir: Path | None,
    motion_info_dir: Path | None,
    truth_path: Path | None,
    config_path: Path | None,
) -> dict[str, Any]:
    """Turn the contract into the settings that will run, refusing conflicts."""
    settings = contract["candidate"]["settings"]
    if settings.get("state") != "set":
        raise RunnerRefusal("candidate.settings is unset; there is nothing to execute")
    value = settings["value"]
    resolved = value["resolved_configuration"]

    execution_mode = value["execution_mode"]
    if execution_mode != "retained_sort_replay":
        raise RunnerRefusal(
            f"this runner implements retained_sort_replay; the contract declares "
            f"{execution_mode!r}. Correct the contract or the runner -- not the description."
        )
    if option == "unwarped_identity" and value["intervention_family"] != "option_b_unwarped_identity":
        raise RunnerRefusal(
            f"--option unwarped_identity contradicts the contract's intervention_family "
            f"{value['intervention_family']!r}"
        )
    if option == "external_warp":
        raise RunnerRefusal(
            "Option A (external voltage registration) is not the selected candidate; its "
            "implementation dependency is unresolved in the contract."
        )

    # the digest the contract promises is recomputed, never trusted
    recomputed = canonical_digest(resolved)
    if recomputed != value["configuration_digest"]:
        raise RunnerRefusal(
            f"configuration_digest mismatch: contract says {value['configuration_digest'][:12]}, "
            f"recomputed {recomputed[:12]}"
        )

    comparators = contract["comparators"]
    source_sort_id = value["inputs"]["source_sort_id"]
    matching = [c for c in comparators.values() if c["sort_id"] == source_sort_id]
    if not matching:
        raise RunnerRefusal(f"declared source_sort_id {source_sort_id!r} names no comparator")
    comparator = matching[0]
    curated = Path(comparator["curated"])

    if snippet_dir is not None and Path(snippet_dir).resolve() != curated.resolve():
        _refuse_conflicting_override("--snippet-dir", str(snippet_dir), str(curated))
    if truth_path is not None:
        raise RunnerRefusal(
            "refusing --truth: a retained_sort_replay has no injected truth. Its endpoint is the "
            "production amplitude-completeness statistic on real rows."
        )
    if config_path is not None:
        supplied = json.loads(Path(config_path).read_text())
        if canonical_digest(supplied) != recomputed:
            _refuse_conflicting_override("--config", str(config_path), "the contract's own resolved_configuration")

    motion_node = resolved["motion"]
    if motion_node["mode"] == "declared_absent" and motion_info_dir is not None:
        _refuse_conflicting_override("--motion-info-dir", str(motion_info_dir), "motion.mode=declared_absent")
    if motion_node["mode"] == "qualified_field" and motion_info_dir is None:
        declared = motion_node.get("qualified_motion_field") or {}
        if not declared.get("path"):
            raise RunnerRefusal(
                "motion.mode is 'qualified_field' but the contract names no motion field and none "
                "was supplied. A motion-aware arm must identify the motion field it consumed; "
                "missing motion is never zero motion."
            )

    arms = resolved["execution"]["arms"]
    if arm not in arms:
        raise RunnerRefusal(f"the contract declares arms {sorted(arms)}, not {arm!r}")
    arm_node = arms[arm]
    endpoint = (float(arm_node["endpoint_interval_s"][0]), float(arm_node["endpoint_interval_s"][1]))
    processing = (
        float(arm_node["processing_interval_s"][0]),
        float(arm_node["processing_interval_s"][1]),
    )
    _check_intervals(contract, arm, endpoint, processing)

    identity_config = UnwarpedIdentityConfig(
        epoch_duration_s=float(resolved["identity_link"]["epoch_duration_s"]),
        epoch_overlap_s=float(resolved["identity_link"]["epoch_overlap_s"]),
        epoch_grid_origin_s=float(resolved["execution"]["epoch_grid_origin_s"]),
        min_spikes_per_epoch=int(resolved["identity_link"]["min_spikes_per_epoch"]),
        max_spatial_distance_um=float(resolved["identity_link"]["max_spatial_distance_um"]),
        max_amplitude_ratio=float(resolved["identity_link"]["max_amplitude_ratio"]),
        min_waveform_cosine=float(resolved["identity_link"]["min_waveform_cosine"]),
        waveform_channel_neighbourhood_um=float(
            resolved["identity_link"]["waveform_channel_neighbourhood_um"]
        ),
        ambiguity_threshold_ratio=float(resolved["identity_link"]["ambiguity_threshold_ratio"]),
        max_refractory_violation_increase=float(
            resolved["identity_link"]["max_refractory_violation_increase"]
        ),
        refractory_period_ms=float(resolved["identity_link"]["refractory_period_ms"]),
    )
    healthy_min_coverage = float(
        resolved["completeness_qc"].get("healthy_min_coverage_of_eligible_population", 0.5)
    )
    qc_node = resolved["completeness_qc"]
    completeness_config = CompletenessConfig(
        spikes_per_window=int(qc_node["spikes_per_window"]),
        max_isi_s=float(qc_node["max_isi_s"]),
        window_indexing=str(qc_node["window_indexing"]),
        max_family_amplitude_scale_ratio=float(qc_node["max_family_amplitude_scale_ratio"]),
        min_finite_interior_windows=int(qc_node.get("min_finite_interior_windows", 2)),
    )

    return {
        "contract": contract,
        "option": option,
        "arm": arm,
        "arm_name": arm_node["name"],
        "execution_mode": execution_mode,
        "intervention_family": value["intervention_family"],
        "source_sort_id": source_sort_id,
        "curated": curated,
        "qc_dir": Path(comparator["qc_dir"]),
        "endpoint_interval_s": endpoint,
        "processing_interval_s": processing,
        "identity_config": identity_config,
        "completeness_config": completeness_config,
        "healthy_min_coverage": healthy_min_coverage,
        "motion_node": motion_node,
        "motion_info_dir": motion_info_dir,
        "configuration_digest": recomputed,
        "fs_hz": float(contract["recording"]["sampling_frequency_hz"]),
        "duration_s": float(contract["recording"]["duration_s"]),
        "labels_node": resolved["labels"],
    }


def _check_intervals(
    contract: dict[str, Any],
    arm: str,
    endpoint: tuple[float, float],
    processing: tuple[float, float],
) -> None:
    """Bounds, recording limits, permitted context and sealed-panel disjointness.

    Checked at runtime against the contract that is actually loaded, not only in
    a test of the shipped JSON.
    """
    intervals = contract["intervals"]
    duration = float(contract["recording"]["duration_s"])
    for label, (start, stop) in (("endpoint", endpoint), ("processing", processing)):
        if not stop > start:
            raise RunnerRefusal(f"{arm} {label} interval must have stop > start, got [{start}, {stop}]")
        if start < 0.0 or stop > duration:
            raise RunnerRefusal(
                f"{arm} {label} interval [{start}, {stop}] leaves the recording [0, {duration}]"
            )
    if not (processing[0] <= endpoint[0] and endpoint[1] <= processing[1]):
        raise RunnerRefusal(
            f"{arm} endpoint interval {endpoint} is not inside its processing interval {processing}"
        )

    buffer_s = float(intervals["sealed_panel"]["exclusion_buffer_s"])
    sealed = [
        (start - buffer_s, stop + buffer_s)
        for start, stop in _pairs(intervals["sealed_panel"]["windows_s"])
    ]
    for window in sealed:
        if _overlaps(processing, window):
            raise RunnerRefusal(
                f"{arm} processing interval {processing} intersects sealed window {window} "
                "(expanded by the exclusion buffer); the sealed panel is not consumed here"
            )

    development = _pairs(intervals["development_windows"]["windows_s"])
    healthy = _pairs(intervals["healthy_control_intervals"]["windows"])
    if arm == "case":
        # named before the containment check below, which would also refuse it
        # but only say "not in a development window"
        for window in healthy:
            if _overlaps(processing, window):
                raise RunnerRefusal(
                    f"the case arm's processing interval {processing} intersects reserved healthy "
                    f"evaluation interval {window}"
                )
        permitted = development
        why = "development windows"
    else:
        # the healthy arm is allowed its own reserved evaluation interval, and
        # only that one, on top of the development windows
        matching = [w for w in healthy if w[0] <= endpoint[0] and endpoint[1] <= w[1]]
        if not matching:
            raise RunnerRefusal(
                f"the healthy arm's endpoint interval {endpoint} is not one of the contract's "
                f"reserved healthy control intervals {healthy}"
            )
        permitted = development + matching
        why = "development windows plus this arm's own reserved healthy control interval"
    if not _covered_by(processing, permitted):
        raise RunnerRefusal(
            f"{arm} processing interval {processing} is not contained in {why}; a run may read "
            "only its declared interval and permitted context"
        )


# --------------------------------------------------------------------------- #
# loading the declared inputs
# --------------------------------------------------------------------------- #
def load_replay_rows(resolved: dict[str, Any]) -> dict[str, Any]:
    """Read the declared curated sort and keep only the declared interval."""
    curated = resolved["curated"]
    arrays = load_curated_arrays(resolved["source_sort_id"], curated)

    extra: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name in EXTRA_HASHED_FILES:
        path = curated / name
        if not path.exists():
            raise RunnerRefusal(
                f"{name} is missing from {curated}. Real depths and a physical-channel waveform "
                "representation are required inputs; they are never defaulted to zeros."
            )
        extra[name], digests[name] = _read_hashed_array(path)

    positions = np.asarray(extra["spike_positions.npy"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise RunnerRefusal("spike_positions.npy must be (n_spikes, >=2) to supply real depths")
    depth_full = positions[:, 1]
    if depth_full.shape[0] != arrays.times.size:
        raise RunnerRefusal(
            f"spike_positions.npy has {depth_full.shape[0]} rows for {arrays.times.size} spikes"
        )

    templates_full = np.asarray(extra["spike_templates.npy"]).reshape(-1)
    if templates_full.size != arrays.times.size:
        raise RunnerRefusal(
            f"spike_templates.npy has {templates_full.size} rows for {arrays.times.size} spikes"
        )

    # `templates.npy` as KS4 exports it is in the WHITENED space, so its
    # per-channel amplitudes are not the waveform on the probe: the whitening
    # matrix mixes neighbouring channels, which can move a peak channel and
    # change a similarity score. The contract declares a *physical* channel
    # representation, so undo it, with the same convention the repo's donor
    # implementation uses (`testing/ladder_donors.py::_dewhitened_shape`).
    dewhitened = np.asarray(extra["templates.npy"], dtype=np.float64) @ np.asarray(
        extra["whitening_mat_inv.npy"], dtype=np.float64
    )

    fs = resolved["fs_hz"]
    start_s, stop_s = resolved["processing_interval_s"]
    seconds = arrays.times.astype(np.float64) / fs
    inside = np.flatnonzero((seconds >= start_s) & (seconds < stop_s))

    # `arrays` is stable-time-sorted and carries the original row ids, so the
    # subset below keeps original row identity while the export keeps the
    # original integer sample clock.
    row_id = arrays.row_id[inside]
    inputs = ReplayInput(
        row_id=row_id,
        sample=arrays.times[inside],
        cluster=arrays.clusters[inside].astype(np.int64),
        depth_um=depth_full[arrays.row_id[inside]],
        amplitude=arrays.amplitudes[inside],
        template=templates_full[arrays.row_id[inside]].astype(np.int64),
        template_bank=dewhitened,
        channel_positions_um=np.asarray(extra["channel_positions.npy"], dtype=np.float64),
        fs_hz=fs,
    )
    whitened = np.asarray(extra["templates.npy"], dtype=np.float64)
    peak_whitened = np.argmax(whitened.max(axis=1) - whitened.min(axis=1), axis=1)
    peak_physical = np.argmax(dewhitened.max(axis=1) - dewhitened.min(axis=1), axis=1)
    return {
        "inputs": inputs,
        "waveform_representation": {
            "declared": "probe_physical_channels",
            "transform": "templates.npy @ whitening_mat_inv.npy",
            "why": (
                "KS4 exports templates in the whitened space; whitening mixes neighbouring "
                "channels, so untransformed templates are not a physical-channel waveform."
            ),
            "n_templates": int(dewhitened.shape[0]),
            "n_templates_whose_peak_channel_moved": int(
                np.count_nonzero(peak_whitened != peak_physical)
            ),
        },
        "n_rows_total": int(arrays.times.size),
        "n_rows_in_interval": int(inside.size),
        "n_rows_outside_interval_not_processed": int(arrays.times.size - inside.size),
        "was_time_ordered": bool(arrays.was_time_ordered),
        "extra_input_sha256": digests,
    }


def resolve_motion(resolved: dict[str, Any], inputs: ReplayInput) -> MotionDeclaration:
    """Build the motion declaration, or refuse. Never a silent zero."""
    node = resolved["motion_node"]
    mode = node["mode"]
    if mode == "declared_absent":
        return MotionDeclaration(mode="declared_absent", rationale=node.get("rationale", ""))
    if mode != "qualified_field":
        raise RunnerRefusal(f"unknown motion mode {mode!r} in the contract")

    declared = node.get("qualified_motion_field") or {}
    path = Path(declared["path"])
    data = path.read_bytes()
    digest = _sha256_bytes(data)
    if declared.get("sha256") and declared["sha256"] != digest:
        raise RunnerRefusal(
            f"the motion field at {path} hashes to {digest[:12]}, not the declared "
            f"{declared['sha256'][:12]}"
        )
    # duration supplied so a field carrying acquisition-clock times is refused
    # rather than silently interpolated against the wrong origin
    field = load_qualified_motion_field(path, recording_duration_s=resolved["duration_s"])
    interpolated = interpolate_motion_at_spikes(field, inputs.seconds(), inputs.depth_um)
    return MotionDeclaration(
        mode="qualified_field",
        displacement_um=interpolated["displacement_um"],
        field_identity={"path": str(path), "sha256": digest, **field["metadata"]},
        rationale=node.get("rationale", ""),
    )


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def _read_label_table(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    with path.open() as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return {}
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                out[int(row[0])] = row[1].strip()
    return out


def export_arm(
    out_dir: Path,
    *,
    inputs: ReplayInput,
    unit_of_row: dict[int, int],
    contributors: dict[int, list[int]],
    curated: Path,
    arm_label: str,
) -> dict[str, Any]:
    """Write one arm's train, its labels, and where each label came from.

    Original labels are preserved verbatim alongside the new table rather than
    overwritten, and a family assembled from more than one original cluster is
    labelled ``unvalidated`` -- never ``good``. Nothing downstream has looked at
    it yet, and this runner is not entitled to say it is a well isolated unit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    row_index = {int(r): i for i, r in enumerate(inputs.row_id.tolist())}
    ordered_rows = np.array(sorted(unit_of_row), dtype=np.int64)
    positions = np.array([row_index[int(r)] for r in ordered_rows], dtype=np.int64)

    np.save(out_dir / "spike_times.npy", inputs.sample[positions])
    np.save(out_dir / "spike_clusters.npy",
            np.array([unit_of_row[int(r)] for r in ordered_rows], dtype=np.int64))
    np.save(out_dir / "spike_row_id.npy", ordered_rows)
    # Deliberately NOT called amplitudes.npy: that is a different observable.
    np.save(out_dir / "qc_amplitudes.npy", inputs.amplitude[positions])

    original_group = _read_label_table(curated / "cluster_group.tsv")
    original_ks = _read_label_table(curated / "cluster_KSLabel.tsv")
    for name in ("cluster_group.tsv", "cluster_KSLabel.tsv"):
        source = curated / name
        if source.exists():
            shutil.copy(source, out_dir / name.replace(".tsv", ".original.tsv"))

    rows: list[dict[str, Any]] = []
    for unit in sorted(set(unit_of_row.values())):
        cids = contributors.get(unit, [])
        if len(cids) == 1:
            label = original_group.get(cids[0]) or original_ks.get(cids[0]) or "unsorted"
            source = f"original label of cluster {cids[0]}"
        else:
            label = UNVALIDATED_LABEL
            source = "new family assembled from a link; not validated downstream yet"
        rows.append(
            {
                "unit_id": unit,
                "label": label,
                "label_source": source,
                "contributing_original_clusters": " ".join(str(c) for c in cids),
                "n_contributing_clusters": len(cids),
            }
        )

    with (out_dir / "cluster_group.tsv").open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["cluster_id", "group"])
        for row in rows:
            writer.writerow([row["unit_id"], row["label"]])
    with (out_dir / "unit_provenance.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["unit_id"])
        writer.writeheader()
        writer.writerows(rows)

    label_counts: dict[str, int] = {}
    for row in rows:
        label_counts[row["label"]] = label_counts.get(row["label"], 0) + 1
    return {
        "arm": arm_label,
        "dir": str(out_dir),
        "n_rows": int(ordered_rows.size),
        "n_units": len(rows),
        "label_counts": label_counts,
        "n_unvalidated_families": sum(1 for r in rows if r["label"] == UNVALIDATED_LABEL),
        "original_labels_preserved": [
            p.name for p in sorted(out_dir.glob("*.original.tsv"))
        ],
    }


# --------------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------------- #
def _unit_rows(unit_of_row: dict[int, int], unit: int) -> np.ndarray:
    return np.array(sorted(r for r, u in unit_of_row.items() if u == unit), dtype=np.int64)


def evaluate_case_arm(
    *,
    inputs: ReplayInput,
    baseline_of_row: dict[int, int],
    candidate_of_row: dict[int, int],
    contributors: dict[int, list[int]],
    cluster_id: int,
    endpoint_interval_s: tuple[float, float],
    completeness_config: CompletenessConfig,
    margins: dict[str, Any],
) -> dict[str, Any]:
    """Completeness, identity retention and contamination for the named case."""
    row_index = {int(r): i for i, r in enumerate(inputs.row_id.tolist())}
    fs = inputs.fs_hz
    start_s, stop_s = endpoint_interval_s
    seconds = inputs.sample.astype(np.float64) / fs

    baseline_rows = _unit_rows(baseline_of_row, cluster_id)
    if baseline_rows.size == 0:
        raise RunnerRefusal(
            f"cluster {cluster_id} has no retained rows inside the processing interval"
        )
    baseline_pos = np.array([row_index[int(r)] for r in baseline_rows], dtype=np.int64)
    in_endpoint = (seconds[baseline_pos] >= start_s) & (seconds[baseline_pos] < stop_s)
    endpoint_rows = baseline_rows[in_endpoint]

    # the family carrying the neuron: the one holding the plurality of the
    # baseline cluster's endpoint-interval rows
    families = [candidate_of_row[int(r)] for r in endpoint_rows]
    if not families:
        raise RunnerRefusal(
            f"cluster {cluster_id} has no rows inside the endpoint interval {endpoint_interval_s}"
        )
    values, counts = np.unique(np.asarray(families), return_counts=True)
    carrier = int(values[int(np.argmax(counts))])
    retained_fraction = float(counts.max() / counts.sum())

    candidate_rows = _unit_rows(candidate_of_row, carrier)
    candidate_pos = np.array([row_index[int(r)] for r in candidate_rows], dtype=np.int64)

    baseline_qc = train_completeness(
        inputs.sample[baseline_pos], inputs.amplitude[baseline_pos],
        fs_hz=fs, interval_s=endpoint_interval_s, config=completeness_config,
        label=f"baseline cluster {cluster_id}",
    )
    candidate_qc = train_completeness(
        inputs.sample[candidate_pos], inputs.amplitude[candidate_pos],
        fs_hz=fs, interval_s=endpoint_interval_s, config=completeness_config,
        label=f"candidate family {carrier}",
    )

    contributor_medians: dict[int, float] = {}
    for cid in contributors.get(carrier, []):
        rows = _unit_rows(baseline_of_row, cid)
        if rows.size == 0:
            continue
        pos = np.array([row_index[int(r)] for r in rows], dtype=np.int64)
        window = (seconds[pos] >= start_s) & (seconds[pos] < stop_s)
        if not window.any():
            continue
        contributor_medians[cid] = float(np.median(inputs.amplitude[pos][window]))
    scale_check = family_amplitude_scale_check(contributor_medians, completeness_config)

    completeness = compare_completeness(
        baseline_qc, candidate_qc,
        scale_check=scale_check,
        margin_pp=float(margins["completeness"]["value"]),
        config=completeness_config,
    )

    def _rvf(positions: np.ndarray) -> float:
        window = (seconds[positions] >= start_s) & (seconds[positions] < stop_s)
        return refractory_violation_fraction(inputs.sample[positions][window], fs, 1.5)

    baseline_rvf = _rvf(baseline_pos)
    candidate_rvf = _rvf(candidate_pos)
    contamination_margin = float(margins["contamination"]["value"])
    identity_floor = float(margins["identity"]["value"])

    return {
        "cluster_id": cluster_id,
        "carrier_family": carrier,
        "carrier_contributing_clusters": contributors.get(carrier, []),
        "endpoint_interval_s": [start_s, stop_s],
        "completeness": {
            "baseline": baseline_qc,
            "candidate": candidate_qc,
            "family_amplitude_scale_check": scale_check,
            "gate": completeness,
        },
        "identity": {
            "retained_fraction": retained_fraction,
            "floor": identity_floor,
            "n_baseline_endpoint_rows": int(endpoint_rows.size),
            "n_families_the_baseline_cluster_split_into": int(values.size),
            "verdict": VERDICT_PASS if retained_fraction >= identity_floor else "fail",
            "decision_rule": margins["identity"]["decision_rule"],
        },
        "contamination": {
            "baseline_refractory_violation_fraction": baseline_rvf,
            "candidate_refractory_violation_fraction": candidate_rvf,
            "increase": candidate_rvf - baseline_rvf,
            "max_tolerated_increase": contamination_margin,
            "scored_on": "the complete exported train over the endpoint interval, not an anchor",
            "verdict": (
                VERDICT_PASS
                if (candidate_rvf - baseline_rvf) <= contamination_margin
                else "fail"
            ),
        },
    }


def evaluate_healthy_arm(
    *,
    inputs: ReplayInput,
    baseline_of_row: dict[int, int],
    candidate_of_row: dict[int, int],
    contributors: dict[int, list[int]],
    endpoint_interval_s: tuple[float, float],
    completeness_config: CompletenessConfig,
    margins: dict[str, Any],
    min_coverage: float = 0.5,
) -> dict[str, Any]:
    """Per-cluster completeness preservation on a reserved healthy interval."""
    row_index = {int(r): i for i, r in enumerate(inputs.row_id.tolist())}
    fs = inputs.fs_hz
    start_s, stop_s = endpoint_interval_s
    seconds = inputs.sample.astype(np.float64) / fs
    tolerance = float(margins["healthy_interval_preservation"]["value"])

    rows_by_unit: dict[int, list[int]] = {}
    for row, unit in baseline_of_row.items():
        rows_by_unit.setdefault(unit, []).append(row)

    per_cluster: list[dict[str, Any]] = []
    for cluster_id, rows in sorted(rows_by_unit.items()):
        pos = np.array([row_index[int(r)] for r in rows], dtype=np.int64)
        window = (seconds[pos] >= start_s) & (seconds[pos] < stop_s)
        if window.sum() < completeness_config.spikes_per_window:
            continue
        baseline_qc = train_completeness(
            inputs.sample[pos], inputs.amplitude[pos], fs_hz=fs,
            interval_s=endpoint_interval_s, config=completeness_config,
            label=f"baseline cluster {cluster_id}",
        )
        endpoint_rows = np.asarray(rows, dtype=np.int64)[window]
        families = [candidate_of_row[int(r)] for r in endpoint_rows]
        values, counts = np.unique(np.asarray(families), return_counts=True)
        carrier = int(values[int(np.argmax(counts))])
        candidate_rows = _unit_rows(candidate_of_row, carrier)
        candidate_pos = np.array([row_index[int(r)] for r in candidate_rows], dtype=np.int64)
        candidate_qc = train_completeness(
            inputs.sample[candidate_pos], inputs.amplitude[candidate_pos], fs_hz=fs,
            interval_s=endpoint_interval_s, config=completeness_config,
            label=f"candidate family {carrier}",
        )
        contributor_medians: dict[int, float] = {}
        for cid in contributors.get(carrier, []):
            crows = _unit_rows(baseline_of_row, cid)
            if crows.size == 0:
                continue
            cpos = np.array([row_index[int(r)] for r in crows], dtype=np.int64)
            cwin = (seconds[cpos] >= start_s) & (seconds[cpos] < stop_s)
            if cwin.any():
                contributor_medians[cid] = float(np.median(inputs.amplitude[cpos][cwin]))
        scale_check = family_amplitude_scale_check(contributor_medians, completeness_config)
        measurable = (
            baseline_qc["status"] == "measured"
            and candidate_qc["status"] == "measured"
            and scale_check["compatible"]
        )
        per_cluster.append(
            {
                "cluster_id": cluster_id,
                "carrier_family": carrier,
                "retained_fraction": float(counts.max() / counts.sum()),
                "baseline_missing_pct": baseline_qc["missing_pct_median"],
                "candidate_missing_pct": candidate_qc["missing_pct_median"],
                "increase_pp": (
                    float(candidate_qc["missing_pct_median"] - baseline_qc["missing_pct_median"])
                    if measurable else None
                ),
                "measurable": measurable,
                "baseline_status": baseline_qc["status"],
                "candidate_status": candidate_qc["status"],
                "amplitude_scale_compatible": scale_check["compatible"],
            }
        )

    # The eligible population is fixed before anything is measured: every
    # original cluster with rows in the endpoint interval. Completeness can only
    # be compared on the subset with enough fits, so coverage is reported against
    # that fixed denominator rather than against whatever happened to be
    # measurable -- four comparisons say nothing about the other ninety-six.
    eligible = sorted({c for r, c in baseline_of_row.items()
                       if start_s <= seconds[row_index[int(r)]] < stop_s})
    identity = _identity_preservation(
        inputs=inputs, baseline_of_row=baseline_of_row, candidate_of_row=candidate_of_row,
        eligible=eligible, row_index=row_index, seconds=seconds,
        start_s=start_s, stop_s=stop_s,
    )

    increases = [c["increase_pp"] for c in per_cluster if c["increase_pp"] is not None]
    coverage = len(increases) / len(eligible) if eligible else 0.0
    if not increases or coverage < min_coverage:
        completeness_verdict, worst = "inconclusive", (max(increases) if increases else None)
    else:
        worst = max(increases)
        completeness_verdict = VERDICT_PASS if worst <= tolerance else "fail"

    # Preservation is the conjunction: a completeness comparison on a handful of
    # clusters cannot certify an arm that fragmented the rest.
    if identity["verdict"] == "fail":
        verdict = "fail"
    elif completeness_verdict == VERDICT_PASS:
        verdict = VERDICT_PASS
    else:
        verdict = completeness_verdict

    return {
        "endpoint_interval_s": [start_s, stop_s],
        "max_tolerated_increase_pp": tolerance,
        "verdict": verdict,
        "identity_preservation": identity,
        "completeness": {
            "verdict": completeness_verdict,
            "n_eligible_clusters": len(eligible),
            "n_clusters_with_enough_spikes_to_consider": len(per_cluster),
            "n_clusters_measurable_in_both_arms": len(increases),
            "coverage_of_eligible_population": coverage,
            "min_coverage_required": min_coverage,
            "worst_increase_pp": worst,
            "note": (
                "Coverage below the required fraction makes the completeness comparison "
                "`inconclusive`, never `pass`: the unmeasured clusters are not evidence of "
                "preservation."
            ),
        },
        "per_cluster": per_cluster,
        "decision_rule": margins["healthy_interval_preservation"]["decision_rule"],
    }


def _identity_preservation(
    *,
    inputs: ReplayInput,
    baseline_of_row: dict[int, int],
    candidate_of_row: dict[int, int],
    eligible: list[int],
    row_index: dict[int, int],
    seconds: np.ndarray,
    start_s: float,
    stop_s: float,
) -> dict[str, Any]:
    """Did the candidate keep every eligible cluster intact over the interval?

    Measured on the whole eligible population, not on the completeness subset.
    A cluster is preserved when all of its rows in the interval carry one family
    id. Any split is a failure of the invariant, reported as such.
    """
    families: dict[int, set[int]] = {}
    for row, cluster in baseline_of_row.items():
        if start_s <= seconds[row_index[int(row)]] < stop_s:
            families.setdefault(cluster, set()).add(candidate_of_row[int(row)])
    split = sorted(c for c in eligible if len(families.get(c, set())) > 1)
    merged: dict[int, set[int]] = {}
    for cluster in eligible:
        for family in families.get(cluster, set()):
            merged.setdefault(family, set()).add(cluster)
    return {
        "n_eligible_clusters": len(eligible),
        "n_clusters_preserved_intact": len(eligible) - len(split),
        "n_clusters_split": len(split),
        "clusters_split": split[:50],
        "n_families_merged_from_multiple_clusters": sum(
            1 for cs in merged.values() if len(cs) > 1
        ),
        "coverage_of_eligible_population": 1.0,
        "verdict": VERDICT_PASS if not split else "fail",
        "note": (
            "Identity preservation is measured on every eligible cluster, so it has full "
            "coverage even when completeness does not."
        ),
    }


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run_bakeoff(
    *,
    option: str,
    mode: str,
    arm: str = "case",
    out_root: Path | None = None,
    snippet_dir: Path | None = None,
    motion_info_dir: Path | None = None,
    truth_path: Path | None = None,
    config_path: Path | None = None,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    if option not in OPTIONS:
        raise RunnerRefusal(f"unknown option {option!r}, must be one of {OPTIONS}")
    if mode not in MODES:
        raise RunnerRefusal(f"unknown mode {mode!r}, must be one of {MODES}")
    if arm not in ARMS:
        raise RunnerRefusal(f"unknown arm {arm!r}, must be one of {ARMS}")

    contract = load_contract(contract_path)
    resolved = resolve_settings(
        contract, option=option, arm=arm, snippet_dir=snippet_dir,
        motion_info_dir=motion_info_dir, truth_path=truth_path, config_path=config_path,
    )

    root = reject_unsafe_out_root(
        Path(out_root) if out_root is not None else Path(contract["output_root"]),
        [resolved["curated"], resolved["qc_dir"]],
    )
    validation_mode = "execution" if mode in EXECUTION_MODES else "authoring"
    report = validate(contract_path, mode=validation_mode, out_root=root)

    print("--- first pipeline candidate: retained-sort replay ---")
    print(f"contract          : {contract.get('contract_id')} ({validation_mode})")
    print(f"option / arm/ mode: {option} / {arm} ({resolved['arm_name']}) / {mode}")
    print(f"execution mode    : {resolved['execution_mode']}")
    print(f"source sort       : {resolved['source_sort_id']} -> {resolved['curated']}")
    print(f"config digest     : {resolved['configuration_digest'][:12]}")
    print(f"motion            : {resolved['motion_node']['mode']}")
    print(f"processing / endpt: {resolved['processing_interval_s']} / {resolved['endpoint_interval_s']}")
    print(f"out root          : {root}")

    root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {"validation": report.to_dict()}
    status = "completed"

    try:
        if mode == "verify":
            stages["verify"] = _verify(resolved)
        else:
            stages.update(_execute(resolved, root, option))
    except Exception as exc:  # a manifest is written even when a stage fails
        status = "failed"
        stages["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
        raise
    finally:
        manifest = {
            "schema": BAKEOFF_SCHEMA,
            "option": option,
            "arm": arm,
            "arm_name": resolved["arm_name"],
            "mode": mode,
            "status": status,
            "contract_id": contract.get("contract_id"),
            "contract_digest": report.contract_digest,
            "configuration_digest": resolved["configuration_digest"],
            "execution_mode": resolved["execution_mode"],
            "out_root": str(root),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stages": stages,
            "provenance": git_worktree_state(),
        }
        _atomic_write(
            root / f"{arm}__{option}__{BAKEOFF_MANIFEST}",
            json.dumps(manifest, indent=2, default=str) + "\n",
        )
    print(f"manifest written to {root / f'{arm}__{option}__{BAKEOFF_MANIFEST}'}")
    return manifest


def _verify(resolved: dict[str, Any]) -> dict[str, Any]:
    """Check the inputs execution will actually consume, and nothing else."""
    curated = resolved["curated"]
    required = ("spike_times.npy", "spike_clusters.npy", "full_st.npy", "kept_spikes.npy")
    missing = [n for n in (*required, *EXTRA_HASHED_FILES) if not (curated / n).exists()]
    if missing:
        raise RunnerRefusal(f"declared input {curated} is missing files execution needs: {missing}")
    digests = {n: _sha256_bytes((curated / n).read_bytes()) for n in EXTRA_HASHED_FILES}
    return {
        "declared_input": str(curated),
        "files_execution_will_consume": sorted((*required, *EXTRA_HASHED_FILES)),
        "extra_input_sha256": digests,
        "processing_interval_s": list(resolved["processing_interval_s"]),
        "endpoint_interval_s": list(resolved["endpoint_interval_s"]),
        "motion_mode": resolved["motion_node"]["mode"],
        "passed": True,
    }


def _execute(resolved: dict[str, Any], root: Path, option: str) -> dict[str, Any]:
    loaded = load_replay_rows(resolved)
    inputs: ReplayInput = loaded["inputs"]
    arm = resolved["arm"]
    stages: dict[str, Any] = {"inputs": {k: v for k, v in loaded.items() if k != "inputs"}}

    baseline_of_row = {int(r): int(c) for r, c in zip(inputs.row_id.tolist(), inputs.cluster.tolist())}
    baseline_contributors = {c: [c] for c in sorted(set(baseline_of_row.values()))}
    baseline_export = export_arm(
        root / f"{arm}__baseline_export",
        inputs=inputs, unit_of_row=baseline_of_row, contributors=baseline_contributors,
        curated=resolved["curated"], arm_label="baseline",
    )
    stages["baseline_export"] = baseline_export

    if option == "control":
        stages["control"] = {
            "note": (
                "control arm: the retained sort replayed through the same export interface, so "
                "the candidate has a measured comparator rather than an absent one."
            )
        }
        return stages

    motion = resolve_motion(resolved, inputs)
    replay = run_unwarped_identity_replay(
        inputs,
        motion=motion,
        config=resolved["identity_config"],
        processing_interval_s=resolved["processing_interval_s"],
        output_dir=root / f"{arm}__unwarped_identity",
    )
    stages["unwarped_identity"] = replay["manifest"]

    candidate_of_row = {int(r): int(f) for r, f in replay["row_family"].items()}
    contributors = {int(f): list(c) for f, c in replay["family_contributors"].items()}

    # The partition is over original clusters, so every row in the interval
    # already has a family through its cluster -- including rows in no eligible
    # epoch. There is no `unassigned` class to give an extra family id to, which
    # is what produced a second fragment per cluster in v1.
    missing = set(baseline_of_row) - set(candidate_of_row)
    if missing:
        raise RunnerRefusal(
            f"{len(missing)} retained rows in the interval received no family. Every row must "
            "reach the export through its original cluster; this is the preservation invariant."
        )
    stages["preservation"] = _check_partition_preserved(baseline_of_row, candidate_of_row)

    stages["candidate_export"] = export_arm(
        root / f"{arm}__candidate_export",
        inputs=inputs, unit_of_row=candidate_of_row, contributors=contributors,
        curated=resolved["curated"], arm_label="candidate",
    )

    margins = _contract_margins(resolved)
    if arm == "case":
        cluster_id = _case_cluster_id(resolved)
        stages["endpoints"] = evaluate_case_arm(
            inputs=inputs,
            baseline_of_row=baseline_of_row,
            candidate_of_row=candidate_of_row,
            contributors=contributors,
            cluster_id=cluster_id,
            endpoint_interval_s=resolved["endpoint_interval_s"],
            completeness_config=resolved["completeness_config"],
            margins=margins,
        )
    else:
        stages["endpoints"] = evaluate_healthy_arm(
            inputs=inputs,
            baseline_of_row=baseline_of_row,
            candidate_of_row=candidate_of_row,
            contributors=contributors,
            endpoint_interval_s=resolved["endpoint_interval_s"],
            completeness_config=resolved["completeness_config"],
            margins=margins,
            min_coverage=resolved["healthy_min_coverage"],
        )
    _atomic_write(
        root / f"{arm}__endpoints.json",
        json.dumps(stages["endpoints"], indent=2, default=str) + "\n",
    )
    return stages


def _check_partition_preserved(
    baseline_of_row: dict[int, int], candidate_of_row: dict[int, int]
) -> dict[str, Any]:
    """Measure what the candidate did to the input partition, on the real rows.

    Reported for every run, not only when something is wrong. Three numbers say
    the whole story: how many original clusters were split (must be zero -- this
    replay has no splitting operation), how many families were merged from more
    than one cluster, and whether the two partitions are identical up to
    renumbering.
    """
    families_of_cluster: dict[int, set[int]] = {}
    clusters_of_family: dict[int, set[int]] = {}
    for row, cluster in baseline_of_row.items():
        family = candidate_of_row[row]
        families_of_cluster.setdefault(cluster, set()).add(family)
        clusters_of_family.setdefault(family, set()).add(cluster)

    split = sorted(c for c, f in families_of_cluster.items() if len(f) > 1)
    merged = sorted(f for f, c in clusters_of_family.items() if len(c) > 1)
    return {
        "n_original_clusters": len(families_of_cluster),
        "n_candidate_families": len(clusters_of_family),
        "n_original_clusters_split": len(split),
        "original_clusters_split": split[:50],
        "n_families_merged_from_multiple_clusters": len(merged),
        "families_merged_from_multiple_clusters": merged[:50],
        "input_partition_preserved_exactly": not split and not merged,
        "n_rows": len(baseline_of_row),
        "rule": (
            "Splitting an original cluster is not an operation this replay has. A non-zero "
            "n_original_clusters_split is a defect, not a result."
        ),
    }


def _contract_margins(resolved: dict[str, Any]) -> dict[str, Any]:
    return dict(resolved["contract"]["acceptance"]["margins"])


def _case_cluster_id(resolved: dict[str, Any]) -> int:
    failure = resolved["contract"]["acceptance"]["practical_failure"]["value"]
    if failure["sort_id"] != resolved["source_sort_id"]:
        raise RunnerRefusal(
            f"the named practical failure belongs to sort {failure['sort_id']!r}, but the "
            f"candidate replays {resolved['source_sort_id']!r}"
        )
    return int(failure["cluster_id"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--option", choices=OPTIONS, required=True)
    ap.add_argument("--arm", choices=ARMS, default="case")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--snippet-dir", type=Path, default=None)
    ap.add_argument("--motion-info-dir", type=Path, default=None)
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = ap.parse_args(argv)

    try:
        run_bakeoff(
            option=args.option, arm=args.arm, mode=args.mode, out_root=args.out_root,
            snippet_dir=args.snippet_dir, motion_info_dir=args.motion_info_dir,
            truth_path=args.truth, config_path=args.config, contract_path=args.contract,
        )
    except (RunnerRefusal, ContractRefusal) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
