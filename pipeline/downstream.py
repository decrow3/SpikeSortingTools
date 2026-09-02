"""Identity-guarded downstream stages for an accepted rescue sort.

These helpers deliberately keep the legacy curation and QC implementations,
while adding the production safeguards that the historical run sheets lacked:
an immutable source-sort identity, restart requests, completion receipts, and
explicit refusal to reuse outputs produced from another sort.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import fingerprint
from .sorting import SORT_MANIFEST


SORT_IDENTITY_SCHEMA = "rescue-sort-identity-v1"
STAGE_REQUEST_SCHEMA = "rescue-downstream-stage-request-v1"
STAGE_RECEIPT_SCHEMA = "rescue-downstream-stage-receipt-v1"

IDENTITY_FILES = (
    "spike_times.npy",
    "spike_clusters.npy",
    "templates.npy",
    "cluster_KSLabel.tsv",
    "ops.npy",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, path)


def _sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_sort_identity(kilosort_dir: Path) -> dict[str, Any]:
    """Fingerprint the complete sort manifest and interpretation-critical files."""
    kilosort_dir = Path(kilosort_dir)
    manifest_path = kilosort_dir / SORT_MANIFEST
    sorter_output = kilosort_dir / "sorter_output"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sort manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True:
        raise RuntimeError(f"Sort is not complete: {manifest_path}")
    files = {}
    for name in IDENTITY_FILES:
        path = sorter_output / name
        if not path.is_file():
            raise FileNotFoundError(f"Sort identity input is missing: {path}")
        files[name] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    payload = {
        "schema_version": SORT_IDENTITY_SCHEMA,
        "sort_directory": str(kilosort_dir.resolve()),
        "request_digest": manifest.get("request_digest"),
        "recording_request_digest": manifest.get("recording_request_digest"),
        "sort_summary": manifest.get("summary"),
        "files": files,
    }
    payload["identity_digest"] = fingerprint(payload)
    return payload


def pin_sort_identity(kilosort_dir: Path, identity_path: Path) -> dict[str, Any]:
    """Create an immutable identity receipt, or verify the existing receipt."""
    current = build_sort_identity(kilosort_dir)
    identity_path = Path(identity_path)
    if identity_path.exists():
        pinned = json.loads(identity_path.read_text())
        if pinned.get("identity_digest") != current["identity_digest"]:
            raise RuntimeError(
                "Current sorter files differ from the pinned sort identity; "
                "refusing to attach downstream outputs"
            )
        return pinned
    pinned = {**current, "pinned_at": _utc_now()}
    _atomic_json(identity_path, pinned)
    return pinned


def validate_sort_identity(kilosort_dir: Path, identity_path: Path) -> dict[str, Any]:
    if not Path(identity_path).is_file():
        raise FileNotFoundError(f"Missing pinned sort identity: {identity_path}")
    return pin_sort_identity(kilosort_dir, identity_path)


def _required_files_exist(paths: Iterable[Path]) -> bool:
    return all(Path(path).is_file() for path in paths)


def ensure_stage_request(
    request_path: Path,
    *,
    stage: str,
    sort_identity: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Pin a stage request before expensive work so interruptions can resume."""
    payload = {
        "schema_version": STAGE_REQUEST_SCHEMA,
        "stage": stage,
        "sort_identity_digest": sort_identity["identity_digest"],
        "settings": settings,
    }
    payload["request_digest"] = fingerprint(payload)
    request_path = Path(request_path)
    if request_path.exists():
        saved = json.loads(request_path.read_text())
        if saved.get("request_digest") != payload["request_digest"]:
            raise RuntimeError(f"Existing {stage} request belongs to another configuration")
        return saved
    payload["created_at"] = _utc_now()
    _atomic_json(request_path, payload)
    return payload


def completed_stage_receipt(
    receipt_path: Path,
    *,
    request: dict[str, Any],
    required_files: Iterable[Path],
) -> dict[str, Any] | None:
    receipt_path = Path(receipt_path)
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("request_digest") != request["request_digest"]:
        raise RuntimeError(f"Existing stage receipt does not match {request['stage']}")
    if receipt.get("complete") is not True or not _required_files_exist(required_files):
        raise RuntimeError(f"Incomplete receipt for stage {request['stage']}")
    return receipt


def write_stage_receipt(
    receipt_path: Path,
    *,
    request: dict[str, Any],
    required_files: Iterable[Path],
    summary: dict[str, Any],
) -> dict[str, Any]:
    required = [Path(path) for path in required_files]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Stage {request['stage']} is missing outputs: {missing}")
    receipt = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "stage": request["stage"],
        "request_digest": request["request_digest"],
        "sort_identity_digest": request["sort_identity_digest"],
        "complete": True,
        "completed_at": _utc_now(),
        "required_files": [str(path.resolve()) for path in required],
        "summary": summary,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def run_diagnostics_stage(
    sorter_output: Path,
    output_dir: Path,
    sort_identity: dict[str, Any],
    *,
    probe: str,
    duration_s: float,
    criteria_path: Path,
    jitter_replicates: int = 250,
    seed: int = 20250804,
) -> dict[str, Any]:
    """Run the frozen full-probe evaluator with an identity-bound receipt."""
    output_dir = Path(output_dir)
    request = ensure_stage_request(
        output_dir / "diagnostics_request.json",
        stage="full_probe_diagnostics",
        sort_identity=sort_identity,
        settings={
            "sorter_output": str(Path(sorter_output).resolve()),
            "probe": probe,
            "duration_s": duration_s,
            "jitter_replicates": jitter_replicates,
            "seed": seed,
            "criteria_path": str(Path(criteria_path).resolve()),
        },
    )
    required = (
        output_dir / "summary.json",
        output_dir / "unit_metrics.csv",
        output_dir / "similar_template_pairs.csv",
        output_dir / "acceptance_decision.json",
    )
    receipt_path = output_dir / "diagnostics_receipt.json"
    if receipt := completed_stage_receipt(
        receipt_path, request=request, required_files=required
    ):
        return receipt
    from testing.luke_full_probe_rescue_diagnostics import run as run_diagnostics
    from testing.luke_imec0_rescue_acceptance import evaluate

    summary = run_diagnostics(
        Path(sorter_output),
        output_dir,
        jitter_replicates,
        seed,
        probe=probe,
        duration_override_s=duration_s,
    )
    decision = evaluate(output_dir, Path(criteria_path))
    _atomic_json(output_dir / "acceptance_decision.json", decision)
    return write_stage_receipt(
        receipt_path,
        request=request,
        required_files=required,
        summary={
            "n_ks_good": summary["n_ks_good"],
            "nearby_similar_good_good_pairs": summary[
                "nearby_similar_good_good_pairs"
            ],
            "decision": decision["decision"],
        },
    )


def run_pair_audit_stage(
    sorter_output: Path,
    pair_path: Path,
    sidecar_path: Path,
    output_dir: Path,
    sort_identity: dict[str, Any],
) -> dict[str, Any]:
    """Run the artifact-aware CCG audit for every current good-good pair."""
    output_dir = Path(output_dir)
    request = ensure_stage_request(
        output_dir / "pair_audit_request.json",
        stage="artifact_aware_similar_pair_audit",
        sort_identity=sort_identity,
        settings={
            "sorter_output": str(Path(sorter_output).resolve()),
            "pair_path": str(Path(pair_path).resolve()),
            "sidecar_path": str(Path(sidecar_path).resolve()),
        },
    )
    required = (
        output_dir / "summary.json",
        output_dir / "good_unit_artifact_proximity.csv",
        output_dir / "similar_good_pair_audit.csv",
    )
    receipt_path = output_dir / "pair_audit_receipt.json"
    if receipt := completed_stage_receipt(
        receipt_path, request=request, required_files=required
    ):
        return receipt
    from testing.luke_imec0_similar_pair_audit import run as run_pair_audit

    summary = run_pair_audit(
        Path(sorter_output), Path(pair_path), Path(sidecar_path), output_dir
    )
    return write_stage_receipt(
        receipt_path,
        request=request,
        required_files=required,
        summary=summary,
    )


def run_curation_stage(
    sorter_output: Path,
    output_dir: Path,
    sort_identity: dict[str, Any],
    *,
    cosine_threshold: float = 0.90,
    ccg_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run the established final curation into ``cur/cur_output``."""
    output_dir = Path(output_dir)
    request = ensure_stage_request(
        output_dir / "curation_request.json",
        stage="legacy_compatible_curation",
        sort_identity=sort_identity,
        settings={
            "sorter_output": str(Path(sorter_output).resolve()),
            "strategy": "run_cur_final_cosine",
            "cosine_threshold": cosine_threshold,
            "ccg_threshold": ccg_threshold,
            "automatic_artifact_pair_merging": False,
        },
    )
    curated = output_dir / "cur_output"
    required = (
        curated / "spike_times.npy",
        curated / "spike_clusters.npy",
        curated / "cluster_KSLabel.tsv",
        curated / "ops.npy",
    )
    receipt_path = output_dir / "curation_receipt.json"
    if receipt := completed_stage_receipt(
        receipt_path, request=request, required_files=required
    ):
        return receipt

    import spikeinterface.full as si
    from pipelineold import KilosortResults
    from pipelineold.curation_postpatch import run_cur_final

    source = Path(sorter_output)
    sorter = si.read_kilosort(folder_path=source)
    results = KilosortResults(source)
    run_cur_final(
        sorter,
        results,
        output_dir,
        recalc=False,
        cosine_thresh=cosine_threshold,
        ccg_thresh=ccg_threshold,
        ks4_out_path=source,
    )
    clusters = np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    labels = np.genfromtxt(
        curated / "cluster_KSLabel.tsv", delimiter="\t", names=True, dtype=None,
        encoding="utf-8"
    )
    label_name = next(name for name in labels.dtype.names if name != "cluster_id")
    good_count = int(
        sum(str(value).strip().lower() == "good" for value in labels[label_name])
    )
    return write_stage_receipt(
        receipt_path,
        request=request,
        required_files=required,
        summary={
            "spike_count": int(clusters.size),
            "unit_count": int(np.unique(clusters).size),
            "kilosort_good_unit_count": good_count,
            "curated_output": str(curated.resolve()),
        },
    )


def run_qc_stage(
    recording_dir: Path,
    curated_output: Path,
    output_dir: Path,
    sort_identity: dict[str, Any],
    *,
    waveform_seed: int = 0,
) -> dict[str, Any]:
    """Run restartable legacy-compatible waveform, presence, and RV QC."""
    output_dir = Path(output_dir)
    request = ensure_stage_request(
        output_dir / "qc_request.json",
        stage="legacy_compatible_qc",
        sort_identity=sort_identity,
        settings={
            "recording_dir": str(Path(recording_dir).resolve()),
            "curated_output": str(Path(curated_output).resolve()),
            "waveform_seed": waveform_seed,
            "waveform_extractor": "ordered_chunked_local_memmap_v1",
            "waveform_read_chunk_duration_s": 1.0,
            "waveforms_per_unit": 512,
            "waveform_samples": 82,
        },
    )
    required = (
        output_dir / "waveforms/waveforms.npz",
        output_dir / "refractory/refractory_qc.npz",
        output_dir / "refractory/refractory_qc.pdf",
        output_dir / "amp_truncation/truncation_qc.npz",
        output_dir / "amp_truncation/present_qc.npz",
        output_dir / "amp_truncation/truncation_qc.pdf",
    )
    receipt_path = output_dir / "qc_receipt.json"
    if receipt := completed_stage_receipt(
        receipt_path, request=request, required_files=required
    ):
        return receipt

    from spikeinterface.core import load
    from pipelineold import KilosortResults, run_qc

    np.random.seed(waveform_seed)
    recording = load(Path(recording_dir))
    results = KilosortResults(Path(curated_output))
    run_qc(recording, results, output_dir, recalc=False)
    return write_stage_receipt(
        receipt_path,
        request=request,
        required_files=required,
        summary={
            "unit_count": int(np.unique(results.spike_clusters).size),
            "waveform_seed": waveform_seed,
        },
    )


def run_matlab_export_stage(
    curated_output: Path,
    qc_dir: Path,
    sort_identity: dict[str, Any],
) -> dict[str, Any]:
    """Export the historical QC MAT files and probe coordinates atomically."""
    import scipy.io as sio

    qc_dir = Path(qc_dir)
    request = ensure_stage_request(
        qc_dir / "matlab_export_request.json",
        stage="legacy_compatible_matlab_export",
        sort_identity=sort_identity,
        settings={"curated_output": str(Path(curated_output).resolve())},
    )
    inputs = {
        "waveforms_data.mat": qc_dir / "waveforms/waveforms.npz",
        "refractory_data.mat": qc_dir / "refractory/refractory_qc.npz",
        "truncation_data.mat": qc_dir / "amp_truncation/truncation_qc.npz",
        "presence_data.mat": qc_dir / "amp_truncation/present_qc.npz",
    }
    outputs = [qc_dir / name for name in inputs]
    ops_mat = Path(curated_output) / "ops.mat"
    required = (*outputs, ops_mat)
    receipt_path = qc_dir / "matlab_export_receipt.json"
    if receipt := completed_stage_receipt(
        receipt_path, request=request, required_files=required
    ):
        return receipt
    for name, source in inputs.items():
        if not source.is_file():
            raise FileNotFoundError(f"QC output missing: {source}")
        target = qc_dir / name
        partial = target.with_suffix(target.suffix + ".partial")
        with np.load(source, allow_pickle=True) as data:
            sio.savemat(partial, {key: data[key] for key in data.files}, appendmat=False)
        os.replace(partial, target)
    ops = np.load(Path(curated_output) / "ops.npy", allow_pickle=True).item()
    partial_ops = ops_mat.with_suffix(".mat.partial")
    sio.savemat(
        partial_ops,
        {"xc": np.asarray(ops["xc"]), "yc": np.asarray(ops["yc"])},
        appendmat=False,
    )
    os.replace(partial_ops, ops_mat)
    return write_stage_receipt(
        receipt_path,
        request=request,
        required_files=required,
        summary={"mat_files": [str(path.resolve()) for path in required]},
    )


def _population_identity(sorter_output: Path) -> dict[str, Any]:
    sorter_output = Path(sorter_output)
    files = {}
    for name in ("spike_times.npy", "spike_clusters.npy", "cluster_KSLabel.tsv"):
        path = sorter_output / name
        if not path.is_file():
            raise FileNotFoundError(f"Curated comparison input missing: {path}")
        files[name] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    payload = {"sorter_output": str(sorter_output.resolve()), "files": files}
    payload["digest"] = fingerprint(payload)
    return payload


def run_postcuration_comparison_stage(
    curated_outputs: dict[str, Path],
    output_dir: Path,
    sort_identity: dict[str, Any],
    *,
    probe: str,
    duration_s: float,
    jitter_replicates: int = 250,
    seed: int = 20250804,
) -> dict[str, Any]:
    """Apply the matched evaluator to new, legacy, and claim-mask curation."""
    import pandas as pd
    from testing.luke_full_probe_rescue_diagnostics import run as run_diagnostics

    output_dir = Path(output_dir)
    population_identities = {
        name: _population_identity(path) for name, path in curated_outputs.items()
    }
    request = ensure_stage_request(
        output_dir / "comparison_request.json",
        stage="postcuration_population_comparison",
        sort_identity=sort_identity,
        settings={
            "populations": population_identities,
            "probe": probe,
            "duration_s": duration_s,
            "jitter_replicates": jitter_replicates,
            "seed": seed,
        },
    )
    comparison_json = output_dir / "postcuration_comparison.json"
    comparison_csv = output_dir / "postcuration_comparison.csv"
    required = (comparison_json, comparison_csv)
    receipt_path = output_dir / "comparison_receipt.json"
    if receipt := completed_stage_receipt(
        receipt_path, request=request, required_files=required
    ):
        return receipt

    rows = []
    for name, sorter in curated_outputs.items():
        method_dir = output_dir / name
        summary = run_diagnostics(
            Path(sorter),
            method_dir,
            jitter_replicates,
            seed,
            probe=probe,
            duration_override_s=duration_s,
        )
        units = pd.read_csv(method_dir / "unit_metrics.csv")
        good = units[units.ks_good.astype(bool)]
        rows.append(
            {
                "method": name,
                "sorter_output": str(Path(sorter).resolve()),
                "spike_count": int(summary["n_spikes"]),
                "unit_count": int(summary["n_units"]),
                "ks_good_count": int(summary["n_ks_good"]),
                "stable_good_count": int(summary["good_units_presence_ge_0_9"]),
                "stable_good_fraction": float(
                    summary["good_units_presence_ge_0_9"]
                    / max(summary["n_ks_good"], 1)
                ),
                "median_good_contamination_pct": float(
                    summary["median_contamination_pct_good"]
                ),
                "median_good_rate_hz": float(good.mean_rate_hz.median()),
                "good_units_gt_1hz": int((good.mean_rate_hz > 1).sum()),
                "good_units_gt_5hz": int((good.mean_rate_hz > 5).sum()),
                "good_units_gt_10hz": int((good.mean_rate_hz > 10).sum()),
                "similar_good_good_pairs": int(
                    summary["nearby_similar_good_good_pairs"]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(comparison_csv, index=False)
    payload = {
        "sort_identity_digest": sort_identity["identity_digest"],
        "stage_matched": True,
        "methods": rows,
        "interpretation_guardrail": (
            "Automatic post-curation metrics support comparison but do not replace "
            "manual waveform and artifact review."
        ),
    }
    _atomic_json(comparison_json, payload)
    return write_stage_receipt(
        receipt_path,
        request=request,
        required_files=required,
        summary={"methods": rows},
    )


def write_conservative_decision(
    output: Path,
    sort_identity: dict[str, Any],
    *,
    pair_audit_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the prespecified decision without any automatic waiver path."""
    decision = {
        "decision": "reject_universal_default",
        "promotion_allowed": False,
        "sort_identity_digest": sort_identity["identity_digest"],
        "pair_audit_complete": pair_audit_summary is not None,
        "pair_audit_summary": pair_audit_summary,
        "guardrail": (
            "The decision remains reject_universal_default until the similar-pair "
            "failure is resolved or conservatively discounted by reviewed evidence."
        ),
        "updated_at": _utc_now(),
    }
    _atomic_json(Path(output), decision)
    return decision
