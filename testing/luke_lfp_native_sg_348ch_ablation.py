"""Run a matched 348-channel native-LFP versus SG25 filter ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import fingerprint
from pipeline.preprocess import RECORDING_MANIFEST_SCHEMA
from pipeline.sorting import run_kilosort4
from testing.luke_external_warp_pipeline import _materialize_arm
from testing.luke_full_session_medicine import save, sha
from testing.luke_lfp_ap_100s_matrix_analysis import analyze_arm, regime_metrics
from testing.luke_lfp_ap_100s_sort_matrix import INTERPOLATION, INTERVAL_S, REFERENCE_S, corrected
from testing.luke_lfp_native_100s_sort import load_native_field
from testing.luke_lfp_savgol_100s_sort import load_evaluation_events, load_savgol_field
from testing.luke_medicine_rigid_queue import checked_int16


SCHEMA = "luke-lfp-native-sg-348ch-ablation-v1"
ARMS = ("lfp_native", "lfp_savgol")


def verify_config(cfg: dict) -> None:
    if cfg.get("schema") != SCHEMA:
        raise ValueError("wrong native/SG ablation schema")
    for key, hash_key in [
        ("recording_manifest", "recording_manifest_sha256"),
        ("lfp_native_motion", "lfp_native_motion_sha256"),
        ("lighthouse_events", "lighthouse_events_sha256"),
        ("lighthouse_families", "lighthouse_families_sha256"),
        ("lighthouse_regimes", "lighthouse_regimes_sha256"),
        ("baseline_contract", "baseline_contract_sha256"),
        ("source_proof_receipt", "source_proof_receipt_sha256"),
        ("source_proof_config", "source_proof_config_sha256"),
        ("source_proof_stderr", "source_proof_stderr_sha256"),
        ("source_proof_worker", "source_proof_worker_sha256"),
    ]:
        if sha(Path(cfg[key])) != cfg[hash_key]:
            raise RuntimeError(f"frozen input changed: {key}")


def validate_recent_source_proof(cfg: dict, source_dir: Path) -> dict:
    """Reuse a just-completed full hash whose worker failed only at the channel gate."""
    manifest = json.loads(Path(cfg["recording_manifest"]).read_text())
    receipt = json.loads(Path(cfg["source_proof_receipt"]).read_text())
    proof_cfg = json.loads(Path(cfg["source_proof_config"]).read_text())
    stderr = Path(cfg["source_proof_stderr"]).read_text()
    if manifest.get("schema_version") != RECORDING_MANIFEST_SCHEMA or not manifest.get("complete"):
        raise RuntimeError("accepted recording manifest is incomplete")
    if receipt.get("state") != "failed" or receipt.get("returncode") != 1:
        raise RuntimeError("source validation proof has unexpected terminal state")
    if "native correction does not retain the baseline common channels" not in stderr:
        raise RuntimeError("source validation proof did not reach the expected post-hash gate")
    if proof_cfg.get("recording_content_sha256") != cfg["recording_content_sha256"]:
        raise RuntimeError("source validation proof used a different recording digest")
    if proof_cfg.get("recording_manifest_sha256") != cfg["recording_manifest_sha256"]:
        raise RuntimeError("source validation proof used a different manifest")
    finished = datetime.fromisoformat(receipt["finished_at"])
    age_s = (datetime.now(timezone.utc) - finished).total_seconds()
    if age_s < 0 or age_s > 6 * 3600:
        raise RuntimeError("source validation proof is not recent")
    binaries = manifest.get("recording_binary_files", [])
    if len(binaries) != 1:
        raise RuntimeError("expected one accepted recording binary")
    binary = source_dir / binaries[0]["name"]
    stat = binary.stat()
    if stat.st_size != binaries[0]["size_bytes"] or stat.st_size != manifest["expected_binary_bytes"]:
        raise RuntimeError("accepted recording size changed")
    if stat.st_mtime_ns != int(cfg["recording_binary_mtime_ns"]):
        raise RuntimeError("accepted recording mtime changed after the full-hash proof")
    return {
        "policy": "reuse recent complete source hash from failed v1; failure occurred at the later common-channel gate",
        "proof_receipt": cfg["source_proof_receipt"],
        "proof_finished_at": receipt["finished_at"],
        "proof_age_s_at_validation": age_s,
        "recording_content_sha256": cfg["recording_content_sha256"],
        "recording_binary_size_bytes": stat.st_size,
        "recording_binary_mtime_ns": stat.st_mtime_ns,
        "manifest_sha256": cfg["recording_manifest_sha256"],
        "proof_worker_sha256": cfg["source_proof_worker_sha256"],
    }


def plot_results(arms: pd.DataFrame, regimes: pd.DataFrame, output: Path) -> None:
    order = list(ARMS)
    labels = {"lfp_native": "LFP raw\n250 Hz", "lfp_savgol": "LFP SG25\n250 Hz"}
    colors = {"lfp_native": "#7C3AED", "lfp_savgol": "#0F9D8A"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (regime, title) in zip(
        axes.flat[:3],
        [("all", "All held-out lighthouse events"), ("quiet", "Lighthouse-quiet events"), ("movement", "Lighthouse-movement events")],
    ):
        data = regimes.loc[regimes.regime.eq(regime)].set_index("arm").reindex(order)
        ax.bar(range(2), data.family_macro_single_cluster, color=[colors[x] for x in order], edgecolor="#374151")
        ax.set_xticks(range(2), [labels[x] for x in order])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    ax = axes.flat[3]
    data = arms.set_index("arm").reindex(order)
    ax.bar(range(2), data.good_units, color=[colors[x] for x in order], edgecolor="#374151")
    ax.set_xticks(range(2), [labels[x] for x in order])
    ax.set_title("KS-good units")
    ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Decisive native-250-Hz LFP filter ablation\nMatched 930–1030 s voltage and 348-channel support")
    fig.savefig(output / "01_native_lfp_filter_ablation.png", dpi=180)
    fig.savefig(output / "01_native_lfp_filter_ablation.pdf")
    plt.close(fig)


def analyze(cfg: dict, output: Path, arm_roots: dict[str, Path]) -> dict:
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    events, family = load_evaluation_events(cfg)
    fields = np.load(output / "resolved_candidate_fields.npz", allow_pickle=False)
    aggregates, units, matches = [], [], []
    for arm in ARMS:
        aggregate, unit, matched = analyze_arm(arm, arm_roots[arm], events, fields)
        aggregates.append(aggregate)
        units.append(unit)
        matches.append(matched)
    arm_table = pd.DataFrame(aggregates)
    unit_table = pd.concat(units, ignore_index=True)
    regime_table = regime_metrics(unit_table, family)
    native_arm = arm_table.set_index("arm").loc["lfp_native"]
    native_regime = regime_table.loc[regime_table.arm.eq("lfp_native")].set_index("regime")
    for column in ["macro_lighthouse_recovery", "macro_single_cluster_all_events", "macro_duplicate_event_fraction", "median_contamination_pct", "median_refractory_fraction", "good_units"]:
        arm_table[f"delta_sg_minus_native_{column}"] = np.where(
            arm_table.arm.eq("lfp_savgol"), arm_table[column] - native_arm[column], np.nan
        )
    for column in ["family_macro_recovery", "family_macro_single_cluster", "family_macro_duplicate_fraction", "median_family_fragment_clusters"]:
        regime_table[f"delta_sg_minus_native_{column}"] = np.where(
            regime_table.arm.eq("lfp_savgol"),
            regime_table[column] - regime_table.regime.map(native_regime[column]),
            np.nan,
        )
    arm_table.to_csv(analysis / "arm_summary.csv", index=False)
    regime_table.to_csv(analysis / "regime_summary.csv", index=False)
    unit_table.to_csv(analysis / "lighthouse_unit_metrics.csv", index=False)
    pd.concat(matches, ignore_index=True).to_csv(analysis / "matched_events.csv", index=False)
    plot_results(arm_table, regime_table, analysis)
    result = {
        "status": "complete",
        "primary_endpoint": "family-macro single-cluster fraction among frozen held-out strict lighthouse events",
        "arms": json.loads(arm_table.to_json(orient="records")),
        "regimes": json.loads(regime_table.to_json(orient="records")),
        "ablation": "same native field, support, reference, interpolation, voltage, channels, and sorter; SG25/order-2 filtering is the only arm difference",
    }
    save(analysis / "summary.json", result)
    return result


def execute(cfg: dict, output: Path) -> None:
    from spikeinterface.core import load

    verify_config(cfg)
    source_dir = Path(cfg["recording"])
    source_proof = validate_recent_source_proof(cfg, source_dir)
    source_manifest = json.loads(Path(cfg["recording_manifest"]).read_text())
    source = load(source_dir)
    baseline_contract = json.loads(Path(cfg["baseline_contract"]).read_text())
    if baseline_contract["interval_s"] != list(INTERVAL_S) or baseline_contract["reference_window_s"] != list(REFERENCE_S):
        raise RuntimeError("baseline time contract changed")
    if baseline_contract["interpolation_parameters"] != INTERPOLATION:
        raise RuntimeError("baseline interpolation contract changed")
    native, native_audit = load_native_field(Path(cfg["lfp_native_motion"]))
    savgol, savgol_audit = load_savgol_field(Path(cfg["lfp_native_motion"]))
    fields = {"lfp_native": native, "lfp_savgol": savgol}
    source_ids = {str(value): value for value in source.get_channel_ids().tolist()}
    baseline_ids = [source_ids[value] for value in baseline_contract["common_channel_ids"]]
    retained = {arm: set(corrected(source, field).get_channel_ids().tolist()) for arm, field in fields.items()}
    ids = [value for value in baseline_ids if all(value in retained[arm] for arm in ARMS)]
    removed = [str(value) for value in baseline_ids if value not in ids]
    if len(ids) != 348 or removed != ["imec0.ap#AP24", "imec0.ap#AP25"]:
        raise RuntimeError("unexpected native/SG common-channel intersection")
    fs = float(source.get_sampling_frequency())
    start, stop = [int(round(value * fs)) for value in INTERVAL_S]
    contract = {
        "schema": SCHEMA,
        "development_only": True,
        "arms": list(ARMS),
        "interval_s": list(INTERVAL_S),
        "reference_window_s": list(REFERENCE_S),
        "common_channels": len(ids),
        "common_channel_ids": [str(value) for value in ids],
        "removed_from_prior_350_channel_contract": removed,
        "dtype_policy": "float32 interpolation; round-nearest checked int16 materialization",
        "temporal_interpolation": "linear",
        "spatial_interpolation": "kriging",
        "interpolation_parameters": INTERPOLATION,
        "internal_kilosort_motion": False,
        "native_audit": native_audit,
        "savgol_audit": savgol_audit,
        "source_validation": source_proof,
        "checkpoint_policy": "No within-sort checkpoint; preserve failure evidence and do not restart automatically.",
    }
    contract["digest"] = fingerprint(contract)
    save(output / "contract.json", contract)
    save(output / "field_resolution_audit.json", {"native": native_audit, "savgol": savgol_audit})
    np.savez_compressed(
        output / "resolved_candidate_fields.npz",
        lfp_native_time_s=native["time_s"],
        lfp_native_displacement_um=native["displacement_um"][:, 0],
        lfp_savgol_time_s=savgol["time_s"],
        lfp_savgol_displacement_um=savgol["displacement_um"][:, 0],
    )
    summaries, arm_roots = {}, {}
    for arm in ARMS:
        arm_root = output / "arms" / arm
        arm_root.mkdir(parents=True)
        arm_roots[arm] = arm_root
        view = checked_int16(
            corrected(source, fields[arm]).frame_slice(start_frame=start, end_frame=stop).channel_slice(channel_ids=ids)
        )
        expected_bytes = (stop - start) * len(ids) * 2
        if shutil.disk_usage(output).free < expected_bytes + int(cfg["sort_scratch_reserve_bytes"]):
            raise RuntimeError(f"insufficient disk before {arm}")
        save(output / "status.json", {"stage": "materialize", "arm": arm, "updated_unix": time.time(), "pid": os.getpid()})
        manifest = _materialize_arm(view, arm_root / "recording", source_manifest=source_manifest, request={"schema": SCHEMA, "contract_digest": contract["digest"], "arm": arm}, n_jobs=int(cfg["materialize_jobs"]))
        if manifest["num_samples"] != stop - start or manifest["num_channels"] != len(ids):
            raise RuntimeError(f"materialized extent changed for {arm}")
        save(output / "status.json", {"stage": "sort", "arm": arm, "updated_unix": time.time(), "pid": os.getpid()})
        sort_manifest = run_kilosort4(arm_root / "recording", arm_root / "kilosort4")
        ops = np.load(arm_root / "kilosort4/sorter_output/ops.npy", allow_pickle=True).item()
        if int(ops["nblocks"]) != 0 or ops["dshift"] is not None:
            raise RuntimeError(f"internal Kilosort motion unexpectedly active for {arm}")
        summaries[arm] = sort_manifest["summary"]
        save(arm_root / "summary.json", {"status": "complete", "arm": arm, "sort_summary": sort_manifest["summary"]})
    save(output / "status.json", {"stage": "analysis", "updated_unix": time.time(), "pid": os.getpid()})
    analysis = analyze(cfg, output, arm_roots)
    save(output / "summary.json", {"status": "complete", "contract_digest": contract["digest"], "arms": summaries, "analysis_status": analysis["status"], "scientific_status": "development_filter_ablation_complete_requires_review"})
    save(output / "status.json", {"stage": "complete", "updated_unix": time.time(), "pid": os.getpid()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "request.json", cfg)
    if args.dummy:
        save(output / "status.json", {"stage": "dummy_running", "pid": os.getpid()})
        time.sleep(20)
        save(output / "summary.json", {"status": "dummy_complete"})
        return
    execute(cfg, output)


if __name__ == "__main__":
    main()
