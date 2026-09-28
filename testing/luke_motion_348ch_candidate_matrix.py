"""Run unwarped, AP-rigid, and DARTsort-native additions on frozen 348-channel support."""

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
from pipeline.sorting import run_kilosort4
from testing.luke_external_warp_pipeline import _materialize_arm
from testing.luke_full_session_medicine import save, sha
from testing.luke_lfp_ap_100s_matrix_analysis import analyze_arm, regime_metrics
from testing.luke_lfp_ap_100s_sort_matrix import INTERPOLATION, INTERVAL_S, REFERENCE_S, corrected
from testing.luke_lfp_savgol_100s_sort import load_evaluation_events
from testing.luke_medicine_rigid_queue import checked_int16


SCHEMA = "luke-motion-348ch-candidate-matrix-v1"
NEW_ARMS = ("unwarped", "ap_rigid", "dartsort_native")
ALL_ARMS = ("unwarped", "ap_rigid", "dartsort_native", "lfp_native", "lfp_savgol")


def verify_config(cfg: dict) -> None:
    if cfg.get("schema") != SCHEMA:
        raise ValueError("wrong comparison schema")
    for key, hash_key in [
        ("recording_manifest", "recording_manifest_sha256"),
        ("ap_resolved_fields", "ap_resolved_fields_sha256"),
        ("dartsort_motion_export", "dartsort_motion_export_sha256"),
        ("dartsort_motion_audit", "dartsort_motion_audit_sha256"),
        ("dartsort_motion_pickle", "dartsort_motion_pickle_sha256"),
        ("dartsort_request", "dartsort_request_sha256"),
        ("dartsort_result", "dartsort_result_sha256"),
        ("dartsort_receipt", "dartsort_receipt_sha256"),
        ("lfp_348_contract", "lfp_348_contract_sha256"),
        ("lfp_resolved_fields", "lfp_resolved_fields_sha256"),
        ("lighthouse_events", "lighthouse_events_sha256"),
        ("lighthouse_families", "lighthouse_families_sha256"),
        ("lighthouse_regimes", "lighthouse_regimes_sha256"),
    ]:
        if sha(Path(cfg[key])) != cfg[hash_key]:
            raise RuntimeError(f"frozen input changed: {key}")


def validate_source_proof(cfg: dict, source_dir: Path) -> dict:
    receipt = json.loads(Path(cfg["source_proof_receipt"]).read_text())
    prior_cfg = json.loads(Path(cfg["source_proof_config"]).read_text())
    stderr = Path(cfg["source_proof_stderr"]).read_text()
    manifest = json.loads(Path(cfg["recording_manifest"]).read_text())
    if receipt.get("state") != "failed" or receipt.get("returncode") != 1:
        raise RuntimeError("source proof is not the preserved post-hash failed attempt")
    if "native correction does not retain the baseline common channels" not in stderr:
        raise RuntimeError("source proof failure did not occur at the later channel gate")
    if prior_cfg.get("recording_content_sha256") != cfg["recording_content_sha256"]:
        raise RuntimeError("source proof used a different recording hash")
    if sha(Path(cfg["source_proof_receipt"])) != cfg["source_proof_receipt_sha256"]:
        raise RuntimeError("source proof receipt changed")
    if sha(Path(cfg["source_proof_config"])) != cfg["source_proof_config_sha256"]:
        raise RuntimeError("source proof config changed")
    if sha(Path(cfg["source_proof_stderr"])) != cfg["source_proof_stderr_sha256"]:
        raise RuntimeError("source proof stderr changed")
    if sha(Path(cfg["source_proof_worker"])) != cfg["source_proof_worker_sha256"]:
        raise RuntimeError("source proof worker changed")
    finished = datetime.fromisoformat(receipt["finished_at"])
    age_s = (datetime.now(timezone.utc) - finished).total_seconds()
    if age_s < 0 or age_s > 6 * 3600:
        raise RuntimeError("source proof is not recent")
    binaries = manifest.get("recording_binary_files", [])
    if len(binaries) != 1:
        raise RuntimeError("expected one accepted recording binary")
    binary = source_dir / binaries[0]["name"]
    stat = binary.stat()
    if stat.st_size != binaries[0]["size_bytes"] or stat.st_size != manifest["expected_binary_bytes"]:
        raise RuntimeError("accepted recording size changed")
    if stat.st_mtime_ns != int(cfg["recording_binary_mtime_ns"]):
        raise RuntimeError("accepted recording mtime changed")
    return {
        "policy": "reuse recent complete 241 GB source hash from preserved post-hash gate failure",
        "proof_finished_at": receipt["finished_at"],
        "proof_age_s_at_validation": age_s,
        "recording_content_sha256": cfg["recording_content_sha256"],
        "binary_size_bytes": stat.st_size,
        "binary_mtime_ns": stat.st_mtime_ns,
    }


def load_fields(cfg: dict) -> tuple[dict, dict]:
    ap = np.load(cfg["ap_resolved_fields"], allow_pickle=False)
    dart = np.load(cfg["dartsort_motion_export"], allow_pickle=False)
    fields = {
        "ap_rigid": {
            "time_s": np.asarray(ap["ap_time_s"], float),
            "depth_um": np.array([0.0]),
            "displacement_um": np.asarray(ap["ap_rigid_displacement_um"], float)[:, None],
        },
        "dartsort_native": {
            "time_s": np.asarray(dart["time_s"], float),
            "depth_um": np.asarray(dart["depth_um"], float),
            "displacement_um": np.asarray(dart["displacement_um"], float),
        },
    }
    for name, field in fields.items():
        if field["displacement_um"].shape != (len(field["time_s"]), len(field["depth_um"])):
            raise RuntimeError(f"field shape mismatch: {name}")
        if not np.all(np.diff(field["time_s"]) > 0) or not np.all(np.diff(field["depth_um"]) > 0):
            raise RuntimeError(f"field axes are not increasing: {name}")
        if not np.all(np.isfinite(field["displacement_um"])):
            raise RuntimeError(f"field contains nonfinite values: {name}")
    audit = json.loads(Path(cfg["dartsort_motion_audit"]).read_text())
    return fields, audit


def plot_results(arms: pd.DataFrame, regimes: pd.DataFrame, output: Path) -> None:
    labels = {
        "unwarped": "Unwarped",
        "ap_rigid": "AP rigid",
        "dartsort_native": "DARTsort\nnative",
        "lfp_native": "LFP raw\n250 Hz",
        "lfp_savgol": "LFP SG25\n250 Hz",
    }
    colors = {
        "unwarped": "#6B7280",
        "ap_rigid": "#D97706",
        "dartsort_native": "#2563EB",
        "lfp_native": "#7C3AED",
        "lfp_savgol": "#0F9D8A",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    panels = [("all", "All held-out lighthouse events"), ("quiet", "Lighthouse-quiet events"), ("movement", "Lighthouse-movement events")]
    for ax, (regime, title) in zip(axes.flat[:3], panels):
        data = regimes.loc[regimes.regime.eq(regime)].set_index("arm").reindex(ALL_ARMS)
        ax.bar(range(len(ALL_ARMS)), data.family_macro_single_cluster, color=[colors[x] for x in ALL_ARMS], edgecolor="#374151")
        ax.set_xticks(range(len(ALL_ARMS)), [labels[x] for x in ALL_ARMS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    data = arms.set_index("arm").reindex(ALL_ARMS)
    axes.flat[3].bar(range(len(ALL_ARMS)), data.good_units, color=[colors[x] for x in ALL_ARMS], edgecolor="#374151")
    axes.flat[3].set_xticks(range(len(ALL_ARMS)), [labels[x] for x in ALL_ARMS])
    axes.flat[3].set_title("KS-good units")
    axes.flat[3].grid(axis="y", alpha=0.15)
    fig.suptitle("Matched 930–1030 s motion-correction comparison\nIdentical AP26–AP373 voltage and KS4 settings")
    fig.savefig(output / "01_motion_348ch_candidate_matrix.png", dpi=180)
    fig.savefig(output / "01_motion_348ch_candidate_matrix.pdf")
    plt.close(fig)


def analyze(cfg: dict, output: Path, new_roots: dict[str, Path]) -> dict:
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    events, family = load_evaluation_events(cfg)
    fields = np.load(output / "resolved_candidate_fields.npz", allow_pickle=False)
    lfp_root = Path(cfg["lfp_348_output"])
    roots = {**new_roots, "lfp_native": lfp_root / "arms/lfp_native", "lfp_savgol": lfp_root / "arms/lfp_savgol"}
    aggregates, units, matches = [], [], []
    for arm in ALL_ARMS:
        aggregate, unit, matched = analyze_arm(arm, roots[arm], events, fields)
        aggregates.append(aggregate)
        units.append(unit)
        matches.append(matched)
    arm_table = pd.DataFrame(aggregates)
    unit_table = pd.concat(units, ignore_index=True)
    regime_table = regime_metrics(unit_table, family)
    baseline_arm = arm_table.set_index("arm").loc["unwarped"]
    baseline_regime = regime_table.loc[regime_table.arm.eq("unwarped")].set_index("regime")
    for column in ["macro_lighthouse_recovery", "macro_single_cluster_all_events", "macro_duplicate_event_fraction", "median_contamination_pct", "median_refractory_fraction", "good_units"]:
        arm_table[f"delta_vs_unwarped_{column}"] = arm_table[column] - baseline_arm[column]
    for column in ["family_macro_recovery", "family_macro_single_cluster", "family_macro_duplicate_fraction", "median_family_fragment_clusters"]:
        regime_table[f"delta_vs_unwarped_{column}"] = regime_table[column] - regime_table.regime.map(baseline_regime[column])
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
        "comparison": "identical 930-1030 s source voltage, AP26-AP373 support, interpolation policy, reference window, and KS4 settings",
    }
    save(analysis / "summary.json", result)
    return result


def execute(cfg: dict, output: Path) -> None:
    from spikeinterface.core import load

    verify_config(cfg)
    source_dir = Path(cfg["recording"])
    source_proof = validate_source_proof(cfg, source_dir)
    source_manifest = json.loads(Path(cfg["recording_manifest"]).read_text())
    source = load(source_dir)
    lfp_contract = json.loads(Path(cfg["lfp_348_contract"]).read_text())
    if lfp_contract["interval_s"] != list(INTERVAL_S) or lfp_contract["reference_window_s"] != list(REFERENCE_S):
        raise RuntimeError("348-channel time contract changed")
    if lfp_contract["interpolation_parameters"] != INTERPOLATION:
        raise RuntimeError("348-channel interpolation contract changed")
    ids_by_name = {str(value): value for value in source.get_channel_ids().tolist()}
    id_names = lfp_contract["common_channel_ids"]
    if len(id_names) != 348 or id_names != [f"imec0.ap#AP{i}" for i in range(26, 374)]:
        raise RuntimeError("unexpected frozen 348-channel contract")
    ids = [ids_by_name[name] for name in id_names]
    fields, dart_audit = load_fields(cfg)
    dart = np.load(cfg["dartsort_motion_export"], allow_pickle=False)
    if not np.allclose(dart["geom"], source.get_channel_locations()):
        raise RuntimeError("DARTsort motion geometry differs from accepted source")
    for name, field in fields.items():
        retained = set(corrected(source, field).get_channel_ids().tolist())
        missing = [str(value) for value in ids if value not in retained]
        if missing:
            raise RuntimeError(f"{name} does not retain frozen channels: {missing}")
    fs = float(source.get_sampling_frequency())
    start, stop = [int(round(value * fs)) for value in INTERVAL_S]
    contract = {
        "schema": SCHEMA,
        "development_only": True,
        "arms_sorted_here": list(NEW_ARMS),
        "arms_compared": list(ALL_ARMS),
        "interval_s": list(INTERVAL_S),
        "reference_window_s": list(REFERENCE_S),
        "common_channels": len(ids),
        "common_channel_ids": id_names,
        "dtype_policy": "float32 interpolation; round-nearest checked int16 materialization",
        "temporal_interpolation": "linear",
        "spatial_interpolation": "kriging",
        "interpolation_parameters": INTERPOLATION,
        "internal_kilosort_motion": False,
        "dartsort_motion": dart_audit,
        "source_validation": source_proof,
        "checkpoint_policy": "No within-sort checkpoint; preserve failure evidence and do not restart automatically.",
    }
    contract["digest"] = fingerprint(contract)
    save(output / "contract.json", contract)
    prior_lfp = np.load(cfg["lfp_resolved_fields"], allow_pickle=False)
    np.savez_compressed(
        output / "resolved_candidate_fields.npz",
        ap_time_s=fields["ap_rigid"]["time_s"],
        ap_rigid_displacement_um=fields["ap_rigid"]["displacement_um"][:, 0],
        dartsort_time_s=fields["dartsort_native"]["time_s"],
        dartsort_depth_um=fields["dartsort_native"]["depth_um"],
        dartsort_displacement_um=fields["dartsort_native"]["displacement_um"],
        lfp_native_time_s=prior_lfp["lfp_native_time_s"],
        lfp_native_displacement_um=prior_lfp["lfp_native_displacement_um"],
        lfp_savgol_time_s=prior_lfp["lfp_savgol_time_s"],
        lfp_savgol_displacement_um=prior_lfp["lfp_savgol_displacement_um"],
    )
    summaries, roots = {}, {}
    for arm in NEW_ARMS:
        root = output / "arms" / arm
        root.mkdir(parents=True)
        roots[arm] = root
        base = source if arm == "unwarped" else corrected(source, fields[arm])
        view = base.frame_slice(start_frame=start, end_frame=stop).channel_slice(channel_ids=ids)
        if arm != "unwarped":
            view = checked_int16(view)
        expected_bytes = (stop - start) * len(ids) * 2
        if shutil.disk_usage(output).free < expected_bytes + int(cfg["sort_scratch_reserve_bytes"]):
            raise RuntimeError(f"insufficient disk before {arm}")
        save(output / "status.json", {"stage": "materialize", "arm": arm, "updated_unix": time.time(), "pid": os.getpid()})
        manifest = _materialize_arm(view, root / "recording", source_manifest=source_manifest, request={"schema": SCHEMA, "contract_digest": contract["digest"], "arm": arm}, n_jobs=int(cfg["materialize_jobs"]))
        if manifest["num_samples"] != stop - start or manifest["num_channels"] != len(ids):
            raise RuntimeError(f"materialized extent changed for {arm}")
        save(output / "status.json", {"stage": "sort", "arm": arm, "updated_unix": time.time(), "pid": os.getpid()})
        sort_manifest = run_kilosort4(root / "recording", root / "kilosort4")
        ops = np.load(root / "kilosort4/sorter_output/ops.npy", allow_pickle=True).item()
        if int(ops["nblocks"]) != 0 or ops["dshift"] is not None:
            raise RuntimeError(f"internal Kilosort motion unexpectedly active for {arm}")
        summaries[arm] = sort_manifest["summary"]
        save(root / "summary.json", {"status": "complete", "arm": arm, "sort_summary": sort_manifest["summary"]})
    save(output / "status.json", {"stage": "analysis", "updated_unix": time.time(), "pid": os.getpid()})
    analysis = analyze(cfg, output, roots)
    save(output / "summary.json", {"status": "complete", "contract_digest": contract["digest"], "arms": summaries, "analysis_status": analysis["status"], "scientific_status": "development_comparison_complete_requires_review"})
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
