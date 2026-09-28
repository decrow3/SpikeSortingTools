"""Run the decisive unfiltered native-250-Hz LFP correction control."""

from __future__ import annotations

import argparse
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
from pipeline.preprocess import validate_accepted_recording
from pipeline.sorting import run_kilosort4
from testing.luke_external_warp_pipeline import _materialize_arm
from testing.luke_full_session_medicine import save, sha
from testing.luke_lfp_ap_100s_matrix_analysis import analyze_arm, regime_metrics
from testing.luke_lfp_ap_100s_sort_matrix import (
    INTERPOLATION,
    INTERVAL_S,
    REFERENCE_S,
    _reference,
    corrected,
)
from testing.luke_lfp_savgol_100s_sort import load_evaluation_events
from testing.luke_medicine_rigid_queue import checked_int16


SCHEMA = "luke-lfp-native-100s-sort-v1"
ARM = "lfp_native"
EXPECTED_GRID_S = 0.004


def load_native_field(path: Path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    required = {"time_s", "displacement_um", "supported", "invalid"}
    if not required.issubset(z.files):
        raise RuntimeError("native LFP package is missing arrays")
    time_s = np.asarray(z["time_s"], float)
    raw = np.asarray(z["displacement_um"], float)
    supported = np.asarray(z["supported"], bool) & ~np.asarray(z["invalid"], bool) & np.isfinite(raw)
    dt = float(np.median(np.diff(time_s)))
    if not np.all(np.diff(time_s) > 0) or not np.isclose(dt, EXPECTED_GRID_S, rtol=0.01):
        raise RuntimeError("unexpected native LFP grid")
    target = (time_s >= INTERVAL_S[0]) & (time_s < INTERVAL_S[1])
    if target.sum() != 25_000:
        raise RuntimeError("unexpected native LFP interval extent")
    bad = np.flatnonzero(target & ~supported)
    groups = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1) if len(bad) else []
    max_gap = max((len(group) for group in groups), default=0)
    if max_gap > 10:
        raise RuntimeError("native LFP gap exceeds bounded 40 ms policy")
    filled = raw.copy()
    filled[~supported] = np.interp(time_s[~supported], time_s[supported], raw[supported])
    centered, offset = _reference(time_s, filled[:, None])
    step = np.diff(filled[target])
    field = {
        "time_s": time_s,
        "depth_um": np.array([0.0]),
        "displacement_um": centered,
    }
    audit = {
        "source_grid_s": dt,
        "source_rate_hz": 1.0 / dt,
        "filter": None,
        "interval_samples": int(target.sum()),
        "supported_interval_samples": int((target & supported).sum()),
        "gap_samples_filled": int(len(bad)),
        "gap_count": int(len(groups)),
        "maximum_gap_samples": int(max_gap),
        "gap_policy": "linear interpolation of bounded native-grid gaps no longer than 40 ms",
        "reference_window_s": list(REFERENCE_S),
        "reference_offset_um": float(offset[0]),
        "range_um": float(np.ptp(filled[target])),
        "maximum_4ms_step_um": float(np.max(np.abs(step))),
        "control_policy": "frozen unfiltered native-grid control for the SG25/order-2 arm",
    }
    return field, audit


def verify_config(cfg: dict) -> None:
    if cfg.get("schema") != SCHEMA:
        raise ValueError("wrong native LFP control schema")
    for key, hash_key in [
        ("recording_manifest", "recording_manifest_sha256"),
        ("lfp_native_motion", "lfp_native_motion_sha256"),
        ("lighthouse_events", "lighthouse_events_sha256"),
        ("lighthouse_families", "lighthouse_families_sha256"),
        ("lighthouse_regimes", "lighthouse_regimes_sha256"),
        ("baseline_contract", "baseline_contract_sha256"),
        ("baseline_summary", "baseline_summary_sha256"),
        ("baseline_arm_summary", "baseline_arm_summary_sha256"),
        ("baseline_regime_summary", "baseline_regime_summary_sha256"),
        ("savgol_contract", "savgol_contract_sha256"),
        ("savgol_summary", "savgol_summary_sha256"),
        ("savgol_arm_summary", "savgol_arm_summary_sha256"),
        ("savgol_regime_summary", "savgol_regime_summary_sha256"),
    ]:
        if sha(Path(cfg[key])) != cfg[hash_key]:
            raise RuntimeError(f"frozen input changed: {key}")


def plot_comparison(arms: pd.DataFrame, regimes: pd.DataFrame, output: Path) -> None:
    order = ["unwarped", "lfp_rigid", "lfp_native", "lfp_savgol", "ap_rigid", "ap_nonrigid"]
    labels = {
        "unwarped": "unwarped",
        "lfp_rigid": "LFP raw\n4 Hz field",
        "lfp_native": "LFP raw\n250 Hz field",
        "lfp_savgol": "LFP SG\n250 Hz field",
        "ap_rigid": "AP\nrigid",
        "ap_nonrigid": "AP\nnonrigid",
    }
    colors = {
        "unwarped": "#6B7280",
        "lfp_rigid": "#84A80B",
        "lfp_native": "#7C3AED",
        "lfp_savgol": "#0F9D8A",
        "ap_rigid": "#D97706",
        "ap_nonrigid": "#2563EB",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (regime, title) in zip(
        axes.flat[:3],
        [("all", "All held-out lighthouse events"), ("quiet", "Lighthouse-quiet events"), ("movement", "Lighthouse-movement events")],
    ):
        data = regimes.loc[regimes.regime.eq(regime)].set_index("arm").reindex(order)
        ax.bar(range(len(order)), data.family_macro_single_cluster, color=[colors[x] for x in order], edgecolor="#374151")
        ax.set_xticks(range(len(order)), [labels[x] for x in order])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    ax = axes.flat[3]
    data = arms.set_index("arm").reindex(order)
    ax.bar(range(len(order)), data.good_units, color=[colors[x] for x in order], edgecolor="#374151")
    ax.set_xticks(range(len(order)), [labels[x] for x in order])
    ax.set_title("KS-good units")
    ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Matched 930–1030 s correction-sort comparison\nDecisive native-250-Hz unfiltered control isolates Savitzky–Golay smoothing")
    fig.savefig(output / "01_native_lfp_filter_ablation.png", dpi=180)
    fig.savefig(output / "01_native_lfp_filter_ablation.pdf")
    plt.close(fig)


def analyze(cfg: dict, output: Path, arm_root: Path) -> dict:
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    events, family = load_evaluation_events(cfg)
    fields = np.load(output / "resolved_candidate_field.npz", allow_pickle=False)
    aggregate, units, matched = analyze_arm(ARM, arm_root, events, fields)
    native_regimes = regime_metrics(units, family)
    baseline = Path(cfg["baseline_matrix"]) / "analysis"
    savgol = Path(cfg["savgol_output"]) / "analysis"
    savgol_arm = pd.read_csv(savgol / "arm_summary_with_savgol.csv").loc[lambda x: x.arm.eq("lfp_savgol")]
    savgol_regimes = pd.read_csv(savgol / "regime_summary_with_savgol.csv").loc[lambda x: x.arm.eq("lfp_savgol")]
    arms = pd.concat([pd.read_csv(baseline / "arm_summary.csv"), savgol_arm, pd.DataFrame([aggregate])], ignore_index=True)
    regimes = pd.concat([pd.read_csv(baseline / "regime_summary.csv"), savgol_regimes, native_regimes], ignore_index=True)
    arms.to_csv(analysis / "arm_summary_native_filter_ablation.csv", index=False)
    regimes.to_csv(analysis / "regime_summary_native_filter_ablation.csv", index=False)
    units.to_csv(analysis / "lfp_native_lighthouse_unit_metrics.csv", index=False)
    matched.to_csv(analysis / "lfp_native_matched_events.csv", index=False)
    plot_comparison(arms, regimes, analysis)
    result = {
        "status": "complete",
        "primary_endpoint": "family-macro single-cluster fraction among frozen held-out strict lighthouse events",
        "arm": aggregate,
        "regimes": json.loads(native_regimes.to_json(orient="records")),
        "baseline_matrix": cfg["baseline_matrix"],
        "savgol_output": cfg["savgol_output"],
        "ablation": "lfp_native versus lfp_savgol differs only by SG25/order-2 filtering",
    }
    save(analysis / "summary.json", result)
    return result


def execute(cfg: dict, output: Path) -> None:
    from spikeinterface.core import load

    verify_config(cfg)
    source_dir = Path(cfg["recording"])
    source_manifest = validate_accepted_recording(source_dir)
    if source_manifest["recording_content_sha256"] != cfg["recording_content_sha256"]:
        raise RuntimeError("accepted recording content changed")
    source = load(source_dir)
    baseline_contract = json.loads(Path(cfg["baseline_contract"]).read_text())
    savgol_contract = json.loads(Path(cfg["savgol_contract"]).read_text())
    if baseline_contract["interval_s"] != list(INTERVAL_S) or savgol_contract["interval_s"] != list(INTERVAL_S):
        raise RuntimeError("comparison interval changed")
    if baseline_contract["reference_window_s"] != list(REFERENCE_S) or savgol_contract["reference_window_s"] != list(REFERENCE_S):
        raise RuntimeError("comparison reference changed")
    if baseline_contract["interpolation_parameters"] != INTERPOLATION or savgol_contract["interpolation_parameters"] != INTERPOLATION:
        raise RuntimeError("comparison interpolation changed")
    if savgol_contract["savgol_audit"]["window_length_samples"] != 25 or savgol_contract["savgol_audit"]["polyorder"] != 2:
        raise RuntimeError("SG comparison contract changed")
    source_ids = {str(value): value for value in source.get_channel_ids().tolist()}
    ids = [source_ids[value] for value in baseline_contract["common_channel_ids"]]
    if savgol_contract["common_channel_ids"] != baseline_contract["common_channel_ids"]:
        raise RuntimeError("SG and baseline channel contracts differ")
    field, audit = load_native_field(Path(cfg["lfp_native_motion"]))
    corrected_source = corrected(source, field)
    if not set(ids).issubset(set(corrected_source.get_channel_ids().tolist())):
        raise RuntimeError("native correction does not retain the baseline common channels")
    fs = float(source.get_sampling_frequency())
    start, stop = [int(round(value * fs)) for value in INTERVAL_S]
    view = checked_int16(corrected_source.frame_slice(start_frame=start, end_frame=stop).channel_slice(channel_ids=ids))
    contract = {
        "schema": SCHEMA,
        "development_only": True,
        "arm": ARM,
        "interval_s": list(INTERVAL_S),
        "reference_window_s": list(REFERENCE_S),
        "common_channels": len(ids),
        "common_channel_ids": [str(value) for value in ids],
        "dtype_policy": "float32 interpolation; round-nearest checked int16 materialization",
        "temporal_interpolation": "linear",
        "spatial_interpolation": "kriging",
        "interpolation_parameters": INTERPOLATION,
        "internal_kilosort_motion": False,
        "native_audit": audit,
        "baseline_contract_digest": baseline_contract["digest"],
        "savgol_contract_digest": savgol_contract["digest"],
        "checkpoint_policy": "No within-sort checkpoint; preserve failure evidence and do not restart automatically.",
    }
    contract["digest"] = fingerprint(contract)
    save(output / "contract.json", contract)
    save(output / "field_resolution_audit.json", audit)
    np.savez_compressed(output / "resolved_candidate_field.npz", lfp_native_time_s=field["time_s"], lfp_native_displacement_um=field["displacement_um"][:, 0])
    arm_root = output / "arm"
    arm_root.mkdir()
    expected_bytes = (stop - start) * len(ids) * 2
    if shutil.disk_usage(output).free < expected_bytes + int(cfg["sort_scratch_reserve_bytes"]):
        raise RuntimeError("insufficient disk")
    save(output / "status.json", {"stage": "materialize", "arm": ARM, "updated_unix": time.time(), "pid": os.getpid()})
    manifest = _materialize_arm(view, arm_root / "recording", source_manifest=source_manifest, request={"schema": SCHEMA, "contract_digest": contract["digest"], "arm": ARM}, n_jobs=int(cfg["materialize_jobs"]))
    if manifest["num_samples"] != stop - start or manifest["num_channels"] != len(ids):
        raise RuntimeError("materialized extent changed")
    save(output / "status.json", {"stage": "sort", "arm": ARM, "updated_unix": time.time(), "pid": os.getpid()})
    sort_manifest = run_kilosort4(arm_root / "recording", arm_root / "kilosort4")
    ops = np.load(arm_root / "kilosort4/sorter_output/ops.npy", allow_pickle=True).item()
    if int(ops["nblocks"]) != 0 or ops["dshift"] is not None:
        raise RuntimeError("internal Kilosort motion unexpectedly active")
    save(arm_root / "summary.json", {"status": "complete", "arm": ARM, "sort_summary": sort_manifest["summary"]})
    save(output / "status.json", {"stage": "analysis", "updated_unix": time.time(), "pid": os.getpid()})
    analysis = analyze(cfg, output, arm_root)
    save(output / "summary.json", {"status": "complete", "contract_digest": contract["digest"], "arm": sort_manifest["summary"], "analysis_status": analysis["status"], "scientific_status": "development_filter_ablation_complete_requires_review"})
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
