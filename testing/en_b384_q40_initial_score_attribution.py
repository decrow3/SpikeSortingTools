#!/usr/bin/env python3
"""Fixed initial matched-filter attribution on retained B384 q+40 snippets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import conv1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kilosort.io import load_ops
from npx_preprocessing.motion.lattice_remap_si import sample_stepwise_shifts
from testing.en_rounded_field import rounded_rigid_field


SCHEMA = "en-b384-q40-initial-score-attribution-v1"
CAPTURE = ROOT / "testing/outputs/en_b384_q40_waveform_capture_20261001_v2"
PREP = ROOT / "testing/outputs/en_b384_q40_snippet_preparation_20261001_v7"
PREFLIGHT = ROOT / "testing/outputs/en_b384_q40_initial_score_attribution_preflight_20261001_v1"
OUTPUT = ROOT / "testing/outputs/en_b384_q40_initial_score_attribution_20261001_v1"
BANK = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/B384/pre_extraction_snapshot/Wall3.npy")
WPCA = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/B384/pre_extraction_snapshot/wPCA.npy")
OPS = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/B384/sorter_output/ops.npy")
FIELD = Path("/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/estimation_huklaban1_v1/candidate_fields.npz")
MATCHER = ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/template_matching.py"
FS = 29999.835983263598
BOUNDARY = np.arange(380, 384, dtype=np.int64)
EXPECTED = {
    CAPTURE / "SNIPPETS.npz": "fa2efaabc4ef6a839e195e64c022b7f74cc385577a30c21e8e273672dd1ebe2b",
    CAPTURE / "WINDOWS.csv": "ffd3b8176337874c332ac15179cc6e53c188bceabda9c517f9595188f757a9e5",
    PREP / "SELECTED_PAIRS.csv": "d6dd0941aa2775613d5684586be0435b7d7c927f018ac3db338b4ff063cfd278",
    PREP / "SELECTED_ISOLATED_CONTROLS.csv": "28532b7bac6ddf18eaba735ed7a9f4055f90c7d7d19d5163bef4ebce89a21929",
    BANK: "3cfa49a4bf9b8d4c0c46709749be9d9f005e7584091ca08785311b090533a608",
    WPCA: "f7a69044b0551370cd0520e007fea6119315bcb02347d5403683036d325847d5",
    OPS: "7a8f5d219de1b98776134abb00f637777f85ba94113758ae47a1e4b26b2a44f0",
    FIELD: "547ff39d1c91d819b758246c701603d868bfd4d50bd6b791e3d5b7140c01c71d",
    MATCHER: "39235cc98428dbb279706718f74a1c3d84568ef31a4f9d1d997a60f7002b1a66",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_inputs() -> None:
    for path, expected in EXPECTED.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen input differs: {path}: {observed}")


def fixtures() -> dict[str, float | str]:
    # Installed conv1d is cross-correlation. At one center it must equal the
    # explicit reconstructed-template dot product used by the attribution.
    x = torch.arange(44, dtype=torch.float64).reshape(4, 11) / 13 - 1.1
    w = torch.tensor([[0.3, -0.2, 0.7, 0.1, -0.4],
                      [-0.6, 0.8, 0.2, -0.5, 0.9]], dtype=torch.float64)
    u = torch.tensor([[0.2, -0.7, 0.4, 0.9],
                      [0.8, 0.1, -0.3, 0.5]], dtype=torch.float64)
    center = 6
    waveform = torch.einsum("pc,pt->ct", u, w)
    explicit = torch.sum(waveform * x[:, center-2:center+3])
    temporal = conv1d(x.unsqueeze(1), w.unsqueeze(1), padding=2)
    installed = torch.einsum("pc,cpl->l", u, temporal)[center]
    conv_error = float(torch.abs(explicit - installed))
    if conv_error > 1e-12:
        raise RuntimeError("projection formula differs from installed conv1d/einsum construction")

    # Nonsymmetric whitening makes the column selection in Wrot[:, S]
    # load-bearing; selecting rows after whitening would be a different test.
    matrix = torch.tensor([[1.0, 2.0, -1.0], [3.0, -4.0, 0.5], [2.5, 0.2, 1.7]], dtype=torch.float64)
    pre = torch.tensor([[2.0, -1.0], [0.5, 3.0], [-2.0, 4.0]], dtype=torch.float64)
    template = torch.tensor([[0.4, -0.1], [1.2, 0.3], [-0.7, 2.0]], dtype=torch.float64)
    subset = torch.tensor([0, 2])
    complement = torch.tensor([1])
    full = torch.sum(template * (matrix @ pre))
    part = torch.sum(template * (matrix[:, subset] @ pre[subset]))
    rest = torch.sum(template * (matrix[:, complement] @ pre[complement]))
    closure_error = float(torch.abs(full - part - rest))
    if closure_error > 1e-12:
        raise RuntimeError("nonsymmetric whitening decomposition fixture failed")
    return {
        "installed_conv1d_einsum_projection": "pass",
        "installed_projection_max_abs_error": conv_error,
        "nonsymmetric_whitening_column_decomposition": "pass",
        "nonsymmetric_linear_closure_abs_error": closure_error,
    }


def load_metadata() -> tuple[np.lib.npyio.NpzFile, pd.DataFrame, np.ndarray, np.ndarray, float]:
    snippets = np.load(CAPTURE / "SNIPPETS.npz", allow_pickle=False)
    windows = pd.read_csv(CAPTURE / "WINDOWS.csv")
    with np.load(FIELD, allow_pickle=False) as field:
        field_time = np.asarray(field["time_s"], dtype=np.float64)
        state, _ = rounded_rigid_field(field["rigid_displacement_um"], 40.0)
    dt = float(np.median(np.diff(field_time)))
    if not np.allclose(np.diff(field_time), dt, rtol=0, atol=1e-6):
        raise RuntimeError("frozen field time grid differs")
    return snippets, windows, field_time, state, dt


def select_endpoints() -> tuple[pd.DataFrame, pd.DataFrame]:
    snippets, windows, field_time, state, dt = load_metadata()
    raw = snippets["raw"]
    axes = snippets["global_samples"]
    selected, excluded = [], []
    for index, window in windows.iterrows():
        observed_states = sample_stepwise_shifts(field_time, state, axes[index] / FS, cell_width_s=dt)
        stable_q40 = bool(np.all(observed_states == 40.0))
        raw_boundary_zero = bool(np.all(raw[index, BOUNDARY] == 0))
        endpoints = [("first", int(window.first_time_global), int(window.first_template))]
        if window.kind == "pair":
            endpoints.append(("second", int(window.second_time_global), int(window.second_template)))
        for endpoint, event, template_id in endpoints:
            center_global = event + 10
            matches = np.flatnonzero(axes[index] == center_global)
            center_index = int(matches[0]) if len(matches) == 1 else -1
            support_fits = center_index >= 30 and center_index + 30 < axes.shape[1]
            row = {
                "window_index": int(index), "window_id": window.window_id,
                "kind": window.kind, "endpoint": endpoint, "unit_id": int(window.unit_id),
                "selection_role": window.selection_role, "event_global": event,
                "matcher_center_global": center_global, "center_index": center_index,
                "template_id": template_id, "stable_q40_complete_window": stable_q40,
                "raw_rows_380_383_zero_complete_window": raw_boundary_zero,
                "full_61_sample_support_fits": support_fits,
            }
            reasons = []
            if not stable_q40:
                reasons.append("complete_window_not_stable_q40")
            if not raw_boundary_zero:
                reasons.append("raw_boundary_not_all_zero")
            if not support_fits:
                reasons.append("full_template_support_does_not_fit")
            if reasons:
                excluded.append({**row, "exclusion_reason": ";".join(reasons)})
            else:
                selected.append(row)
    return pd.DataFrame(selected), pd.DataFrame(excluded)


def preflight() -> None:
    verify_inputs()
    if PREFLIGHT.exists():
        raise FileExistsError(PREFLIGHT)
    PREFLIGHT.mkdir(parents=True)
    selected, excluded = select_endpoints()
    selected.to_csv(PREFLIGHT / "SELECTED_ENDPOINTS.csv", index=False)
    excluded.to_csv(PREFLIGHT / "EXCLUDED_ENDPOINTS.csv", index=False)
    fixture_result = fixtures()
    write_json(PREFLIGHT / "FIXTURES.json", fixture_result)
    source_hash = sha256(Path(__file__))
    protocol = {
        "schema": SCHEMA,
        "status": "frozen_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hash,
        "selected_endpoints": len(selected),
        "excluded_endpoints": len(excluded),
        "selection_rule": (
            "Include every already-selected endpoint/control whose complete retained 152-sample window is "
            "stable +40 um, has raw rows 380:383 exactly zero, and contains full 61-sample template support."
        ),
        "score_definition": (
            "signed initial score=<Twhite,Wrot@preW>/sqrt(sum(Utemplate^2)); boundary uses "
            "Wrot[:,380:384]@preW[380:384]; remaining=original-boundary."
        ),
        "prediction": (
            "For unit 449, the median signed boundary/original share exceeds 0.5, the median boundary score "
            "is positive, and a strict majority of endpoints have boundary score greater than remaining score."
        ),
        "prediction_rule_frozen_before_scores": True,
        "threshold": {"name": "actual Th_learned", "value": 9.0, "use": "descriptive only"},
        "interpretation_limits": (
            "Initial scores are before peeling and are not final peel-conditional detection scores; no inference "
            "about disappearance of detections, production gates, biological identity, or purity."
        ),
        "prohibited_actions_observed": {
            "recording_reads": 0, "matcher_reruns": 0, "model_fits": 0,
            "threshold_searches": 0, "rf_or_holdout_accesses": 0, "numeric_transfers": 0,
        },
    }
    write_json(PREFLIGHT / "PREFLIGHT.json", protocol)
    products = ["PREFLIGHT.json", "SELECTED_ENDPOINTS.csv", "EXCLUDED_ENDPOINTS.csv", "FIXTURES.json"]
    manifest = {name: {"sha256": sha256(PREFLIGHT / name), "bytes": (PREFLIGHT / name).stat().st_size} for name in products}
    write_json(PREFLIGHT / "MANIFEST.json", manifest)
    write_json(PREFLIGHT / "COMPLETE.json", {"schema": SCHEMA, "status": "frozen_preflight",
        "manifest_sha256": sha256(PREFLIGHT / "MANIFEST.json")})


def execute() -> None:
    verify_inputs()
    receipt = json.loads((PREFLIGHT / "PREFLIGHT.json").read_text())
    if receipt["status"] != "frozen_preflight" or receipt["source_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("source differs from frozen preflight")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    selected = pd.read_csv(PREFLIGHT / "SELECTED_ENDPOINTS.csv")
    snippets = np.load(CAPTURE / "SNIPPETS.npz", allow_pickle=False)
    prewhite = torch.from_numpy(snippets["prewhite"])
    captured_white = torch.from_numpy(snippets["whitened"])
    bank = torch.from_numpy(np.load(BANK, allow_pickle=False)).to(torch.float32)
    wpca = torch.from_numpy(np.load(WPCA, allow_pickle=False)).to(torch.float32)
    ops = load_ops(OPS, device=torch.device("cpu"))
    wrot = ops["Wrot"].to(torch.float32)
    if int(ops["nt"]) != 61 or float(ops["Th_learned"]) != 9.0:
        raise RuntimeError("matcher nt or learned threshold differs")
    if not torch.equal(wpca, ops["wPCA"].cpu()):
        raise RuntimeError("wPCA binding differs")
    complement = torch.tensor(np.setdiff1d(np.arange(384), BOUNDARY), dtype=torch.long)
    boundary_index = torch.tensor(BOUNDARY, dtype=torch.long)
    rows = []
    for endpoint in selected.itertuples(index=False):
        lo, hi = int(endpoint.center_index) - 30, int(endpoint.center_index) + 31
        pre = prewhite[int(endpoint.window_index), :, lo:hi]
        observed_white = captured_white[int(endpoint.window_index), :, lo:hi]
        full_white = wrot @ pre
        boundary_white = wrot[:, boundary_index] @ pre[boundary_index]
        remaining_white = wrot[:, complement] @ pre[complement]
        u = bank[int(endpoint.template_id)]
        waveform = torch.einsum("pc,pt->ct", u, wpca)
        norm = torch.sqrt(torch.sum(u**2))
        original = torch.sum(waveform * full_white) / norm
        boundary = torch.sum(waveform * boundary_white) / norm
        remaining_direct = torch.sum(waveform * remaining_white) / norm
        remaining = original - boundary
        temporal = conv1d(full_white.unsqueeze(1), wpca.unsqueeze(1), padding=30)
        installed_score = torch.einsum("pc,cpl->l", u, temporal)[30] / norm
        closure_error = torch.abs(original - boundary - remaining_direct)
        projection_error = torch.abs(original - installed_score)
        capture_error = torch.max(torch.abs(full_white - observed_white))
        if max(float(closure_error), float(projection_error), float(capture_error)) > 2e-5:
            raise RuntimeError(f"{endpoint.window_id}/{endpoint.endpoint}: numerical binding failed")
        original_value = float(original)
        boundary_value = float(boundary)
        remaining_value = float(remaining)
        share = boundary_value / original_value if original_value > 0 else None
        rows.append({
            **endpoint._asdict(),
            "initial_signed_score": original_value,
            "boundary_signed_score": boundary_value,
            "remaining_boundary_zeroed_signed_score": remaining_value,
            "zero_baseline_score": 0.0,
            "boundary_fraction_of_positive_original": share,
            "boundary_positive": boundary_value > 0,
            "boundary_greater_than_remaining": bool(original_value > 0 and boundary_value > remaining_value),
            "original_above_Th_learned_9": original_value > 9.0,
            "boundary_alone_above_Th_learned_9": boundary_value > 9.0,
            "remaining_above_Th_learned_9": remaining_value > 9.0,
            "linear_closure_abs_error": float(closure_error),
            "installed_projection_abs_error": float(projection_error),
            "captured_whitened_binding_max_abs_error": float(capture_error),
        })
    scores = pd.DataFrame(rows)
    scores.to_csv(OUTPUT / "SCORE_ATTRIBUTION.csv", index=False)

    group_rows = []
    for keys, group in scores.groupby(["unit_id", "selection_role"], sort=True):
        shares = group.boundary_fraction_of_positive_original.dropna()
        group_rows.append({
            "unit_id": int(keys[0]), "selection_role": keys[1], "endpoints": len(group),
            "median_initial_signed_score": float(group.initial_signed_score.median()),
            "median_boundary_signed_score": float(group.boundary_signed_score.median()),
            "median_remaining_signed_score": float(group.remaining_boundary_zeroed_signed_score.median()),
            "median_boundary_fraction_of_positive_original": float(shares.median()) if len(shares) else None,
            "boundary_greater_than_remaining_count": int(group.boundary_greater_than_remaining.sum()),
            "original_above_9_count": int(group.original_above_Th_learned_9.sum()),
            "boundary_alone_above_9_count": int(group.boundary_alone_above_Th_learned_9.sum()),
            "remaining_above_9_count": int(group.remaining_above_Th_learned_9.sum()),
        })
    groups = pd.DataFrame(group_rows)
    groups.to_csv(OUTPUT / "GROUP_SUMMARY.csv", index=False)
    target = scores[scores.unit_id.eq(449)]
    target_shares = target.boundary_fraction_of_positive_original.dropna()
    prediction = {
        "unit_449_endpoints": len(target),
        "median_boundary_fraction_of_positive_original": float(target_shares.median()),
        "median_boundary_signed_score": float(target.boundary_signed_score.median()),
        "boundary_greater_than_remaining_count": int(target.boundary_greater_than_remaining.sum()),
        "strict_majority_required": len(target) // 2 + 1,
    }
    prediction["pass"] = bool(
        prediction["median_boundary_fraction_of_positive_original"] > 0.5
        and prediction["median_boundary_signed_score"] > 0
        and prediction["boundary_greater_than_remaining_count"] >= prediction["strict_majority_required"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    target_mask = scores.unit_id.eq(449)
    for mask, label, color in ((~target_mask, "other selected units", "#0072B2"),
                               (target_mask, "unit 449", "#D55E00")):
        axes[0].scatter(scores.loc[mask, "remaining_boundary_zeroed_signed_score"],
                        scores.loc[mask, "boundary_signed_score"], label=label, color=color, alpha=.8)
        axes[1].scatter(scores.loc[mask, "initial_signed_score"],
                        scores.loc[mask, "remaining_boundary_zeroed_signed_score"], label=label, color=color, alpha=.8)
    limits = [float(scores[["boundary_signed_score", "remaining_boundary_zeroed_signed_score"]].min().min()),
              float(scores[["boundary_signed_score", "remaining_boundary_zeroed_signed_score"]].max().max())]
    axes[0].plot(limits, limits, "k--", linewidth=1, label="boundary = remaining")
    axes[0].axhline(9, color="0.5", linestyle=":", linewidth=1)
    axes[0].set_xlabel("remaining score (boundary zeroed)")
    axes[0].set_ylabel("boundary contribution")
    axes[0].set_title("Fixed linear score decomposition")
    axes[0].legend(fontsize=8)
    axes[1].axhline(9, color="0.5", linestyle=":", linewidth=1, label="Th_learned = 9")
    axes[1].axvline(9, color="0.5", linestyle=":", linewidth=1)
    axes[1].set_xlabel("original initial signed score")
    axes[1].set_ylabel("boundary-zeroed remaining score")
    axes[1].set_title("Threshold comparison is descriptive only")
    axes[1].legend(fontsize=8)
    fig.savefig(OUTPUT / "initial_score_attribution.png", dpi=160)
    plt.close(fig)

    summary = {
        "schema": SCHEMA, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(Path(__file__)), "selected_endpoints": len(scores),
        "excluded_endpoints": int(receipt["excluded_endpoints"]), "prediction": prediction,
        "max_linear_closure_abs_error": float(scores.linear_closure_abs_error.max()),
        "max_installed_projection_abs_error": float(scores.installed_projection_abs_error.max()),
        "max_captured_whitened_binding_abs_error": float(scores.captured_whitened_binding_max_abs_error.max()),
        "threshold": 9.0,
        "scope": "Retained snippets, saved bank/ops/field/selection only; zero recording reads and zero matcher reruns.",
        "interpretation": (
            "Fixed-model initial-score attribution only. Initial scores precede peeling and do not establish final "
            "detection disappearance, biological identity, purity, or a production repair decision."
        ),
    }
    write_json(OUTPUT / "SUMMARY.json", summary)
    products = ["SUMMARY.json", "SCORE_ATTRIBUTION.csv", "GROUP_SUMMARY.csv", "initial_score_attribution.png"]
    manifest = {name: {"sha256": sha256(OUTPUT / name), "bytes": (OUTPUT / name).stat().st_size} for name in products}
    write_json(OUTPUT / "MANIFEST.json", manifest)
    write_json(OUTPUT / "COMPLETE.json", {"schema": SCHEMA, "status": "complete",
        "manifest_sha256": sha256(OUTPUT / "MANIFEST.json")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "execute"), required=True)
    args = parser.parse_args()
    preflight() if args.mode == "preflight" else execute()


if __name__ == "__main__":
    main()
