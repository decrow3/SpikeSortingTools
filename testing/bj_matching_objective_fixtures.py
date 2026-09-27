#!/usr/bin/env python3
"""CPU-only BJ algebra and physical-sampling fixtures for DARTsort matching."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DARTSORT_COMMIT = "edcfe1b51d672b4136eb13cc78c0875da804b851"
SCALE_MIN = 0.75
SCALE_MAX = 4.0 / 3.0
DEFAULT_INV_LAMBDA = 10_000.0
THRESHOLDSQ = 36.0


def scaled_objective(conv, normsq, inv_lambda, scale_min=SCALE_MIN, scale_max=SCALE_MAX):
    """Source-equivalent scalar form of matching_base._scaled_coarse_objective."""
    conv = np.asarray(conv, dtype=float)
    normsq = np.asarray(normsq, dtype=float)
    b = conv + inv_lambda
    a = normsq + inv_lambda
    scale = np.clip(b / a, scale_min, scale_max)
    scale = np.where(conv < 0, 0.0, scale)
    objective = 2.0 * scale * b - scale * scale * a - inv_lambda
    return scale, objective


def free_objective(conv, normsq):
    conv = np.maximum(np.asarray(conv, dtype=float), 0.0)
    scale = conv / np.asarray(normsq, dtype=float)
    return scale, conv * scale


def direct_residual_decrease(x, template, scale, inv_lambda):
    x = np.asarray(x, dtype=float)
    template = np.asarray(template, dtype=float)
    return (
        np.dot(x, x) - np.dot(x - scale * template, x - scale * template)
        - inv_lambda * (scale - 1.0) ** 2
    )


def algebra_rows():
    base = np.array([3.0, -4.0, 0.0, 0.0], dtype=float)
    base *= np.sqrt(50.0 / np.dot(base, base))
    n0 = float(base @ base)
    rows = []
    for alpha in (0.5, 0.75, 1.0, 4.0 / 3.0, 2.0):
        candidate = alpha * base
        fixed_signal = base
        conv = float(fixed_signal @ candidate)
        normsq = float(candidate @ candidate)
        free_scale, free_obj = free_objective(conv, normsq)
        bound_scale, bound_obj = scaled_objective(conv, normsq, 0.0)
        reg_scale, reg_obj = scaled_objective(conv, normsq, DEFAULT_INV_LAMBDA)
        no_scale_obj = 2.0 * conv - normsq
        co_signal = alpha * base
        co_conv = float(co_signal @ candidate)
        co_norm = normsq
        co_scale, co_obj = scaled_objective(co_conv, co_norm, 0.0)
        assert np.isclose(
            reg_obj,
            direct_residual_decrease(fixed_signal, candidate, reg_scale, DEFAULT_INV_LAMBDA),
        )
        rows.append({
            "alpha": alpha, "base_normsq": n0, "candidate_normsq": normsq,
            "fixed_signal_conv": conv,
            "free_unbounded_scale": float(free_scale),
            "free_unbounded_objective": float(free_obj),
            "free_bounded_scale": float(bound_scale),
            "free_bounded_objective": float(bound_obj),
            "regularized_default_scale": float(reg_scale),
            "regularized_default_objective": float(reg_obj),
            "unscaled_objective": float(no_scale_obj),
            "coscaled_signal_scale": float(co_scale),
            "coscaled_signal_objective": float(co_obj),
            "free_bounded_pass_threshold": bool(bound_obj >= THRESHOLDSQ),
            "regularized_default_pass_threshold": bool(reg_obj >= THRESHOLDSQ),
        })
    free = np.array([row["free_unbounded_objective"] for row in rows])
    assert np.allclose(free, n0)
    assert np.isclose(rows[2]["regularized_default_objective"], n0)
    return rows


def physical_sampling_rows():
    electrodes = np.arange(-160.0, 161.0, 40.0)
    sigma_um = 30.0
    base_spatial = np.exp(-0.5 * (electrodes / sigma_um) ** 2)
    temporal = np.array([0.0, -0.25, -1.0, -0.25, 0.0])
    rows = []
    exact_waveforms = []
    interp_waveforms = []
    for center in np.linspace(0.0, 40.0, 9):
        exact_spatial = np.exp(-0.5 * ((electrodes - center) / sigma_um) ** 2)
        # Shift the samples of the zero-centred continuous waveform using linear
        # interpolation. This approximation is distinct from exact sampling.
        interp_spatial = np.interp(
            electrodes - center, electrodes, base_spatial, left=0.0, right=0.0
        )
        exact = temporal[:, None] * exact_spatial[None, :]
        interp = temporal[:, None] * interp_spatial[None, :]
        exact_waveforms.append(exact)
        interp_waveforms.append(interp)
        denom = np.linalg.norm(exact) * np.linalg.norm(interp)
        rows.append({
            "continuous_center_um": float(center),
            "continuous_peak_ptp": 1.0,
            "exact_sampled_max_channel_ptp": float(np.ptp(exact, axis=0).max()),
            "linear_interp_max_channel_ptp": float(np.ptp(interp, axis=0).max()),
            "exact_sampled_normsq": float(np.sum(exact * exact)),
            "linear_interp_normsq": float(np.sum(interp * interp)),
            "interp_vs_exact_relative_l2": float(np.linalg.norm(interp - exact) / np.linalg.norm(exact)),
            "interp_vs_exact_cosine": float(np.vdot(interp, exact) / denom),
        })
    assert min(row["exact_sampled_max_channel_ptp"] for row in rows) < 0.85
    assert all(row["continuous_peak_ptp"] == 1.0 for row in rows)
    return electrodes, rows, np.asarray(exact_waveforms), np.asarray(interp_waveforms)


def file_sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    algebra = algebra_rows()
    electrodes, sampling, _, _ = physical_sampling_rows()

    for name, rows in (("scaling_algebra.csv", algebra), ("physical_sampling.csv", sampling)):
        with (args.output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    center = [row["continuous_center_um"] for row in sampling]
    axes[0].plot(center, [row["continuous_peak_ptp"] for row in sampling],
                 color="#000000", ls=":", label="continuous peak")
    axes[0].plot(center, [row["exact_sampled_max_channel_ptp"] for row in sampling],
                 color="#0072B2", ls="-", marker="o", label="exact electrode samples")
    axes[0].plot(center, [row["linear_interp_max_channel_ptp"] for row in sampling],
                 color="#D55E00", ls="--", marker="s", label="linear interpolation")
    axes[0].set(xlabel="fractional spatial centre (µm)", ylabel="maximum sampled PTP",
                title="Physical sampling vs interpolation")
    axes[0].legend(frameon=False)
    alpha = [row["alpha"] for row in algebra]
    axes[1].plot(alpha, [row["free_unbounded_objective"] for row in algebra],
                 color="#009E73", ls="-", marker="o", label="free scale, unbounded")
    axes[1].plot(alpha, [row["free_bounded_objective"] for row in algebra],
                 color="#0072B2", ls="--", marker="s", label="free scale, bounded")
    axes[1].plot(alpha, [row["regularized_default_objective"] for row in algebra],
                 color="#D55E00", ls="-.", marker="^", label="default regularized")
    axes[1].axhline(THRESHOLDSQ, color="#000000", ls=":", label="threshold²")
    axes[1].set(xlabel="candidate template scalar α", ylabel="objective",
                title="Fixed signal, scalar-rescaled template")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(args.output / "matching_objective_fixtures.png", dpi=180)
    plt.close(fig)

    summary = {
        "schema": "bj-h1-matching-objective-audit-v1",
        "dartsort_commit": DARTSORT_COMMIT,
        "source_scope": {
            "matching_config": "src/dartsort/util/internal_config.py:759",
            "scale_bounds": "src/dartsort/peel/matching.py:101",
            "objective": "src/dartsort/peel/matching_util/matching_base.py:219 and 708",
            "drifty_recompute": "src/dartsort/peel/matching_util/drifty.py:285",
            "chunk_center": "src/dartsort/peel/matching.py:260",
        },
        "defaults": {
            "threshold": 6.0, "thresholdsq": THRESHOLDSQ,
            "amplitude_scaling_variance": 0.0001,
            "inv_lambda": DEFAULT_INV_LAMBDA,
            "amplitude_scaling_boundary": 1.0 / 3.0,
            "scale_min": SCALE_MIN, "scale_max": SCALE_MAX,
            "template_type": "drifty", "whitening_strategy": "prewhiten_postapply",
        },
        "findings": [
            "For fixed signal x and candidate alpha*t, the free unbounded optimum c^2/n cancels positive scalar alpha exactly.",
            "Cancellation fails when 1/alpha is outside the scale bounds, when the strong default prior pulls scale toward 1, or when signal and template co-scale (objective scales with alpha^2).",
            "The accepted objective is a whitened full-template dot/norm expression, not a PTP threshold. Drifty interpolation recomputes spatial components, normsq, main channel and pairwise convolution at chunk-centre time.",
            "A smooth continuously translated waveform has constant continuous peak but variable maximum electrode-sampled PTP at fractional pitch. That is physical sampling, separate from interpolation approximation error.",
            "Shape, whitening and support changes do not cancel as a pure scalar. PTP attenuation alone cannot justify norm restoration or post-interpolation normalization.",
        ],
        "physical_sampling": {
            "electrode_pitch_um": 40.0,
            "electrode_positions_um": electrodes.tolist(),
            "exact_sampled_ptp_min": min(x["exact_sampled_max_channel_ptp"] for x in sampling),
            "exact_sampled_ptp_max": max(x["exact_sampled_max_channel_ptp"] for x in sampling),
            "linear_interp_relative_l2_max": max(x["interp_vs_exact_relative_l2"] for x in sampling),
            "linear_interp_cosine_min": min(x["interp_vs_exact_cosine"] for x in sampling),
        },
        "scope_limits": {
            "executed_W2_config_not_reviewed": True,
            "BH_payload_accessed": False,
            "production_matcher_run": False,
            "voltage_read": False,
            "recommendation": "Do not normalize solely from PTP. First bind executed config/whitener/compressed representation and compare objective terms on the same saved bank.",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = """# BJ h1 matching objective audit

Verdict: pure positive scalar rescaling cancels only for the free, unbounded
scale optimum with a fixed signal. DARTsort's default matcher is neither free
nor unbounded: it uses a variance 1e-4 prior (inverse lambda 10000) and scale
bounds [0.75, 1.333...]. Co-scaling signal and template changes objective
energy, and shape/whitening/support changes cannot be reduced to a PTP scalar.

At the audited commit, matching samples motion at each chunk centre, drifty
interpolation recomputes spatial components, whitened normsq, main channel and
pairwise convolution, and the acceptance threshold is applied to the full
objective. Consequently lower sampled PTP alone does not prove a lower
acceptance threshold, harmful smoothing or a need for norm restoration.

The smooth spatial fixture cleanly separates physical sampling from numerical
approximation: a continuous translated waveform keeps constant continuous peak,
while its maximum value on a 40-um electrode grid varies with fractional
position. Linear interpolation adds a separately reported approximation error.

Smallest justified next step: after an approved BH handoff exists, bind its
executed config and saved representation to this source commit, then compare
conv, normsq, selected scale and objective on identical saved templates. Do not
patch normalization from PTP alone.
"""
    (args.output / "README.md").write_text(report)
    files = [args.output / x for x in ("summary.json", "scaling_algebra.csv", "physical_sampling.csv", "matching_objective_fixtures.png", "README.md")]
    with (args.output / "SHA256SUMS").open("w") as f:
        for path in files:
            f.write(f"{file_sha(path)}  {path.name}\n")
    print(json.dumps(summary["physical_sampling"]))


if __name__ == "__main__":
    main()
