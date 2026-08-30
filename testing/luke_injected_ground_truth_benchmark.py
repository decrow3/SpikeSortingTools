"""Plan and validate a Luke injected-ground-truth benchmark without raw I/O.

This module deliberately does not know how to open a Luke recording, run a
sorter, or use a GPU.  It defines the sealed manifest and the small-array
injection/scoring primitives needed before those expensive operations are
implemented.  Injection is only permitted into an unconditioned ``float32``
raw-domain voltage view; injecting directly into stored ``int16`` counts is
rejected because addition can overflow or quantize the ground truth.

Examples
--------
Print the preregistered protocol::

    python testing/luke_injected_ground_truth_benchmark.py --plan-only

Exercise the numerical contract on synthetic arrays only::

    python testing/luke_injected_ground_truth_benchmark.py --synthetic-validation
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "luke-injected-ground-truth-v1"
REQUIRED_STRATA = {
    "depth_bin",
    "polarity_morphology",
    "snr_bin",
    "collision",
    "artifact_proximity",
}


@dataclass(frozen=True)
class InjectionEvent:
    """One known event in a synthetic injection schedule.

    ``sample_index`` is the desired template center in the paired background.
    Phase 1 requires ``channel_shift == 0``.  Phase 2 may vary it over time to
    impose a known drift trajectory before conditioning and motion correction.
    """

    event_id: str
    template_id: str
    sample_index: int
    amplitude_scale: float = 1.0
    channel_shift: int = 0
    collision_group: str | None = None
    artifact_distance_samples: int | None = None


def template_sha256(template: np.ndarray) -> str:
    """Return a content hash that seals a qualified template array."""
    values = np.ascontiguousarray(template)
    header = f"{values.dtype.str}|{values.shape}".encode("ascii")
    return hashlib.sha256(header + values.tobytes()).hexdigest()


def validate_template(
    template: np.ndarray,
    *,
    edge_guard_samples: int = 2,
    edge_tolerance: float = 1e-6,
) -> np.ndarray:
    """Validate a baseline-corrected, zero-edged float32 template."""
    values = np.asarray(template)
    if values.dtype != np.float32:
        raise TypeError("templates must be float32 raw-domain voltages")
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("template must have shape (samples, channels)")
    if edge_guard_samples < 1 or 2 * edge_guard_samples >= values.shape[0]:
        raise ValueError("edge_guard_samples leaves no template interior")
    if not np.all(np.isfinite(values)):
        raise ValueError("template contains non-finite values")
    edges = np.concatenate(
        (values[:edge_guard_samples].ravel(), values[-edge_guard_samples:].ravel())
    )
    if np.any(np.abs(edges) > edge_tolerance):
        raise ValueError(
            "template has nonzero temporal edges; baseline-correct and taper it "
            "before sealing"
        )
    if not np.any(np.abs(values[edge_guard_samples:-edge_guard_samples]) > edge_tolerance):
        raise ValueError("template interior is empty")
    return values


def inject_float32_raw_domain(
    background: np.ndarray,
    templates: Mapping[str, np.ndarray],
    events: Sequence[InjectionEvent],
    *,
    edge_guard_samples: int = 2,
) -> np.ndarray:
    """Add sealed templates to a copy of one unconditioned float32 background.

    No clipping or boundary truncation is allowed.  Collisions are represented
    by multiple events whose support overlaps and are added linearly.
    """
    raw = np.asarray(background)
    if raw.dtype != np.float32:
        raise TypeError(
            "direct int16 injection is forbidden; scale counts into a float32 "
            "raw-domain voltage view first"
        )
    if raw.ndim != 2 or min(raw.shape) == 0:
        raise ValueError("background must have shape (samples, channels)")
    if not np.all(np.isfinite(raw)):
        raise ValueError("background contains non-finite values")

    checked = {
        name: validate_template(value, edge_guard_samples=edge_guard_samples)
        for name, value in templates.items()
    }
    injected = raw.copy()
    for event in events:
        if event.template_id not in checked:
            raise KeyError(f"unknown template_id: {event.template_id}")
        if not np.isfinite(event.amplitude_scale) or event.amplitude_scale <= 0:
            raise ValueError("amplitude_scale must be finite and positive")
        waveform = checked[event.template_id]
        half = waveform.shape[0] // 2
        start = int(event.sample_index) - half
        stop = start + waveform.shape[0]
        channel_start = int(event.channel_shift)
        channel_stop = channel_start + waveform.shape[1]
        if start < 0 or stop > raw.shape[0]:
            raise ValueError(f"event {event.event_id} crosses a time boundary")
        if channel_start < 0 or channel_stop > raw.shape[1]:
            raise ValueError(f"event {event.event_id} crosses a channel boundary")
        injected[start:stop, channel_start:channel_stop] += (
            np.float32(event.amplitude_scale) * waveform
        )
    return injected


def score_sample_detections(
    truth_samples: Sequence[int],
    detected_samples: Sequence[int],
    *,
    tolerance_samples: int,
) -> dict[str, float | int | None]:
    """Greedily score detections against known times and report duplicates."""
    truth = np.asarray(truth_samples, dtype=np.int64)
    detected = np.asarray(detected_samples, dtype=np.int64)
    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be nonnegative")
    if truth.size != np.unique(truth).size:
        raise ValueError("truth sample indices must be unique")

    candidates: list[tuple[int, int, int]] = []
    for truth_index, sample in enumerate(truth):
        for detection_index in np.flatnonzero(
            np.abs(detected - sample) <= tolerance_samples
        ):
            candidates.append(
                (abs(int(detected[detection_index] - sample)), truth_index, int(detection_index))
            )
    matched_truth: set[int] = set()
    matched_detections: set[int] = set()
    errors: list[int] = []
    for error, truth_index, detection_index in sorted(candidates):
        if truth_index in matched_truth or detection_index in matched_detections:
            continue
        matched_truth.add(truth_index)
        matched_detections.add(detection_index)
        errors.append(error)

    nearby_count = sum(
        int(np.sum(np.abs(detected - sample) <= tolerance_samples)) for sample in truth
    )
    duplicate_count = max(0, nearby_count - len(matched_detections))
    true_positive = len(matched_truth)
    recall = true_positive / truth.size if truth.size else float("nan")
    precision = true_positive / detected.size if detected.size else float("nan")
    return {
        "truth_count": int(truth.size),
        "detection_count": int(detected.size),
        "true_positive_count": true_positive,
        "false_negative_count": int(truth.size - true_positive),
        "false_positive_count": int(detected.size - true_positive),
        "duplicate_count": duplicate_count,
        "recall": float(recall),
        "precision": float(precision),
        "median_abs_timing_error_samples": (
            float(np.median(errors)) if errors else None
        ),
    }


def build_benchmark_plan(seed: int = 20250804) -> dict:
    """Return the preregistered, raw-data-free benchmark manifest."""
    plan = {
        "schema_version": SCHEMA_VERSION,
        "seed": int(seed),
        "status": "design_only_no_raw_access",
        "safety_contract": {
            "injection_domain": "unconditioned_raw_voltage_before_preprocessing",
            "array_dtype": "float32",
            "stored_int16_injection_allowed": False,
            "clipping_allowed": False,
            "boundary_truncation_allowed": False,
            "template_temporal_edges": "zero_within_tolerance",
        },
        "sealed_inputs": {
            "template_record_fields": [
                "template_id",
                "donor_window_id",
                "donor_unit_id",
                "array_sha256",
                "shape",
                "dtype",
                "qualification_label",
                "strata",
            ],
            "background_record_fields": [
                "background_id",
                "holdout_window_id",
                "source_fingerprint",
                "start_frame",
                "stop_frame",
            ],
            "seal_before_condition_selection": True,
        },
        "splits": {
            "template_extraction": {
                "purpose": "estimate donor waveforms only",
                "must_be_disjoint_from": ["template_qualification", "evaluation_background"],
            },
            "template_qualification": {
                "purpose": "independent morphology and neural-plausibility review",
                "must_be_disjoint_from": ["template_extraction", "evaluation_background"],
            },
            "evaluation_background": {
                "purpose": "sealed untouched Luke holdout windows",
                "must_be_disjoint_from": ["template_extraction", "template_qualification"],
            },
        },
        "strata": {
            "depth_bin": ["shallow", "middle", "deep"],
            "polarity_morphology": [
                "negative_compact",
                "positive_dominant",
                "broad_or_biphasic",
            ],
            "snr_bin": ["4_to_6", "6_to_10", "greater_than_10"],
            "collision": ["isolated", "nearby_neural", "injected_pair"],
            "artifact_proximity": ["far", "near_but_unblanked", "inside_policy_zone"],
        },
        "balance_policy": {
            "phase_1": "equal quota over depth, polarity/morphology, and SNR; "
            "cross balance collision and artifact proximity with a fixed seed",
            "report_each_stratum": True,
            "pool_only_after_stratum_report": True,
        },
        "paired_conditions": {
            "unit_of_pairing": "same float32 background samples and metadata",
            "conditions": ["uninjected", "injected"],
            "condition_order_blinded_for_scoring": True,
            "require_background_hash_equality": True,
        },
        "phase_1": {
            "motion": "none_imposed",
            "purpose": "conditioning and preprocessing recovery benchmark",
        },
        "phase_2": {
            "enabled_after_phase_1_freeze": True,
            "purpose": "known-drift motion-correction benchmark",
            "trajectory_families": ["rigid", "depth_varying_nonrigid"],
            "trajectory_fields": [
                "event_id",
                "time_s",
                "known_depth_um",
                "channel_shift",
            ],
            "score_against_imposed_not_estimated_motion": True,
        },
        "ground_truth_metrics": {
            "event": [
                "recall",
                "precision",
                "timing_error_samples",
                "depth_localization_error_um",
                "amplitude_error_fraction",
            ],
            "failure_modes": [
                "duplicate_count",
                "fragment_count",
                "false_positive_delta_vs_uninjected",
                "residual_energy_fraction",
            ],
            "unit_family": [
                "template_correlation",
                "presence_ratio",
                "waveform_stability",
            ],
            "phase_2_motion": [
                "trajectory_rmse_um",
                "residual_drift_um",
                "recovery_by_depth_and_motion_speed",
            ],
        },
        "execution_gates": [
            "freeze window and donor manifests",
            "seal qualified float32 templates by SHA-256",
            "pass synthetic CPU validation",
            "verify paired background hashes",
            "run phase 1 before enabling imposed drift",
        ],
    }
    validate_benchmark_plan(plan)
    return plan


def validate_benchmark_plan(plan: Mapping) -> None:
    """Reject a plan that weakens the predeclared safety or balance contract."""
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected benchmark schema version")
    safety = plan.get("safety_contract", {})
    if safety.get("array_dtype") != "float32":
        raise ValueError("benchmark injection dtype must be float32")
    if safety.get("stored_int16_injection_allowed") is not False:
        raise ValueError("stored int16 injection must remain forbidden")
    if set(plan.get("strata", {})) != REQUIRED_STRATA:
        raise ValueError("benchmark must declare all and only the required strata")
    if plan.get("paired_conditions", {}).get("conditions") != ["uninjected", "injected"]:
        raise ValueError("benchmark requires paired uninjected/injected conditions")
    splits = plan.get("splits", {})
    if set(splits) != {
        "template_extraction",
        "template_qualification",
        "evaluation_background",
    }:
        raise ValueError("benchmark requires three disjoint data splits")
    if plan.get("phase_2", {}).get("enabled_after_phase_1_freeze") is not True:
        raise ValueError("phase 2 drift must follow a frozen phase 1")


def run_synthetic_validation() -> dict:
    """Exercise pairing, collision addition, sealing, and timing metrics on CPU."""
    rng = np.random.default_rng(17)
    background = rng.normal(0, 0.05, size=(128, 6)).astype(np.float32)
    original_background = background.copy()
    template = np.zeros((9, 3), dtype=np.float32)
    template[3:6, 1] = np.array([-1.0, -4.0, -1.0], dtype=np.float32)
    events = [
        InjectionEvent("a", "negative", 40, 1.0, 1, "collision-a"),
        InjectionEvent("b", "negative", 41, 0.5, 1, "collision-a"),
    ]
    injected = inject_float32_raw_domain(background, {"negative": template}, events)
    if not np.array_equal(background, original_background):
        raise AssertionError("synthetic background mutation detected")
    if np.array_equal(background, injected):
        raise AssertionError("synthetic injection had no effect")
    metrics = score_sample_detections([40, 41], [40, 41, 41, 90], tolerance_samples=0)
    return {
        "passed": True,
        "dtype": str(injected.dtype),
        "template_sha256": template_sha256(template),
        "paired_background_unchanged": True,
        "collision_peak_delta": float(np.max(np.abs(injected - background))),
        "example_metrics": metrics,
        "spikeinterface_used": False,
        "note": "Pure NumPy is intentional; SpikeInterface remains an optional phase-1 adapter.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--synthetic-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=20250804)
    parser.add_argument("--output", type=Path, help="Optional JSON output for the plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.plan_only and not args.synthetic_validation:
        raise SystemExit("Choose --plan-only and/or --synthetic-validation; no raw runner exists")
    result: dict = {}
    if args.plan_only:
        result["plan"] = build_benchmark_plan(args.seed)
    if args.synthetic_validation:
        result["synthetic_validation"] = run_synthetic_validation()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        if not args.plan_only:
            raise SystemExit("--output is reserved for a plan manifest")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
