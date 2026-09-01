"""Bounded empirical waveform-versus-depth companion for the KS4 audit.

This discovery-only diagnostic uses the ten already sealed donor identities and
their provisional recurrent-unit labels.  It re-extracts those events from the
accepted rescue recording (the same source-domain adapter as the operator
audit), then performs leave-one-interior-depth-state-out linear prediction.

The result can support or leave unvalidated the smooth waveform-versus-depth
assumption.  It cannot rescue a failed operator gate, establish biological unit
identity, open prospective holdout data, or authorize voltage interpolation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_ks4_native_operator_audit import (
    DEFAULT_DONOR_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_RESCUE_RECORDING,
    array_sha256,
    best_scaled_residual,
    centered_cosine,
    load_ks4_state,
    qualify_rescue_domain_templates,
    validate_versions,
)


OUTPUT = DEFAULT_OUTPUT / "waveform_depth_smoothness"


@dataclass(frozen=True)
class SmoothnessGate:
    maximum_families: int = 6
    minimum_eligible_families: int = 3
    minimum_states_per_family: int = 3
    minimum_depth_span_um: float = 20.0
    maximum_worst_family_median_delta_residual: float = -0.005
    minimum_worst_family_median_delta_cosine: float = 0.0
    maximum_worst_family_median_delta_absolute_amplitude_error: float = 0.005


GATE = SmoothnessGate()


def plan() -> dict:
    return {
        "schema_version": "luke-waveform-depth-smoothness-v1",
        "status": "bounded_discovery_companion_not_biological_ground_truth",
        "gate": asdict(GATE),
        "depth_estimator": "waveform_energy_weighted_probe_depth_um",
        "prediction": "linear_interpolation_between_bracketing_amplitude_normalized_full_probe_waveforms",
        "amplitude_prediction": "linear_interpolation_of_peak_amplitude_between_bracketing_states",
        "baseline": "nearest_observed_depth_state",
        "family_source": "sealed_donor_manifest_provisional_unit_id",
        "prospective_holdout_accessed": False,
        "sorter_run": False,
    }


def waveform_depth_um(waveform: np.ndarray, positions: np.ndarray) -> float:
    values = np.asarray(waveform, dtype=np.float64)
    energy = np.sum(np.square(values), axis=0)
    if not np.any(energy):
        raise ValueError("empty waveform has no inferred depth")
    return float(np.sum(energy * np.asarray(positions)[:, 1]) / np.sum(energy))


def peak_amplitude(waveform: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(waveform))))


def amplitude_normalize(waveform: np.ndarray) -> np.ndarray:
    values = np.asarray(waveform, dtype=np.float64)
    return values / max(peak_amplitude(values), np.finfo(float).eps)


def prediction_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    nearest: np.ndarray,
) -> dict[str, float]:
    pred_residual, _ = best_scaled_residual(predicted, observed)
    nearest_residual, _ = best_scaled_residual(nearest, observed)
    observed_amp = peak_amplitude(observed)
    predicted_error = abs(peak_amplitude(predicted) / observed_amp - 1.0)
    nearest_error = abs(peak_amplitude(nearest) / observed_amp - 1.0)
    pred_cosine = centered_cosine(predicted, observed)
    nearest_cosine = centered_cosine(nearest, observed)
    return {
        "predicted_residual_fraction": pred_residual,
        "nearest_residual_fraction": nearest_residual,
        "delta_residual_vs_nearest": pred_residual - nearest_residual,
        "predicted_cosine": pred_cosine,
        "nearest_cosine": nearest_cosine,
        "delta_cosine_vs_nearest": pred_cosine - nearest_cosine,
        "predicted_absolute_amplitude_error": predicted_error,
        "nearest_absolute_amplitude_error": nearest_error,
        "delta_absolute_amplitude_error_vs_nearest": predicted_error - nearest_error,
    }


def leave_one_depth_state_out(
    family_id: int | str,
    states: list[tuple[str, float, np.ndarray]],
) -> pd.DataFrame:
    """Predict each interior state from its immediately bracketing states."""
    ordered = sorted(states, key=lambda item: (item[1], item[0]))
    rows = []
    for index in range(1, len(ordered) - 1):
        left_id, left_depth, left = ordered[index - 1]
        held_id, held_depth, held = ordered[index]
        right_id, right_depth, right = ordered[index + 1]
        if right_depth <= left_depth:
            continue
        alpha = (held_depth - left_depth) / (right_depth - left_depth)
        if not 0 <= alpha <= 1:
            raise AssertionError("held state is not bracketed")
        morphology = (1.0 - alpha) * amplitude_normalize(left) + alpha * amplitude_normalize(right)
        predicted_amplitude = (1.0 - alpha) * peak_amplitude(left) + alpha * peak_amplitude(right)
        predicted = morphology * predicted_amplitude
        nearest_id, nearest_depth, nearest = min(
            (ordered[index - 1], ordered[index + 1]),
            key=lambda item: (abs(item[1] - held_depth), item[0]),
        )
        rows.append(
            {
                "family_id": family_id,
                "held_template_id": held_id,
                "held_depth_um": held_depth,
                "left_template_id": left_id,
                "left_depth_um": left_depth,
                "right_template_id": right_id,
                "right_depth_um": right_depth,
                "nearest_template_id": nearest_id,
                "nearest_depth_um": nearest_depth,
                "alpha": alpha,
                **prediction_metrics(held, predicted, nearest),
            }
        )
    return pd.DataFrame(rows)


def summarize_smoothness(metrics: pd.DataFrame, family_manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if metrics.empty:
        by_family = pd.DataFrame(
            columns=[
                "family_id",
                "cases",
                "median_delta_residual_vs_nearest",
                "median_delta_cosine_vs_nearest",
                "median_delta_absolute_amplitude_error_vs_nearest",
            ]
        )
    else:
        by_family = (
            metrics.groupby("family_id", observed=True)
            .agg(
                cases=("held_template_id", "size"),
                median_delta_residual_vs_nearest=("delta_residual_vs_nearest", "median"),
                median_delta_cosine_vs_nearest=("delta_cosine_vs_nearest", "median"),
                median_delta_absolute_amplitude_error_vs_nearest=(
                    "delta_absolute_amplitude_error_vs_nearest",
                    "median",
                ),
            )
            .reset_index()
        )
    eligible = family_manifest.loc[family_manifest.eligible]
    enough_families = len(eligible) >= GATE.minimum_eligible_families
    if by_family.empty:
        worst_residual = worst_cosine = worst_amplitude = None
        metric_gate = False
    else:
        worst_residual = float(by_family.median_delta_residual_vs_nearest.max())
        worst_cosine = float(by_family.median_delta_cosine_vs_nearest.min())
        worst_amplitude = float(
            by_family.median_delta_absolute_amplitude_error_vs_nearest.max()
        )
        metric_gate = bool(
            worst_residual <= GATE.maximum_worst_family_median_delta_residual
            and worst_cosine >= GATE.minimum_worst_family_median_delta_cosine
            and worst_amplitude
            <= GATE.maximum_worst_family_median_delta_absolute_amplitude_error
        )
    decision = {
        "eligible_family_count": int(len(eligible)),
        "required_eligible_family_count": GATE.minimum_eligible_families,
        "enough_families": bool(enough_families),
        "worst_family_median_delta_residual": worst_residual,
        "worst_family_median_delta_cosine": worst_cosine,
        "worst_family_median_delta_absolute_amplitude_error": worst_amplitude,
        "metric_gate_pass": metric_gate,
        "smoothness_supported": bool(enough_families and metric_gate),
        "status": "supported" if enough_families and metric_gate else "unvalidated",
    }
    return by_family, decision


def run(output_dir: Path = OUTPUT) -> dict:
    import spikeinterface.core as sc

    versions = validate_versions()
    state = load_ks4_state()
    recording = sc.load(DEFAULT_RESCUE_RECORDING)
    templates, qualification, source = qualify_rescue_domain_templates(
        recording, maximum_templates=10
    )
    if not source["passed"] or len(templates) != 10:
        raise RuntimeError("all ten sealed donors must qualify for the companion")
    manifest = pd.read_csv(DEFAULT_DONOR_MANIFEST)
    records = []
    for row in manifest.itertuples(index=False):
        waveform = templates[row.template_id]
        depth = waveform_depth_um(waveform, state["positions"])
        peak = np.unravel_index(int(np.argmax(np.abs(waveform))), waveform.shape)
        records.append(
            {
                "template_id": row.template_id,
                "family_id": int(row.donor_unit_id),
                "window": row.donor_window,
                "depth_um": depth,
                "peak_channel": int(peak[1]),
                "peak_depth_um": float(state["positions"][peak[1], 1]),
                "peak_amplitude": peak_amplitude(waveform),
                "polarity": "negative"
                if abs(float(waveform.min())) >= abs(float(waveform.max()))
                else "positive",
                "array_sha256": array_sha256(waveform),
            }
        )
    states = pd.DataFrame(records)
    family_rows = []
    metric_frames = []
    for family_id, group in states.groupby("family_id", sort=True):
        selected = group.sort_values(["depth_um", "template_id"])
        depth_span = float(selected.depth_um.max() - selected.depth_um.min())
        polarity_stable = selected.polarity.nunique() == 1
        eligible = bool(
            len(selected) >= GATE.minimum_states_per_family
            and depth_span >= GATE.minimum_depth_span_um
            and polarity_stable
        )
        family_rows.append(
            {
                "family_id": int(family_id),
                "states": int(len(selected)),
                "depth_span_um": depth_span,
                "polarity_stable": polarity_stable,
                "eligible": eligible,
            }
        )
        if eligible:
            values = [
                (row.template_id, float(row.depth_um), templates[row.template_id])
                for row in selected.itertuples(index=False)
            ]
            metric_frames.append(leave_one_depth_state_out(family_id, values))
    family_manifest = pd.DataFrame(family_rows)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    by_family, decision = summarize_smoothness(metrics, family_manifest)
    result = {
        "status": "bounded_empirical_waveform_depth_companion",
        "versions": versions,
        "source_domain": source,
        "gate": asdict(GATE),
        "decision": decision,
        "families_total": int(len(family_manifest)),
        "events_total": int(len(states)),
        "prediction_rows": int(len(metrics)),
        "limitations": [
            "Family IDs are provisional discovery-sort labels, not biological ground truth.",
            "Inferred depth is waveform-energy weighted and is not an independently estimated tissue position.",
            "The companion can leave smoothness unvalidated but cannot rescue a failed operator gate.",
        ],
        "prospective_holdout_accessed": False,
        "sorter_run": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frozen_config.json").write_text(json.dumps(plan(), indent=2) + "\n")
    qualification.to_csv(output_dir / "source_domain_qualification.csv", index=False)
    states.to_csv(output_dir / "waveform_states.csv", index=False)
    family_manifest.to_csv(output_dir / "family_manifest.csv", index=False)
    metrics.to_csv(output_dir / "heldout_prediction_metrics.csv", index=False)
    by_family.to_csv(output_dir / "family_summary.csv", index=False)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan_only == args.run:
        raise SystemExit("Choose exactly one of --plan-only or --run")
    result = plan() if args.plan_only else run(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
