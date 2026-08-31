"""Audit and summarize the completed bounded rescue-versus-AIND experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_full_strip_pair_residual_audit import one_to_one_centers


RESULT_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_aind_downstream_bounded_v1"
)
OUTPUT = Path("testing/outputs/luke_aind_downstream_bounded_endpoint_review")
BASELINE = "rescue_ks_car_on"
CHALLENGERS = ("pinned_aind_ks_car_on", "pinned_aind_ks_car_off")
EXPECTED_WINDOWS = ("T1_high_motion", "T2_combined", "T3_combined")
IMEC0_ARTIFACT_SIDECAR = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec0/"
    "artifacts/raw_over_500uv.h5"
)

# direction: +1 means higher is favorable, -1 means lower is favorable.
METRICS = {
    "sealed_event_recovery": ("sealed_event_recovery", 1),
    "median_good_refractory_fraction_1p5ms": ("refractory_behavior", -1),
    "stable_good_fraction_30s": ("continuity", 1),
    "coincidence_excess": ("coincidence", -1),
    "similar_pairs_per_100_good_units": ("duplicate_burden", -1),
    "residual_pairs_supporting_redundancy": ("residuals", -1),
    "kilosort_good_count": ("secondary_good_count", 1),
    "median_good_contamination_pct": ("secondary_contamination", -1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def load_scores(result_root: Path) -> pd.DataFrame:
    scores = pd.read_csv(result_root / "analysis/bounded_condition_scores.csv")
    scores["similar_pairs_per_100_good_units"] = (
        100.0
        * scores.nearby_similar_good_good_pairs
        / scores.kilosort_good_count.replace(0, np.nan)
    )
    return scores


def validate_score_grain(scores: pd.DataFrame) -> dict[str, Any]:
    key = ["probe", "window", "condition"]
    expected = {
        (probe, window, condition)
        for probe in ("imec0", "imec1")
        for window in EXPECTED_WINDOWS
        for condition in (BASELINE, *CHALLENGERS)
    }
    observed = set(map(tuple, scores[key].itertuples(index=False, name=None)))
    duplicate_rows = int(scores.duplicated(key).sum())
    if observed != expected or duplicate_rows:
        raise ValueError(
            f"Unexpected score grain: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}, duplicates={duplicate_rows}"
        )
    event_counts = scores.groupby(["probe", "window"]).sealed_event_count.nunique()
    if not event_counts.eq(1).all():
        raise ValueError("Sealed-event denominators differ across conditions")
    return {
        "score_rows": int(len(scores)),
        "score_columns": int(len(scores.columns)),
        "duplicate_grain_rows": duplicate_rows,
        "complete_probe_window_condition_cells": int(len(observed)),
        "null_counts": {
            column: int(count)
            for column, count in scores.isna().sum().items()
            if count
        },
    }


def validate_manifests(result_root: Path, scores: pd.DataFrame) -> dict[str, Any]:
    config_digests: set[str] = set()
    excess_final = []
    excess_learned = []
    car_only_differences = []
    for row in scores.itertuples(index=False):
        manifest_path = (
            result_root
            / "sorts"
            / row.probe
            / row.window
            / row.condition
            / "bounded_sort_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            raise ValueError(f"Incomplete sort manifest: {manifest_path}")
        config_digests.add(manifest["config_digest"])
        expected_car = row.condition != "pinned_aind_ks_car_off"
        saved = manifest["saved_critical_settings"]
        if saved != {
            "do_CAR": expected_car,
            "nblocks": 0,
            "highpass_cutoff": 300,
        }:
            raise ValueError(f"Critical sorter settings changed: {manifest_path}")
        excess_final.append(int(row.final_excess_spike_count_removed))
        excess_learned.append(int(row.learned_excess_detection_count_removed))

    for probe in ("imec0", "imec1"):
        for window in EXPECTED_WINDOWS:
            paths = {
                condition: result_root
                / "sorts"
                / probe
                / window
                / condition
                / "bounded_sort_manifest.json"
                for condition in CHALLENGERS
            }
            manifests = {
                condition: json.loads(path.read_text())
                for condition, path in paths.items()
            }
            on = manifests["pinned_aind_ks_car_on"]
            off = manifests["pinned_aind_ks_car_off"]
            if on["recording_request_digest"] != off["recording_request_digest"]:
                raise ValueError(f"AIND recordings differ for {probe}/{window}")
            param_differences = {
                key
                for key in set(on["sorter_params"]) | set(off["sorter_params"])
                if on["sorter_params"].get(key) != off["sorter_params"].get(key)
            }
            car_only_differences.append(sorted(param_differences))
            if param_differences != {"do_CAR"}:
                raise ValueError(
                    f"AIND CAR ablation changed other sorter settings for "
                    f"{probe}/{window}: {sorted(param_differences)}"
                )

    if len(config_digests) != 1:
        raise ValueError(f"Multiple config digests: {sorted(config_digests)}")
    return {
        "config_digest": next(iter(config_digests)),
        "completed_sort_manifests": int(len(scores)),
        "aind_car_ablation_cells": int(len(car_only_differences)),
        "aind_car_ablation_only_parameter_difference": "do_CAR",
        "final_excess_spikes_removed_total": int(sum(excess_final)),
        "final_excess_spikes_removed_max_per_sort": int(max(excess_final)),
        "learned_excess_detections_removed_total": int(sum(excess_learned)),
        "learned_excess_detections_removed_max_per_sort": int(max(excess_learned)),
    }


def paired_metric_review(scores: pd.DataFrame) -> pd.DataFrame:
    paired = scores.pivot(
        index=["probe", "window"], columns="condition", values=list(METRICS)
    )
    rows = []
    for challenger in CHALLENGERS:
        for metric, (family, direction) in METRICS.items():
            baseline = paired[(metric, BASELINE)].astype(float)
            candidate = paired[(metric, challenger)].astype(float)
            delta = candidate - baseline
            signed = direction * delta
            finite_relative = baseline.ne(0) & baseline.notna() & candidate.notna()
            relative = delta[finite_relative] / baseline[finite_relative]
            rows.append(
                {
                    "challenger": challenger,
                    "endpoint_family": family,
                    "metric": metric,
                    "favorable_direction": "higher" if direction > 0 else "lower",
                    "n_pairs": int(delta.notna().sum()),
                    "median_baseline": float(baseline.median()),
                    "median_challenger": float(candidate.median()),
                    "median_paired_delta": float(delta.median()),
                    "median_paired_relative_delta": (
                        float(relative.median()) if len(relative) else np.nan
                    ),
                    "favorable_cells": int((signed > 1e-12).sum()),
                    "tied_cells": int((signed.abs() <= 1e-12).sum()),
                    "unfavorable_cells": int((signed < -1e-12).sum()),
                    "imec0_median_delta": float(delta.loc["imec0"].median()),
                    "imec1_median_delta": float(delta.loc["imec1"].median()),
                }
            )
    return pd.DataFrame(rows)


def sealed_event_discordance(result_root: Path) -> pd.DataFrame:
    rows = []
    for probe in ("imec0", "imec1"):
        for window in EXPECTED_WINDOWS:
            condition_events = {}
            for condition in (BASELINE, *CHALLENGERS):
                path = (
                    result_root
                    / "analysis"
                    / probe
                    / window
                    / condition
                    / "sealed_event_recovery.csv"
                )
                events = pd.read_csv(path).set_index("candidate_id")
                recovered = events.recovered.map(
                    lambda value: (
                        value
                        if isinstance(value, (bool, np.bool_))
                        else {"true": True, "false": False}.get(str(value).lower())
                    )
                )
                if recovered.isna().any():
                    raise ValueError(f"Invalid sealed-event recovery labels: {path}")
                condition_events[condition] = recovered.astype(bool)
            baseline = condition_events[BASELINE]
            for challenger in CHALLENGERS:
                candidate = condition_events[challenger]
                if set(candidate.index) != set(baseline.index):
                    raise ValueError(
                        f"Sealed-event identities differ for {probe}/{window}/{challenger}"
                    )
                candidate = candidate.reindex(baseline.index)
                rows.append(
                    {
                        "probe": probe,
                        "window": window,
                        "challenger": challenger,
                        "sealed_events": int(len(baseline)),
                        "both_recovered": int((baseline & candidate).sum()),
                        "both_missed": int((~baseline & ~candidate).sum()),
                        "gained_vs_rescue": int((~baseline & candidate).sum()),
                        "lost_vs_rescue": int((baseline & ~candidate).sum()),
                        "net_recovery_change": int(candidate.sum() - baseline.sum()),
                        "rescue_recovered": int(baseline.sum()),
                        "challenger_recovered": int(candidate.sum()),
                    }
                )
    return pd.DataFrame(rows)


def condition_summary(scores: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "sealed_event_recovery",
        "learned_detection_count",
        "final_spike_count",
        "final_spikes_per_s",
        "median_good_refractory_fraction_1p5ms",
        "stable_good_fraction_30s",
        "coincidence_excess",
        "nearby_similar_good_good_pairs",
        "similar_pairs_per_100_good_units",
        "residual_pairs_supporting_redundancy",
        "kilosort_good_count",
        "median_good_contamination_pct",
    ]
    return scores.groupby("condition")[metrics].median().reset_index()


def good_unit_rate_summary(result_root: Path, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in scores.itertuples(index=False):
        sorter = result_root / "sorts" / row.probe / row.window / row.condition / "sorter_output"
        times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
        clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
        labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t")
        label_column = next(column for column in labels if column != "cluster_id")
        good = labels.loc[
            labels[label_column].astype(str).str.lower().eq("good"), "cluster_id"
        ].to_numpy(int)
        rates = np.asarray(
            [np.sum(clusters == unit) / float(row.duration_s) for unit in good],
            dtype=float,
        )
        rows.append(
            {
                "probe": row.probe,
                "window": row.window,
                "condition": row.condition,
                "good_units": int(len(good)),
                "median_good_unit_rate_hz": float(np.median(rates)),
                "p10_good_unit_rate_hz": float(np.quantile(rates, 0.1)),
                "p90_good_unit_rate_hz": float(np.quantile(rates, 0.9)),
                "maximum_good_unit_rate_hz": float(np.max(rates)),
                "fraction_good_units_below_0p1_hz": float(np.mean(rates < 0.1)),
                "fraction_good_units_above_50_hz": float(np.mean(rates > 50.0)),
            }
        )
    return pd.DataFrame(rows)


def nearest_distance_frames(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if len(reference) == 0:
        return np.full(values.shape, np.iinfo(np.int64).max, dtype=np.int64)
    insertion = np.searchsorted(reference, values)
    left = reference[np.maximum(insertion - 1, 0)]
    right = reference[np.minimum(insertion, len(reference) - 1)]
    return np.minimum(np.abs(values - left), np.abs(right - values))


def residual_artifact_annotation(result_root: Path) -> pd.DataFrame:
    import h5py

    supported = []
    for path in sorted(
        (result_root / "analysis").glob("*/*/*/residual/pair_residual_summary.csv")
    ):
        frame = pd.read_csv(path)
        frame = frame[frame.residual_supports_redundant_templates.astype(bool)]
        for pair in frame.itertuples(index=False):
            supported.append((path, pair))
    rows = []
    for path, pair in supported:
        condition = path.parts[-3]
        window = path.parts[-4]
        probe = path.parts[-5]
        base = {
            "probe": probe,
            "window": window,
            "condition": condition,
            "first_unit": int(pair.first_unit),
            "second_unit": int(pair.second_unit),
            "residual_supported": True,
        }
        if probe != "imec0" or not IMEC0_ARTIFACT_SIDECAR.exists():
            rows.append({**base, "artifact_sidecar_available": False})
            continue
        sorter = result_root / "sorts" / probe / window / condition / "sorter_output"
        manifest = json.loads(
            (sorter.parent / "bounded_sort_manifest.json").read_text()
        )
        times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
        clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
        first = np.sort(times[clusters == int(pair.first_unit)].astype(np.int64))
        second = np.sort(times[clusters == int(pair.second_unit)].astype(np.int64))
        with h5py.File(IMEC0_ARTIFACT_SIDECAR, "r") as handle:
            fs = float(handle.attrs["sampling_frequency_hz"])
            claim = handle["claim_active_sample_index"][:].astype(np.int64)
        offset = int(round(float(manifest["window"]["start_s"]) * fs))
        first_dist = nearest_distance_frames(first + offset, claim)
        second_dist = nearest_distance_frames(second + offset, claim)
        centers = one_to_one_centers(first, second, int(round(0.5e-3 * fs)))
        center_dist = nearest_distance_frames(centers + offset, claim)
        values = {**base, "artifact_sidecar_available": True}
        for label, milliseconds in (("0p5ms", 0.5), ("2ms", 2.0), ("5ms", 5.0)):
            radius = int(round(milliseconds * 1e-3 * fs))
            values[f"first_spike_artifact_{label}_fraction"] = float(
                np.mean(first_dist <= radius)
            )
            values[f"second_spike_artifact_{label}_fraction"] = float(
                np.mean(second_dist <= radius)
            )
            values[f"coincident_artifact_{label}_fraction"] = float(
                np.mean(center_dist <= radius)
            )
        values["one_to_one_coincident_events"] = int(len(centers))
        rows.append(values)
    return pd.DataFrame(rows)


def artifact_title(path: Path) -> str:
    return path.stem.replace("_", " ").title()


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            rendered = f"{float(value):.6g}"
        else:
            rendered = str(value)
        return rendered.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

    headers = [cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def write_markdown_artifacts(output_dir: Path) -> list[Path]:
    """Mirror machine-readable review artifacts as durable Markdown files."""
    written = []
    for csv_path in sorted(output_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        markdown_path = csv_path.with_suffix(".md")
        body = [
            f"# {artifact_title(csv_path)}",
            "",
            f"Markdown mirror of `{csv_path.name}`. The CSV remains authoritative for computation.",
            "",
            dataframe_markdown(frame) if len(frame) else "_No rows._",
            "",
        ]
        markdown_path.write_text("\n".join(body))
        written.append(markdown_path)

    for json_path in sorted(output_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        markdown_path = json_path.with_suffix(".md")
        body = [
            f"# {artifact_title(json_path)}",
            "",
            f"Markdown mirror of `{json_path.name}`. The JSON remains authoritative for computation.",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
        markdown_path.write_text("\n".join(body))
        written.append(markdown_path)

    index_path = output_dir / "README.md"
    lines = [
        "# Luke bounded AIND downstream endpoint-review artifacts",
        "",
        "Human-readable Markdown mirrors of every machine-readable result and audit artifact in this directory.",
        "The CSV and JSON files remain the authoritative inputs for computation.",
        "",
        "## Artifact index",
        "",
    ]
    for markdown_path in sorted(written):
        lines.append(f"- [{artifact_title(markdown_path)}]({markdown_path.name})")
    lines.append("")
    index_path.write_text("\n".join(lines))
    return [index_path, *written]


def main() -> None:
    args = parse_args()
    scores = load_scores(args.result_root)
    audit = validate_score_grain(scores)
    audit.update(validate_manifests(args.result_root, scores))
    paired = paired_metric_review(scores)
    discordance = sealed_event_discordance(args.result_root)
    summary = condition_summary(scores)
    rates = good_unit_rate_summary(args.result_root, scores)
    artifact_pairs = residual_artifact_annotation(args.result_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "scores_with_review_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_endpoint_review.csv", index=False)
    discordance.to_csv(args.output_dir / "sealed_event_discordance.csv", index=False)
    summary.to_csv(args.output_dir / "condition_medians_review.csv", index=False)
    rates.to_csv(args.output_dir / "good_unit_rate_summary.csv", index=False)
    artifact_pairs.to_csv(
        args.output_dir / "residual_supported_pair_artifact_annotation.csv",
        index=False,
    )
    (args.output_dir / "data_quality_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    write_markdown_artifacts(args.output_dir)
    print(json.dumps(audit, indent=2))
    print("\nPaired endpoint review:\n", paired.to_string(index=False))
    print("\nSealed-event discordance:\n", discordance.to_string(index=False))


if __name__ == "__main__":
    main()
