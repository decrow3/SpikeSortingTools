"""Validate the Group 1 handoff and compare the two completed threshold cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import fingerprint
from testing.development_ladder import load_contract
from testing.development_runner import _atomic_json
from testing.ladder_sorter import NAMED_CONFIGS
from testing.sort_comparison import compare_sorts, load_comparison_inputs


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = Path("/mnt/NPX/Luke/20250804/shared_analysis/luke_group1_handoff_v1")
GROUP2 = Path("/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/group2")
GROUP3 = Path("/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/group3")

ARM_PATHS = {
    "9/8": GROUP2 / "long/arms/rescue_9_8_motion_off",
    "9/9": GROUP2 / "long/arms/rescue_9_9_motion_off",
    "10/8": GROUP3 / "long/arms/rescue_10_8_motion_off",
    "10/9": GROUP2 / "long/arms/rescue_10_9_motion_off",
    "10/10": GROUP3 / "long/arms/rescue_10_10_motion_off",
    "12/9": HANDOFF / "arms/rescue_12_9_motion_off",
    "12/10": GROUP3 / "long/arms/rescue_12_10_motion_off",
}

CONFIG_NAMES = {
    "9/8": "rescue_9_8",
    "9/9": "rescue_9_9",
    "10/8": "rescue_10_8",
    "10/9": "rescue_10_9",
    "10/10": "rescue_10_10",
    "12/9": "rescue",
    "12/10": "rescue_12_10",
}

# Every edge is oriented from higher to lower threshold on exactly one axis.
EDGES = (
    ("lower_u_at_l8", "cell_9_10_by_8_9", "10/8", "9/8"),
    ("lower_u_at_l9", "cell_9_10_by_8_9", "10/9", "9/9"),
    ("lower_l_at_u9", "cell_9_10_by_8_9", "9/9", "9/8"),
    ("lower_l_at_u10", "cell_9_10_by_8_9", "10/9", "10/8"),
    ("lower_u_at_l9_high", "cell_10_12_by_9_10", "12/9", "10/9"),
    ("lower_u_at_l10", "cell_10_12_by_9_10", "12/10", "10/10"),
    ("lower_l_at_u10_high", "cell_10_12_by_9_10", "10/10", "10/9"),
    ("lower_l_at_u12", "cell_10_12_by_9_10", "12/10", "12/9"),
)

SCIENTIFIC_CONTRACT_KEYS = ("recording", "spatial_contract", "evaluation", "metrics")
RECORDING_IDENTITY_KEYS = (
    "schema_version",
    "request_digest",
    "recording_content_sha256",
    "expected_binary_bytes",
    "sampling_frequency_hz",
    "num_samples",
    "num_channels",
    "dtype",
    "channel_locations_um",
    "probe_geometry_hash",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_inventory(root: Path) -> dict:
    inventory_path = root / "SHA256SUMS"
    entries: dict[str, str] = {}
    for line in inventory_path.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("*").removeprefix("./")
        if relative in entries or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("invalid or duplicate handoff inventory path")
        entries[relative] = digest
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != inventory_path
    }
    if set(entries) != actual:
        raise RuntimeError("handoff inventory does not exactly cover the staged files")
    for relative, expected in entries.items():
        path = root / relative
        if path.is_symlink():
            raise RuntimeError("handoff inventory may not contain symlinks")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"handoff checksum mismatch: {relative}")
    return {
        "file_count": len(entries),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "total_inventoried_bytes": sum((root / name).stat().st_size for name in entries),
    }


def validate_provenance() -> dict:
    inventory = validate_inventory(HANDOFF)
    group1_contract = load_contract(HANDOFF / "contract/luke0804_hindsight_ladder_v1.json")
    original_contract = load_contract(ROOT / "testing/configs/luke0804_hindsight_ladder_v1.json")
    interaction_contract = load_contract(
        ROOT / "testing/configs/luke0804_threshold_interactions_v1.json"
    )
    if group1_contract.raw != original_contract.raw:
        raise RuntimeError("staged Group 1 contract is not byte-equivalent in meaning")
    for key in SCIENTIFIC_CONTRACT_KEYS:
        if group1_contract.raw[key] != interaction_contract.raw[key]:
            raise RuntimeError(f"cross-contract scientific mismatch: {key}")

    staged_recording = _read_json(HANDOFF / "recording/rescue_recording_manifest.json")
    local_recording = _read_json(GROUP2 / "long/recording/rescue_recording_manifest.json")
    for key in RECORDING_IDENTITY_KEYS:
        if staged_recording.get(key) != local_recording.get(key):
            raise RuntimeError(f"cross-host recording identity mismatch: {key}")
    if staged_recording["environment"]["uv_lock_sha256"] != local_recording["environment"]["uv_lock_sha256"]:
        raise RuntimeError("cross-host production lock mismatch")

    group1_commit = staged_recording["repository"]["git_commit"]
    local_commit = local_recording["repository"]["git_commit"]
    changed = subprocess.run(
        ["git", "diff", "--name-only", group1_commit, local_commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowed_prefixes = (
        "docs/luke_group1_huklaban5_run.md",
        "testing/configs/luke0804_group1_huklaban5_v1.json",
        "testing/luke_group1_job.py",
        "testing/test_luke_group1_job.py",
    )
    if any(name not in allowed_prefixes for name in changed):
        raise RuntimeError("cross-host commits differ in scientific pipeline code")

    manifests = {}
    for label, arm_root in ARM_PATHS.items():
        manifest = _read_json(arm_root / "candidate_manifest.json")
        expected_name = CONFIG_NAMES[label]
        expected = NAMED_CONFIGS[expected_name]
        params = expected.params()
        effective = manifest["effective_settings"]
        if (
            manifest.get("complete") is not True
            or manifest.get("recording_request_digest") != staged_recording["request_digest"]
            or manifest.get("sorter_config_digest") != expected.digest
            or effective.get("effective_nblocks") != 0
            or effective.get("Th_universal") != params["Th_universal"]
            or effective.get("Th_learned") != params["Th_learned"]
        ):
            raise RuntimeError(f"arm manifest identity mismatch: {label}")
        manifests[label] = {
            "candidate_name": manifest["candidate"]["name"],
            "manifest_sha256": hashlib.sha256(
                (arm_root / "candidate_manifest.json").read_bytes()
            ).hexdigest(),
            "sort_identity_digest": manifest["sort_identity_digest"],
            "sorter_config_digest": manifest["sorter_config_digest"],
            "recording_request_digest": manifest["recording_request_digest"],
        }
    return {
        "inventory": inventory,
        "original_contract_digest": original_contract.digest,
        "interaction_contract_digest": interaction_contract.digest,
        "recording_request_digest": staged_recording["request_digest"],
        "recording_content_sha256": staged_recording["recording_content_sha256"],
        "production_lock_sha256": staged_recording["environment"]["uv_lock_sha256"],
        "group1_git_commit": group1_commit,
        "local_recording_git_commit": local_commit,
        "commit_difference_files": changed,
        "arms": manifests,
    }


def _input_paths(label: str) -> tuple[Path, Path]:
    root = ARM_PATHS[label]
    if label == "12/9":
        return root / "cur/cur_output", root / "qc"
    manifest = _read_json(root / "candidate_manifest.json")
    return Path(manifest["curated_output"]), Path(manifest["qc_directory"])


def _strict_map(report: dict) -> dict[int, int]:
    edges = report["edges"]
    primary = report["primary_matches"]
    bdegree = edges.groupby("baseline_cluster").candidate_cluster.nunique()
    cdegree = edges.groupby("candidate_cluster").baseline_cluster.nunique()
    strict = primary[
        primary.baseline_cluster.map(bdegree).eq(1)
        & primary.candidate_cluster.map(cdegree).eq(1)
    ]
    return dict(zip(strict.baseline_cluster.astype(int), strict.candidate_cluster.astype(int)))


def _quartets(reports: dict[str, dict], unit_metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    definitions = {
        "cell_9_10_by_8_9": {
            "anchor": "10/9",
            "arms": ("9/8", "9/9", "10/8", "10/9"),
            "first": "lower_u_at_l9",
            "second": "lower_l_at_u10",
            "third": "lower_l_at_u9",
            "fourth": "lower_u_at_l8",
        },
        "cell_10_12_by_9_10": {
            "anchor": "12/10",
            "arms": ("10/9", "10/10", "12/9", "12/10"),
            "first": "lower_u_at_l10",
            "second": "lower_l_at_u12",
            "third": "lower_l_at_u10_high",
            "fourth": "lower_u_at_l9_high",
        },
    }
    rows = []
    for cell, spec in definitions.items():
        first = _strict_map(reports[spec["first"]])
        second = _strict_map(reports[spec["second"]])
        third = _strict_map(reports[spec["third"]])
        fourth = _strict_map(reports[spec["fourth"]])
        for anchor in sorted(set(first) & set(second)):
            adjacent_a = first[anchor]
            adjacent_b = second[anchor]
            if adjacent_a not in third or adjacent_b not in fourth:
                continue
            diagonal_a = third[adjacent_a]
            diagonal_b = fourth[adjacent_b]
            if diagonal_a != diagonal_b:
                continue
            if cell == "cell_9_10_by_8_9":
                ids = {"10/9": anchor, "9/9": adjacent_a, "10/8": adjacent_b, "9/8": diagonal_a}
                low_u, high_u, low_l, high_l = "9", "10", "8", "9"
            else:
                ids = {"12/10": anchor, "10/10": adjacent_a, "12/9": adjacent_b, "10/9": diagonal_a}
                low_u, high_u, low_l, high_l = "10", "12", "9", "10"
            row = {"cell": cell, "anchor_cluster": int(anchor)}
            for arm, cid in ids.items():
                metric = unit_metrics[arm].loc[int(cid)]
                row[f"cluster_{arm.replace('/', '_')}"] = int(cid)
                row[f"rate_hz_{arm.replace('/', '_')}"] = float(metric.mean_rate_hz)
                row[f"spikes_{arm.replace('/', '_')}"] = int(metric.spike_count)
            ll = f"{low_u}/{low_l}"
            lh = f"{low_u}/{high_l}"
            hl = f"{high_u}/{low_l}"
            hh = f"{high_u}/{high_l}"
            row["lower_l_effect_low_u_log2_rate"] = math.log2(
                row[f"rate_hz_{ll.replace('/', '_')}"] / row[f"rate_hz_{lh.replace('/', '_')}"]
            )
            row["lower_l_effect_high_u_log2_rate"] = math.log2(
                row[f"rate_hz_{hl.replace('/', '_')}"] / row[f"rate_hz_{hh.replace('/', '_')}"]
            )
            row["interaction_log2_rate"] = (
                row["lower_l_effect_low_u_log2_rate"]
                - row["lower_l_effect_high_u_log2_rate"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _interaction_summary(quartets: pd.DataFrame, arm_spikes: dict[str, int]) -> list[dict]:
    cell_arms = {
        "cell_9_10_by_8_9": ("9/8", "9/9", "10/8", "10/9"),
        "cell_10_12_by_9_10": ("10/9", "10/10", "12/9", "12/10"),
    }
    rows = []
    for cell, (ll, lh, hl, hh) in cell_arms.items():
        values = quartets.loc[quartets.cell == cell, "interaction_log2_rate"]
        aggregate = math.log2(arm_spikes[ll] / arm_spikes[lh]) - math.log2(
            arm_spikes[hl] / arm_spikes[hh]
        )
        rows.append(
            {
                "cell": cell,
                "strict_path_consistent_quartets": int(len(values)),
                "median_interaction_log2_rate": float(values.median()) if len(values) else None,
                "q25_interaction_log2_rate": float(values.quantile(0.25)) if len(values) else None,
                "q75_interaction_log2_rate": float(values.quantile(0.75)) if len(values) else None,
                "fraction_positive_interaction": float((values > 0).mean()) if len(values) else None,
                "aggregate_spike_yield_interaction_log2": aggregate,
            }
        )
    return rows


def run(output_root: Path) -> dict:
    provenance = validate_provenance()
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "provenance_validation.json", provenance)

    recording = _read_json(GROUP2 / "long/recording/rescue_recording_manifest.json")
    contract = load_contract(ROOT / "testing/configs/luke0804_threshold_interactions_v1.json")
    evaluation = {
        **contract.raw["evaluation"],
        "sampling_frequency_hz": float(recording["sampling_frequency_hz"]),
        "duration_s": float(contract.raw["recording"]["duration_s"]),
        "minimum_common_time_fraction": float(contract.raw["metrics"]["minimum_common_time_fraction"]),
        "minimum_measurable_unit_fraction": float(contract.raw["metrics"]["minimum_measurable_unit_fraction"]),
    }
    loaded = {}
    for label in ARM_PATHS:
        curated, qc = _input_paths(label)
        loaded[label] = load_comparison_inputs(
            label, curated, qc, sampling_frequency_hz=evaluation["sampling_frequency_hz"]
        )

    reports = {}
    unit_metrics: dict[str, pd.DataFrame] = {}
    pair_rows = []
    for edge, cell, baseline, candidate in EDGES:
        report = compare_sorts(
            loaded[baseline][0],
            loaded[candidate][0],
            loaded[baseline][1],
            loaded[candidate][1],
            evaluation,
            spatial_region=contract.raw["spatial_contract"],
            output_dir=output_root / "comparisons" / edge,
        )
        reports[edge] = report
        for label, frame in (
            (baseline, report["unit_metrics_baseline"]),
            (candidate, report["unit_metrics_candidate"]),
        ):
            indexed = frame.set_index("cluster_id", drop=False)
            if label in unit_metrics and not unit_metrics[label].equals(indexed):
                raise RuntimeError(f"unit metrics changed across comparisons: {label}")
            unit_metrics[label] = indexed
        summary = report["summary"]
        coverage = report["coverage_summary"]
        guardrails = report["guardrail_summary"].set_index("metric")
        pair_rows.append(
            {
                "edge": edge,
                "cell": cell,
                "baseline": baseline,
                "candidate": candidate,
                "baseline_spikes": summary["baseline_spikes"],
                "candidate_spikes": summary["candidate_spikes"],
                "spike_change_pct": 100.0 * (summary["candidate_spikes"] / summary["baseline_spikes"] - 1.0),
                "baseline_units": summary["baseline_units"],
                "candidate_units": summary["candidate_units"],
                "primary_matches": summary["primary_matches"],
                "median_jaccard": summary["median_jaccard"],
                "eligible_units": coverage["baseline_eligible_units"],
                "measurable_units": coverage["amplitude_measurable_both_common_time"],
                "measurable_fraction": coverage["amplitude_measurable_fraction"],
                "coverage_status": coverage["endpoint_status"],
                "median_missingness_improvement_pp": summary["median_missingness_improvement_pp"],
                "refractory_change": float(
                    guardrails.loc["median_refractory_violation_fraction_1_5ms", "candidate_minus_baseline"]
                ),
                "coincidence_excess_change": float(
                    guardrails.loc["chance_aware_near_coincident_excess", "candidate_minus_baseline"]
                ),
                "edge_unit_fraction_change": float(
                    guardrails.loc["edge_unit_fraction", "candidate_minus_baseline"]
                ),
                "edge_spike_fraction_change": float(
                    guardrails.loc["edge_spike_fraction", "candidate_minus_baseline"]
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output_root / "pairwise_summary.csv", index=False)
    quartets = _quartets(reports, unit_metrics)
    quartets.to_csv(output_root / "strict_consistent_quartets.csv", index=False)
    arm_spikes = {label: int(len(value[0]["st"])) for label, value in loaded.items()}
    interactions = _interaction_summary(quartets, arm_spikes)
    result = {
        "schema_version": "luke-threshold-grid-analysis-v1",
        "analysis_digest": fingerprint(
            {"provenance": provenance, "edges": EDGES, "evaluation": evaluation}
        ),
        "pairwise_comparisons": pair_rows,
        "interaction_cells": interactions,
        "interpretation_limits": [
            "Pairwise amplitude estimates are invalid for ranking when coverage is below the prospective 50% gate.",
            "Quartets require strict degree-one edges and agreement around both paths through a cell.",
            "Interaction estimates are descriptive within this one session and do not establish biological identity or causal generalization.",
        ],
    }
    _atomic_json(output_root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2))


if __name__ == "__main__":
    main()
