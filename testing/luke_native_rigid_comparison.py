"""Compare Luke's completed 12/9 native-rigid and motion-off long-strip sorts.

The compact Group 1 handoff contains identity-bound curated spike arrays and
cached amplitude QC for both arms.  This module deliberately does not run a
sort.  It validates the handoff, replays the generic long-sort comparator, and
adds two spike-only standardized QC metrics that do not require omitted PCA or
template arrays: sliding-RP contamination and amplitude-CV range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from spikeinterface.qualitymetrics.misc_metrics import slidingRP_violations

from pipeline.config import fingerprint
from testing.development_ladder import load_contract
from testing.development_runner import _atomic_json
from testing.luke_threshold_grid_analysis import validate_inventory
from testing.sort_comparison import compare_sorts, load_comparison_inputs


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = Path("/mnt/NPX/Luke/20250804/shared_analysis/luke_group1_handoff_v1")
LOCAL_RECORDING = Path(
    "/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/"
    "group2/long/recording/rescue_recording_manifest.json"
)
OFF = HANDOFF / "arms/rescue_12_9_motion_off"
RIGID = HANDOFF / "arms/rescue_12_9_native_rigid"
REMOTE_SUMMARY = HANDOFF / (
    "execution/benchmarks/reports/"
    "rescue_12_9_native_rigid_vs_rescue_12_9_motion_off.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def assess_efficacy(summary: dict, coverage: dict) -> dict:
    """Keep agreement with an incomplete reference out of directional efficacy.

    M/candidate - M/baseline improves for pure candidate dropout and worsens
    for perfect candidate recovery. Neither its sign nor its reversed sign
    identifies biological recovery, so it cannot close or promote this arm.
    """
    value = summary.get("median_missingness_improvement_pp")
    amplitude_measured = (
        coverage.get("endpoint_status") == "measured"
        and value is not None and np.isfinite(value)
    )
    return {
        "status": "motion_diagnostics_required",
        "efficacy_pass": None,
        "amplitude_efficacy_pass": bool(value >= 5.0) if amplitude_measured else None,
        "identity_continuity_status": "not_identified_by_overlap_asymmetry",
        "reason": (
            "Amplitude efficacy is evaluated separately where coverage supports it. "
            "Directional overlap asymmetry cannot identify continuity improvement. "
            "Motion-field and family-level diagnostics remain required before selection."
        ),
        "threshold_reference": "12/9",
    }


def reassess_saved(source_root: Path, output_root: Path) -> dict:
    """Publish a corrected interpretation without recomputing numerical QC.

    Preserve the original directory and attest every copied data file. Notebook
    rendering is separate so old narrative/output cells cannot enter v2.
    """
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("reassessment must use a separate output directory")
    original_bytes = (source_root / "summary.json").read_bytes()
    result = json.loads(original_bytes)
    if result.get("schema_version") != "luke-native-rigid-comparison-v1":
        raise ValueError("expected the historical v1 comparison")
    output_root.mkdir(parents=True, exist_ok=False)
    copied = {}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix not in (".csv", ".json"):
            continue
        relative = path.relative_to(source_root)
        if str(relative) == "summary.json":
            continue
        content = path.read_bytes()
        target = output_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        copied[str(relative)] = hashlib.sha256(content).hexdigest()
    provenance = {
        "source_root": str(source_root),
        "source_summary_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "source_analysis_digest": result["analysis_digest"],
        "copied_data_sha256": copied,
        "scope": "Interpretation correction only; numerical QC is unchanged, not rerun.",
        "decision_policy": "overlap-asymmetry-is-not-efficacy-v2",
    }
    result["schema_version"] = "luke-native-rigid-comparison-v2"
    result["analysis_digest"] = fingerprint(provenance)
    result["overlap_asymmetry"] = result.pop("continuity_proxy_change")
    result["decision"] = assess_efficacy(result["comparison_summary"], result["coverage_summary"])
    result["limitations"].insert(0, "Overlap asymmetry is not directional efficacy; the original closure is withdrawn.")
    result["reassessment_provenance"] = provenance
    _atomic_json(output_root / "summary.json", result)
    return result


def validate_handoff() -> dict:
    inventory = validate_inventory(HANDOFF)
    staged_contract = load_contract(HANDOFF / "contract/luke0804_hindsight_ladder_v1.json")
    local_contract = load_contract(ROOT / "testing/configs/luke0804_hindsight_ladder_v1.json")
    if staged_contract.raw != local_contract.raw:
        raise RuntimeError("staged and local scientific contracts differ")

    staged_recording = _read_json(HANDOFF / "recording/rescue_recording_manifest.json")
    local_recording = _read_json(LOCAL_RECORDING)
    identity_keys = (
        "request_digest", "recording_content_sha256", "sampling_frequency_hz",
        "num_samples", "num_channels", "dtype", "probe_geometry_hash",
    )
    for key in identity_keys:
        if staged_recording.get(key) != local_recording.get(key):
            raise RuntimeError(f"recording identity mismatch: {key}")

    expected = {
        OFF: ("rescue_12_9_motion_off", 0),
        RIGID: ("rescue_12_9_native_rigid", 1),
    }
    arms = {}
    for path, (name, nblocks) in expected.items():
        manifest_path = path / "candidate_manifest.json"
        manifest = _read_json(manifest_path)
        effective = manifest.get("effective_settings", {})
        if (
            manifest.get("complete") is not True
            or manifest.get("contract_digest") != staged_contract.digest
            or manifest.get("recording_request_digest") != staged_recording["request_digest"]
            or manifest.get("candidate", {}).get("name") != name
            or effective.get("effective_nblocks") != nblocks
            or effective.get("Th_universal") != 12
            or effective.get("Th_learned") != 9
        ):
            raise RuntimeError(f"arm manifest mismatch: {name}")
        arms[name] = {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "sort_identity_digest": manifest["sort_identity_digest"],
            "sorter_config_digest": manifest["sorter_config_digest"],
        }
    return {
        "inventory": inventory,
        "contract_digest": staged_contract.digest,
        "recording_request_digest": staged_recording["request_digest"],
        "recording_content_sha256": staged_recording["recording_content_sha256"],
        "arms": arms,
    }


def amplitude_cv_range(
    times: np.ndarray,
    amplitudes: np.ndarray,
    *,
    duration_frames: int,
    average_num_spikes_per_bin: int = 50,
    percentiles: tuple[float, float] = (5.0, 95.0),
    min_num_bins: int = 10,
) -> tuple[float, float]:
    """Replay SI 0.102.1's amplitude-CV metric from canonical QC amplitudes."""
    times = np.asarray(times, dtype=np.int64)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if not len(times) or len(times) != len(amplitudes):
        return np.nan, np.nan
    bin_size = max(1, int(average_num_spikes_per_bin * duration_frames / len(times)))
    edges = np.arange(0, duration_frames + 1, bin_size, dtype=np.int64)
    if len(edges) < 2:
        return np.nan, np.nan
    mean_amplitude = abs(float(np.mean(amplitudes)))
    if not np.isfinite(mean_amplitude) or mean_amplitude == 0:
        return np.nan, np.nan
    left = np.searchsorted(times, edges[:-1])
    right = np.searchsorted(times, edges[1:])
    spreads = np.array([
        np.std(amplitudes[i:j]) / mean_amplitude if j > i else np.nan
        for i, j in zip(left, right)
    ])
    if len(spreads) < min_num_bins or not np.all(np.isfinite(spreads)):
        return np.nan, np.nan
    return (
        float(np.median(spreads)),
        float(np.percentile(spreads, percentiles[1]) - np.percentile(spreads, percentiles[0])),
    )


def standardized_unit_metrics(
    sort: dict,
    *,
    sampling_frequency_hz: float,
    duration_s: float,
) -> pd.DataFrame:
    times = np.asarray(sort["st"], dtype=np.int64)
    clusters = np.asarray(sort["cl"], dtype=np.int64)
    amplitudes = np.asarray(sort["amp"], dtype=float)
    duration_frames = int(round(sampling_frequency_hz * duration_s))
    rows = []
    for cluster in np.unique(clusters):
        keep = clusters == cluster
        unit_times = times[keep]
        contamination_pct = slidingRP_violations(
            [unit_times], sampling_frequency_hz, duration_s,
            0.25, 1.0, 0.5, 10.0, None,
        )
        cv_median, cv_range = amplitude_cv_range(
            unit_times, amplitudes[keep], duration_frames=duration_frames,
        )
        rows.append({
            "cluster_id": int(cluster),
            "spike_count": int(keep.sum()),
            "sliding_rp_contamination": float(contamination_pct / 100.0)
            if np.isfinite(contamination_pct) else np.nan,
            "amplitude_cv_median_standardized": cv_median,
            "amplitude_cv_range": cv_range,
        })
    return pd.DataFrame(rows)


def _median(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def run(output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    provenance = validate_handoff()
    _atomic_json(output_root / "provenance_validation.json", provenance)

    contract = load_contract(ROOT / "testing/configs/luke0804_hindsight_ladder_v1.json")
    recording = _read_json(LOCAL_RECORDING)
    fs = float(recording["sampling_frequency_hz"])
    duration_s = float(contract.raw["recording"]["duration_s"])
    evaluation = {
        **contract.raw["evaluation"],
        "sampling_frequency_hz": fs,
        "duration_s": duration_s,
        "minimum_common_time_fraction": contract.raw["metrics"]["minimum_common_time_fraction"],
        "minimum_measurable_unit_fraction": contract.raw["metrics"]["minimum_measurable_unit_fraction"],
    }
    baseline, baseline_qc = load_comparison_inputs(
        "12/9 motion off", OFF / "cur/cur_output", OFF / "qc", sampling_frequency_hz=fs
    )
    candidate, candidate_qc = load_comparison_inputs(
        "12/9 native rigid", RIGID / "cur/cur_output", RIGID / "qc", sampling_frequency_hz=fs
    )

    comparison = compare_sorts(
        baseline, candidate, baseline_qc, candidate_qc, evaluation,
        spatial_region=contract.raw["spatial_contract"],
        output_dir=output_root / "comparison",
    )
    remote = _read_json(REMOTE_SUMMARY)
    for key, value in remote["summary"].items():
        if key in comparison["summary"] and comparison["summary"][key] != value:
            raise RuntimeError(f"local comparison does not reproduce remote summary: {key}")

    off_standard = standardized_unit_metrics(
        baseline, sampling_frequency_hz=fs, duration_s=duration_s
    )
    rigid_standard = standardized_unit_metrics(
        candidate, sampling_frequency_hz=fs, duration_s=duration_s
    )
    off_standard.to_csv(output_root / "standardized_qc_motion_off.csv", index=False)
    rigid_standard.to_csv(output_root / "standardized_qc_native_rigid.csv", index=False)

    primary = comparison["primary_matches"]
    primary = primary[primary["interior_primary"]].copy()
    paired = primary.merge(
        off_standard.add_prefix("baseline_"),
        left_on="baseline_cluster", right_on="baseline_cluster_id", how="left",
    ).merge(
        rigid_standard.add_prefix("candidate_"),
        left_on="candidate_cluster", right_on="candidate_cluster_id", how="left",
    )
    for metric in (
        "sliding_rp_contamination", "amplitude_cv_median_standardized", "amplitude_cv_range"
    ):
        paired[f"{metric}_change"] = paired[f"candidate_{metric}"] - paired[f"baseline_{metric}"]
    for metric in ("presence_fraction", "active_lifetime_fraction", "firing_rate_cv", "amplitude_cv"):
        b = comparison["unit_metrics_baseline"][["cluster_id", metric]].rename(
            columns={"cluster_id": "baseline_cluster", metric: f"baseline_{metric}"}
        )
        c = comparison["unit_metrics_candidate"][["cluster_id", metric]].rename(
            columns={"cluster_id": "candidate_cluster", metric: f"candidate_{metric}"}
        )
        paired = paired.merge(b, on="baseline_cluster", how="left").merge(c, on="candidate_cluster", how="left")
        paired[f"{metric}_change"] = paired[f"candidate_{metric}"] - paired[f"baseline_{metric}"]
    paired.to_csv(output_root / "matched_interior_unit_metrics.csv", index=False)

    guardrails = comparison["guardrail_summary"].set_index("metric")
    summary = comparison["summary"]
    coverage = comparison["coverage_summary"]
    split = comparison["split_merge_summary"]
    matched_summary = {
        "families": int(len(paired)),
        "median_candidate_minus_baseline": {
            metric: _median(paired, f"{metric}_change")
            for metric in (
                "sliding_rp_contamination", "amplitude_cv_median_standardized",
                "amplitude_cv_range", "presence_fraction", "active_lifetime_fraction",
                "firing_rate_cv", "amplitude_cv",
            )
        },
        "fraction_candidate_worse": {
            metric: float((pd.to_numeric(paired[f"{metric}_change"], errors="coerce").dropna() > 0).mean())
            for metric in ("sliding_rp_contamination", "amplitude_cv_range", "firing_rate_cv", "amplitude_cv")
        },
    }
    continuity_change = summary["median_candidate_retention"] - summary["median_baseline_retention"]
    guardrail_changes = {
        name: float(guardrails.loc[name, "candidate_minus_baseline"])
        for name in (
            "median_refractory_violation_fraction_1_5ms",
            "chance_aware_near_coincident_excess",
            "edge_unit_fraction", "edge_spike_fraction",
        )
    }
    result = {
        "schema_version": "luke-native-rigid-comparison-v2",
        "analysis_digest": fingerprint({
            "decision_policy": "overlap-asymmetry-is-not-efficacy-v2",
            "provenance": provenance, "evaluation": evaluation,
            "baseline_identity": baseline["identity_digest"],
            "candidate_identity": candidate["identity_digest"],
        }),
        "comparison_summary": summary,
        "coverage_summary": coverage,
        "split_merge_summary": split,
        "guardrail_changes": guardrail_changes,
        "overlap_asymmetry": continuity_change,
        "matched_standardized_qc": matched_summary,
        "decision": assess_efficacy(summary, coverage),
        "limitations": [
            "Overlap asymmetry is descriptive agreement, not directional efficacy; neither sign identifies genuine recovery.",
            "Amplitude-completeness efficacy is not rankable below the prospective 50% coverage gate.",
            "Correspondence is spike-train identity, not biological ground truth.",
            "Raw waveform stability, NN metrics, and SD ratio are unavailable in the compact handoff.",
            "Amplitude CV uses canonical sorter-native full_st amplitudes, not microvolts.",
        ],
    }
    _atomic_json(output_root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reassess-from", type=Path,
                        help="Correct a saved v1 interpretation without rerunning numerical QC")
    args = parser.parse_args()
    result = (reassess_saved(args.reassess_from, args.output_root)
              if args.reassess_from else run(args.output_root))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
