"""Compare reviewed Luke events across existing full-session sort variants.

Observed local recovery is paired with a time-jitter null so denser sortings do
not look better merely because accidental spike coincidences are more common.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .luke_raw_high_amplitude_recovery import PROBES
    from .luke_trace_reviewed_events import load_stage, local_match_details
except ImportError:
    from luke_raw_high_amplitude_recovery import PROBES
    from luke_trace_reviewed_events import load_stage, local_match_details


LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
VARIANTS = {
    "patched": "patched_pipeline_results_Luke0804_V2V1_g0_imec1",
    "dredge": "dredge_pipeline_results_Luke0804_V2V1_g0_imec1",
    "dredgetest": "dredgetest_pipeline_results_Luke0804_V2V1_g0_imec1",
    "pipeline": "pipeline_results_Luke0804_V2V1_g0_imec1",
    "pipeline_an5": "pipeline_results_Luke0804_V2V1_g0_imec1_an5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("testing/outputs/luke_multichannel_event_validation/imec1"),
    )
    parser.add_argument("--n-jitters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20250804)
    parser.add_argument("--time-tolerance-ms", type=float, default=0.5)
    parser.add_argument("--depth-tolerance-um", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PROBES["imec1"]
    events = pd.read_csv(args.review_dir / "event_stage_trace.csv")
    populations = {
        "visual_neural_unmatched": (events["review_label"] == "neural")
        & (events["status"] == "unmatched"),
        "automatic_neural_like_unmatched": events["automatic_neural_like"]
        & (events["status"] == "unmatched"),
        "all_unmatched": events["status"] == "unmatched",
    }
    tolerance = int(
        round(args.time_tolerance_ms * 1e-3 * config.sample_rate_hz)
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    quality_rows: list[dict] = []
    for variant, folder in VARIANTS.items():
        stage_path = LUKE_ROOT / folder / "cur" / "cur_sorter_output"
        spike_times, spike_depths = load_stage(stage_path)
        spike_clusters = np.load(stage_path / "spike_clusters.npy", mmap_mode="r").reshape(-1)
        unit_ids, unit_counts = np.unique(spike_clusters, return_counts=True)
        duration_s = float(spike_times[-1] / config.sample_rate_hz)
        labels = pd.read_csv(stage_path / "cluster_KSLabel.tsv", sep="\t")
        contamination = pd.read_csv(stage_path / "cluster_ContamPct.tsv", sep="\t")
        label_column = next(column for column in labels if column != "cluster_id")
        contamination_column = next(
            column for column in contamination if column != "cluster_id"
        )
        contamination_values = contamination[contamination_column].to_numpy(float)
        rates = unit_counts / duration_s
        quality_rows.append(
            {
                "variant": variant,
                "n_spikes": len(spike_times),
                "n_units": len(unit_ids),
                "n_ks_good": int(
                    labels[label_column].astype(str).str.lower().eq("good").sum()
                ),
                "median_unit_rate_hz": float(np.median(rates)),
                "mean_unit_rate_hz": float(np.mean(rates)),
                "median_contamination_pct": float(np.median(contamination_values)),
                "fraction_units_contamination_le_10pct": float(
                    np.mean(contamination_values <= 10.0)
                ),
            }
        )
        for population, mask in populations.items():
            selected = events[mask]
            samples = selected["sample_index"].to_numpy(dtype=np.int64)
            depths = selected["peak_depth_um"].to_numpy(dtype=float)
            observed = float(
                local_match_details(
                    samples,
                    depths,
                    spike_times,
                    spike_depths,
                    tolerance,
                    args.depth_tolerance_um,
                )["present"].mean()
            )
            null = []
            for _ in range(args.n_jitters):
                offset_ms = rng.uniform(20.0, 500.0, len(selected))
                offset_ms *= rng.choice([-1.0, 1.0], len(selected))
                shifted = samples + np.rint(
                    offset_ms * config.sample_rate_hz / 1000.0
                ).astype(np.int64)
                null.append(
                    float(
                        local_match_details(
                            shifted,
                            depths,
                            spike_times,
                            spike_depths,
                            tolerance,
                            args.depth_tolerance_um,
                        )["present"].mean()
                    )
                )
            rows.append(
                {
                    "variant": variant,
                    "population": population,
                    "n_events": len(selected),
                    "n_sorted_spikes": len(spike_times),
                    "observed_recovery": observed,
                    "jitter_null_mean": float(np.mean(null)),
                    "jitter_null_p95": float(np.quantile(null, 0.95)),
                    "recovery_above_null": observed - float(np.mean(null)),
                    "empirical_p": (1 + sum(value >= observed for value in null))
                    / (len(null) + 1),
                    "sorting_path": str(stage_path),
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(args.review_dir / "sort_variant_event_recovery.csv", index=False)
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(args.review_dir / "sort_variant_quality_summary.csv", index=False)

    patched_ops = np.load(
        LUKE_ROOT
        / VARIANTS["patched"]
        / "kilosort4"
        / "sorter_output"
        / "ops.npy",
        allow_pickle=True,
    ).item()
    dredge_ops = np.load(
        LUKE_ROOT / VARIANTS["dredge"] / "cur" / "cur_sorter_output" / "ops.npy",
        allow_pickle=True,
    ).item()
    ignored = {
        "settings",
        "probe",
        "filename",
        "data_dir",
        "cross_peel_claim_ms",
        "cross_peel_claim_um",
        "results_dir",
    }
    patched_settings = patched_ops["settings"]
    dredge_settings = dredge_ops["settings"]
    setting_differences = {}
    for key in sorted(set(patched_settings) | set(dredge_settings)):
        if key in ignored:
            continue
        left = patched_settings.get(key, "<missing>")
        right = dredge_settings.get(key, "<missing>")
        if left != right:
            setting_differences[key] = {"patched": left, "dredge": right}
    summary = {
        "same_recorded_input_path": patched_ops.get("filename")
        == dredge_ops.get("filename"),
        "input_path": patched_ops.get("filename"),
        "non_claim_setting_differences": setting_differences,
        "patched_claim_parameters": {
            "cross_peel_claim_ms": patched_settings.get("cross_peel_claim_ms"),
            "cross_peel_claim_um": patched_settings.get("cross_peel_claim_um"),
        },
        "jitter_null": {
            "iterations": args.n_jitters,
            "offset_ms": "uniform 20-500 ms with random sign",
            "seed": args.seed,
        },
        "interpretation_guardrail": "Higher event recovery is not sufficient; compare duplicate, contamination, refractory, and unit-continuity metrics before selecting a sort.",
    }
    (args.review_dir / "sort_variant_event_recovery.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(results.to_string(index=False))
    print(quality.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
