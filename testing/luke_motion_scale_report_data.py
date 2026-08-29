"""Build bounded, reviewed datasets for the Luke motion-scale report artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_motion_scale_characterization import decompose_spatial_field
from testing.luke_motion_scale_sweep import WINDOW, _load_run_field


ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion_scale_sweep"
)
DEPTHS_UM = np.arange(310.0, 3510.1, 200.0)
RELATIVE_TIMES_S = np.arange(WINDOW.start_s + 1.0, WINDOW.start_s + WINDOW.duration_s, 2.0)


def full_run(probe: str, candidate: str) -> Path:
    runs = sorted((ROOT / "runs" / probe / candidate).glob("full_*"))
    completed = [run for run in runs if (run / "motion.npy").exists()]
    if len(completed) != 1:
        raise RuntimeError(f"Expected one completed full run for {probe}/{candidate}, got {completed}")
    return completed[0]


def rigid_trace(probe: str, candidate: str) -> np.ndarray:
    run = full_run(probe, candidate)
    native_times = np.load(run / "time_bins.npy")
    native_dt = float(np.median(np.diff(native_times)))
    raw_t_start = float(native_times[0] - WINDOW.start_s - native_dt / 2)
    field = _load_run_field(run, RELATIVE_TIMES_S + raw_t_start, DEPTHS_UM)
    trace = decompose_spatial_field(field, DEPTHS_UM)["rigid"]
    return trace - np.median(trace)


def build() -> dict[str, list[dict]]:
    summary = pd.read_csv(ROOT / "motion_scale_sweep_summary.csv")
    agreement = pd.read_csv(ROOT / "motion_scale_sweep_agreement.csv")

    traces = []
    labels = {
        "dredge_nr_200_300": "DREDGE 300/200",
        "decentralized_nr_200_300": "Decentralized 300/200",
    }
    for probe in ("imec0", "imec1"):
        for candidate, label in labels.items():
            for time_s, value in zip(RELATIVE_TIMES_S - WINDOW.start_s, rigid_trace(probe, candidate)):
                traces.append(
                    {
                        "time_s": float(time_s),
                        "displacement_um": round(float(value), 4),
                        "probe": probe,
                        "estimator": label,
                        "series": f"{probe} — {label}",
                    }
                )

    selected = [
        "dredge_nr_current_exact",
        "dredge_nr_current_max80",
        "dredge_nr_100_300",
        "dredge_nr_200_300",
        "dredge_nr_200_600",
        "dredge_nr_400_600",
        "decentralized_nr_200_300",
        "iterative_nr_200_300",
    ]
    full = summary[(summary["split"] == "full") & summary["candidate"].isin(selected)].copy()
    display = {
        "dredge_nr_current_exact": "DREDGE 150/100, cap 37.5",
        "dredge_nr_current_max80": "DREDGE 150/100, cap 80",
        "dredge_nr_100_300": "DREDGE 300/100, cap 80",
        "dredge_nr_200_300": "DREDGE 300/200, cap 80",
        "dredge_nr_200_600": "DREDGE 600/200, cap 80",
        "dredge_nr_400_600": "DREDGE 600/400, cap 80",
        "decentralized_nr_200_300": "Decentralized 300/200",
        "iterative_nr_200_300": "Iterative 300/200",
    }
    candidate_rows = []
    for row in full.itertuples(index=False):
        candidate_rows.append(
            {
                "probe": row.probe,
                "candidate": display[row.candidate],
                "rigid_excursion_um": round(float(row.rigid_excursion_p95_p5_um), 2),
                "median_nonrigid_spread_um": round(float(row.median_nonrigid_spread_um), 2),
                "p95_nonrigid_spread_um": round(float(row.p95_nonrigid_spread_um), 2),
                "residual_corr_length_um": round(float(row.residual_spatial_corr_length_um), 0),
                "half_power_period_s": round(float(row.half_power_period_s), 1),
                "p90_power_period_s": round(float(row.p90_power_period_s), 1),
            }
        )

    sensitivity = [
        row
        for row in candidate_rows
        if row["candidate"]
        in {
            "DREDGE 150/100, cap 37.5",
            "DREDGE 150/100, cap 80",
            "DREDGE 300/200, cap 80",
            "DREDGE 600/200, cap 80",
        }
    ]

    stability = []
    for probe in ("imec0", "imec1"):
        tests = [
            ("Split-half: current 150/100", "split_half", "dredge_nr_current_exact", "dredge_nr_current_exact"),
            ("Split-half: DREDGE 600/200", "split_half", "dredge_nr_200_600", "dredge_nr_200_600"),
            ("Split-half: decentralized 300/200", "split_half", "decentralized_nr_200_300", "decentralized_nr_200_300"),
            ("Cross-method: DREDGE vs decentralized 300/200", "cross_candidate", "dredge_nr_200_300", "decentralized_nr_200_300"),
            ("Cross-method: DREDGE vs iterative 300/200", "cross_candidate", "dredge_nr_200_300", "iterative_nr_200_300"),
        ]
        for label, scope, left, right in tests:
            match = agreement[
                (agreement["scope"] == scope)
                & (agreement["probe"] == probe)
                & (agreement["left_candidate"] == left)
                & (agreement["right_candidate"] == right)
            ]
            if match.empty:
                match = agreement[
                    (agreement["scope"] == scope)
                    & (agreement["probe"] == probe)
                    & (agreement["left_candidate"] == right)
                    & (agreement["right_candidate"] == left)
                ]
            value = float(match.iloc[0]["nonrigid_correlation"])
            stability.append({"probe": probe, "comparison": label, "correlation": round(value, 4)})

    return {
        "motion_traces": traces,
        "parameter_sensitivity": sensitivity,
        "stability": stability,
        "candidate_detail": candidate_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    datasets = build()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in datasets.items():
            pd.DataFrame(rows).to_csv(args.output_dir / f"{name}.csv", index=False)
    print(json.dumps(datasets, allow_nan=False))
