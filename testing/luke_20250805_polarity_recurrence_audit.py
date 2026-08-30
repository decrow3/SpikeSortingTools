"""Cross-session recurrence check for Luke's imec1 polarity/negative-deficit pattern.

Purpose: test hardware/acquisition-architecture vs biology/session as the
source of the imec0/imec1 raw-voltage polarity asymmetry documented for
2025-08-04 (see docs/luke_yates_raw_voltage_audit_notes.md), by repeating the
same matched channel-level event audit on 2025-08-05 -- the next session on
the same probes/headstage -- and comparing per-channel polarity profiles
across days.

This is a discovery-only diagnostic. It reads small time-windowed batches via
memmap (no full-file load), applies the same 300--6000 Hz bandpass and 100 um
local median reference used throughout the Luke raw-voltage work, and detects
fixed 75 uV threshold extrema with the same 0.5 ms / 100 um deduplication used
in testing/luke_yates_raw_voltage_audit.py and
testing/luke_imec1_event_localization_audit.py. It does not open Yates, does
not run a sorter, and does not touch the sealed holdout cohort.

Logic:
- For each of {2025-08-04, 2025-08-05} x {imec0, imec1}, sample n_batches
  evenly spaced 2 s windows across the full session and count local-referenced
  positive/negative extrema per physical channel at a fixed 75 uV threshold.
- Compute a per-channel polarity index log((positive+0.5)/(negative+0.5)).
- Correlate that per-channel profile across days for the same probe (hardware
  recurrence, since anatomy/insertion differs day to day) and across probes
  within the same day (stream-identity control).

A same-probe, cross-day correlation that is much stronger than the
cross-probe, same-day correlation would favor a channel-fixed hardware/
acquisition cause over a session-specific biological one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_yates_raw_voltage_audit import (
    RecordingSpec,
    collapse_candidates,
    extrema_candidates,
    local_median_reference,
    read_batch,
    select_batch_starts,
    spatial_neighbors,
)

DEFAULT_OUTPUT = Path("testing/outputs/luke_20250805_polarity_recurrence_audit")
THRESHOLD_UV = 75.0
STAGE = "common_bandpass_local_reference"


def parse_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(errors="strict").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        result[key.lstrip("~")] = value.strip()
    return result


def build_spec(session: str, probe: str, raw_dir: Path, file_prefix: str, dredge_dir: Path) -> RecordingSpec:
    binary = raw_dir / f"{file_prefix}_t0.{probe}.ap.bin"
    meta = parse_meta(raw_dir / f"{file_prefix}_t0.{probe}.ap.meta")
    processed = dredge_dir / "preprocessed_recording"
    locations = np.load(processed / "properties/location.npy").astype(np.float32)
    n_channels_file = int(float(meta["nSavedChans"]))
    sampling_rate_hz = float(meta["imSampRate"])
    ap_gain = float(meta["imChan0apGain"])
    ai_range_max = float(meta["imAiRangeMax"])
    max_int = float(meta["imMaxInt"])
    gain_uv_per_count = ai_range_max / max_int * 1e6 / ap_gain
    n_frames = binary.stat().st_size // (n_channels_file * 2)
    return RecordingSpec(
        name=f"Luke {probe} session {session}",
        binary=binary,
        n_channels_file=n_channels_file,
        neural_channels=384,
        n_frames=n_frames,
        sampling_rate_hz=sampling_rate_hz,
        gain_uv_per_count=gain_uv_per_count,
        locations_um=locations,
        shanks=np.zeros(384, dtype=np.int16),
        window_kind="session-wide",
    )


def load_specs() -> list[RecordingSpec]:
    specs = [
        build_spec(
            "20250804",
            "imec0",
            Path("/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0/Luke0730_V2V1_g0_imec0"),
            "Luke0730_V2V1_g0",
            Path("/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0"),
        ),
        build_spec(
            "20250804",
            "imec1",
            Path("/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0/Luke0730_V2V1_g0_imec1"),
            "Luke0730_V2V1_g0",
            Path("/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec1"),
        ),
        build_spec(
            "20250805",
            "imec0",
            Path("/mnt/NPX/Luke/20250805/Luke0805_V2V1_g0/Luke0805_V2V1_g0_imec0"),
            "Luke0805_V2V1_g0",
            Path("/mnt/NPX/Luke/20250805/dredge_pipeline_results_Luke0805_V2V1_g0_imec0"),
        ),
        build_spec(
            "20250805",
            "imec1",
            Path("/mnt/NPX/Luke/20250805/Luke0805_V2V1_g0/Luke0805_V2V1_g0_imec1"),
            "Luke0805_V2V1_g0",
            Path("/mnt/NPX/Luke/20250805/dredge_pipeline_results_Luke0805_V2V1_g0_imec1"),
        ),
    ]
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--batch-duration-s", type=float, default=2.0)
    parser.add_argument("--padding-s", type=float, default=0.1)
    parser.add_argument("--threshold-uv", type=float, default=THRESHOLD_UV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def run(n_batches: int, batch_s: float, padding_s: float, threshold_uv: float, output_dir: Path) -> dict:
    sos = butter(3, (300.0, 6000.0), btype="bandpass", fs=30000.0, output="sos")
    rows: list[dict] = []
    for spec in load_specs():
        print(f"Analyzing {spec.name}", flush=True)
        local_neighbors = spatial_neighbors(spec.locations_um, spec.shanks, 100.0)
        starts = select_batch_starts(spec, n_batches, batch_s, padding_s)
        trim = int(round(padding_s * spec.sampling_rate_hz))
        temporal_radius = int(round(0.0005 * spec.sampling_rate_hz))
        for batch_index, start_s in enumerate(starts):
            raw_uv = read_batch(spec, float(start_s), batch_s, padding_s)
            filtered = sosfiltfilt(sos, raw_uv, axis=0).astype(np.float32)
            if trim:
                filtered = filtered[trim:-trim]
            referenced = local_median_reference(filtered, local_neighbors)
            for polarity, negative in (("negative", True), ("positive", False)):
                times, channels, amplitudes = extrema_candidates(referenced, negative)
                selected = np.flatnonzero(amplitudes >= threshold_uv)
                kept_local = collapse_candidates(
                    times[selected],
                    channels[selected],
                    amplitudes[selected],
                    local_neighbors,
                    len(referenced),
                    temporal_radius,
                )
                kept_channels = channels[selected][kept_local]
                counts = np.bincount(kept_channels, minlength=spec.neural_channels)
                for channel, count in enumerate(counts):
                    rows.append(
                        {
                            "dataset": spec.name,
                            "session": spec.name.split()[-1],
                            "probe": spec.name.split()[1],
                            "stage": STAGE,
                            "batch_index": batch_index,
                            "batch_start_s": float(start_s),
                            "duration_s": batch_s,
                            "channel": channel,
                            "y_um": float(spec.locations_um[channel, 1]),
                            "polarity": polarity,
                            "event_count": int(count),
                        }
                    )
            print(f"  batch {batch_index + 1}/{n_batches}", flush=True)

    batch_metrics = pd.DataFrame(rows)
    channel_summary = (
        batch_metrics.groupby(["dataset", "session", "probe", "stage", "channel", "y_um", "polarity"], as_index=False)
        .agg(event_count=("event_count", "sum"), sampled_duration_s=("duration_s", "sum"))
    )
    channel_summary["event_rate_per_s"] = channel_summary["event_count"] / channel_summary["sampled_duration_s"]

    wide = channel_summary.pivot_table(
        index=["dataset", "session", "probe", "channel", "y_um"],
        columns="polarity",
        values="event_count",
        fill_value=0,
    ).reset_index()
    wide["polarity_log_ratio"] = np.log((wide["positive"] + 0.5) / (wide["negative"] + 0.5))

    profiles: dict[tuple[str, str], np.ndarray] = {}
    for (session, probe), part in wide.groupby(["session", "probe"]):
        profiles[(session, probe)] = part.sort_values("channel")["polarity_log_ratio"].to_numpy()

    def corr(key_a: tuple[str, str], key_b: tuple[str, str]) -> dict | None:
        if key_a not in profiles or key_b not in profiles:
            return None
        a, b = profiles[key_a], profiles[key_b]
        result = spearmanr(a, b)
        return {
            "a": f"{key_a[1]} {key_a[0]}",
            "b": f"{key_b[1]} {key_b[0]}",
            "n_channels": int(len(a)),
            "spearman_r": float(result.statistic),
            "spearman_p": float(result.pvalue),
        }

    comparisons = [
        corr(("20250804", "imec1"), ("20250805", "imec1")),
        corr(("20250804", "imec0"), ("20250805", "imec0")),
        corr(("20250804", "imec0"), ("20250804", "imec1")),
        corr(("20250805", "imec0"), ("20250805", "imec1")),
    ]
    comparisons = [row for row in comparisons if row is not None]
    for row, label in zip(
        comparisons,
        (
            "imec1_cross_day_same_probe",
            "imec0_cross_day_same_probe",
            "20250804_cross_probe_same_day",
            "20250805_cross_probe_same_day",
        ),
    ):
        row["comparison"] = label

    ch191 = wide.loc[wide["channel"] == 191, ["dataset", "session", "probe", "positive", "negative", "polarity_log_ratio"]]

    summary_rows = []
    for (session, probe), part in channel_summary.groupby(["session", "probe"]):
        exposure_mm = float(np.ptp(part["y_um"].unique())) / 1000.0
        for polarity in ("negative", "positive"):
            sub = part.loc[part["polarity"] == polarity]
            total_events = int(sub["event_count"].sum())
            events_per_mm_s = total_events / (batch_s * n_batches) / exposure_mm if exposure_mm else float("nan")
            summary_rows.append(
                {
                    "session": session,
                    "probe": probe,
                    "polarity": polarity,
                    "total_events": total_events,
                    "events_per_mm_s": events_per_mm_s,
                    "depth_exposure_mm": exposure_mm,
                }
            )
    summary_table = pd.DataFrame(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    channel_summary.to_csv(output_dir / "channel_event_summary.csv", index=False)
    wide.to_csv(output_dir / "channel_polarity_wide.csv", index=False)
    summary_table.to_csv(output_dir / "session_probe_summary.csv", index=False)
    ch191.to_csv(output_dir / "channel_191_check.csv", index=False)
    receipt = {
        "schema_version": 1,
        "purpose": "hardware_vs_biology_recurrence_check_for_imec1_polarity_pattern",
        "method": {
            "filter": "3rd-order Butterworth 300-6000 Hz, filtfilt",
            "reference": STAGE,
            "threshold_uv": threshold_uv,
            "dedup_ms": 0.5,
            "dedup_um": 100.0,
            "n_batches": n_batches,
            "batch_duration_s": batch_s,
            "padding_s": padding_s,
        },
        "specs": [
            {**{k: v for k, v in asdict(spec).items() if k not in {"locations_um", "shanks"}}, "binary": str(spec.binary)}
            for spec in load_specs()
        ],
        "cross_session_cross_probe_correlations": comparisons,
        "session_probe_summary": summary_table.to_dict(orient="records"),
        "channel_191": ch191.to_dict(orient="records"),
    }
    (output_dir / "decision.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    args = parse_args()
    if args.plan_only:
        for spec in load_specs():
            print(json.dumps({"name": spec.name, "binary": str(spec.binary), "n_frames": spec.n_frames, "sampling_rate_hz": spec.sampling_rate_hz}, indent=2))
        return
    receipt = run(args.n_batches, args.batch_duration_s, args.padding_s, args.threshold_uv, args.output_dir)
    print(json.dumps(receipt["cross_session_cross_probe_correlations"], indent=2))


if __name__ == "__main__":
    main()
