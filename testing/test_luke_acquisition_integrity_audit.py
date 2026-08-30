import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_acquisition_integrity_audit import (
    audit_acquisition,
    disconnected_sites,
    imro_reference_ids,
    polarity_bank_analysis,
)


def _write_stream(root: Path, probe: str, kind: str, rate: float, samples: int, first: int) -> Path:
    path = root / f"run.{probe}.{kind}.meta"
    binary = path.with_suffix(".bin")
    n_channels = 3
    binary.write_bytes(bytes(samples * n_channels * 2))
    channel_map = "(2,2,1)(AP0;0:0)(AP1;1:1)(SY0;2:2)" if kind == "ap" else "(2,2,1)(LF0;0:0)(LF1;1:1)(SY0;2:2)"
    path.write_text(
        "\n".join(
            [
                f"fileSizeBytes={binary.stat().st_size}",
                f"fileTimeSecs={samples / rate}",
                f"firstSample={first}",
                f"imSampRate={rate}",
                f"nSavedChans={n_channels}",
                f"imDatPrb_sn={123 if probe == 'imec0' else 456}",
                "imAiRangeMin=-0.6",
                "imAiRangeMax=0.6",
                "imMaxInt=512",
                "imDatPrb_type=0",
                "imChan0apGain=500",
                "imChan0lfGain=250",
                "~imroTbl=(0,2)(0 0 0 500 250 1)(1 0 0 500 250 1)",
                f"~snsChanMap={channel_map}",
                "~snsGeomMap=(probe,1)(0:0:0:1)(0:20:20:0)",
            ]
        )
        + "\n"
    )
    return path


def test_metadata_only_audit_passes_and_reports_disconnected_site(tmp_path):
    ap = _write_stream(tmp_path, "imec0", "ap", 30_000.0, 3_000, 30_000)
    lf = _write_stream(tmp_path, "imec0", "lf", 2_500.0, 250, 2_500)
    receipt = audit_acquisition([ap, lf])
    assert receipt["status"] == "pass"
    assert receipt["safety"] == {"full_binary_sha1_read": False}
    assert receipt["ap_lf_alignment"][0]["start_aligned"] is True
    assert receipt["ap_lf_alignment"][0]["duration_aligned"] is True
    assert receipt["files"][0]["disconnected_sites"] == [
        {"channel": 1, "shank": 0, "x_um": 20, "y_um": 20}
    ]
    assert receipt["files"][0]["imro_reference_ids"] == [0]
    json.dumps(receipt)


def test_size_mismatch_fails_without_reading_binary(tmp_path):
    ap = _write_stream(tmp_path, "imec0", "ap", 30_000.0, 10, 0)
    text = ap.read_text().replace("fileSizeBytes=60", "fileSizeBytes=62")
    ap.write_text(text)
    receipt = audit_acquisition([ap])
    assert receipt["status"] == "fail"
    assert receipt["files"][0]["checks"]["size_matches_fileSizeBytes"] is False
    assert receipt["files"][0]["sha1"] is None


def test_cross_probe_start_and_duration_alignment(tmp_path):
    first = _write_stream(tmp_path, "imec0", "ap", 30_000.0, 3_000, 30_000)
    second = _write_stream(tmp_path, "imec1", "ap", 30_000.0, 3_000, 30_015)
    receipt = audit_acquisition([first, second])
    assert receipt["status"] == "pass"
    assert receipt["cross_probe_alignment"] == [
        {
            "stream": "ap",
            "probe_count": 2,
            "start_span_s": 0.0004999999999999449,
            "duration_span_s": 0.0,
            "start_aligned": True,
            "duration_aligned": True,
        }
    ]


def test_disconnected_parser_ignores_header():
    assert disconnected_sites("(PRB,1,0,70)(0:11:0:1)(0:43:20:0)")[0]["channel"] == 1
    assert imro_reference_ids("(0,2)(0 0 0 500 250 1)(1 0 2 500 250 1)") == [0, 2]


def test_bank_analysis_requires_explicit_mapping(tmp_path):
    assert polarity_bank_analysis(tmp_path / "unused.csv", None)["status"] == "unavailable"


def test_bank_analysis_controls_smooth_depth_and_detects_bank_effect(tmp_path):
    rows = []
    mappings = []
    for channel in range(40):
        depth = channel * 20.0
        bank = channel % 2
        log_ratio = 0.001 * depth + 1.2 * bank
        negative = 100
        positive = int(round(negative * np.exp(log_ratio)))
        mappings.append({"channel": channel, "electrical_bank": f"bank{bank}"})
        rows.extend(
            [
                {"channel": channel, "y_um": depth, "polarity": "positive", "event_count": positive},
                {"channel": channel, "y_um": depth, "polarity": "negative", "event_count": negative},
            ]
        )
    event_csv = tmp_path / "channel_event_summary.csv"
    map_csv = tmp_path / "mapping.csv"
    pd.DataFrame(rows).to_csv(event_csv, index=False)
    pd.DataFrame(mappings).to_csv(map_csv, index=False)
    result = polarity_bank_analysis(event_csv, map_csv, permutations=99)
    assert result["status"] == "available"
    assert result["bank_partial_r2"] > 0.9
    assert result["permutation_p_value"] <= 0.02
