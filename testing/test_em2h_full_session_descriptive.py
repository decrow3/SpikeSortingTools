import json
from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd

from testing.em2h_full_session_descriptive import (
    domain_codes,
    mask_membership,
    temporal_blocks,
    unit_domain_metrics,
    segment_safe_isi,
    main,
)


ROOT = Path(__file__).resolve().parents[1]


def test_mask_membership_uses_half_open_nonoverlapping_spans():
    times = np.asarray([0.9, 1.0, 1.9, 2.0, 3.0, 3.9, 4.0])
    spans = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert mask_membership(times, spans).tolist() == [False, True, True, False, True, True, False]


def test_domain_codes_match_frozen_precedence():
    fs = 1.0
    times = np.asarray([0, 1, 2, 3])
    field_times = np.asarray([0.0, 1.0, 2.0, 3.0])
    deviation = np.asarray([0.0, -130.0, 0.0, 30.0])
    mask = np.asarray([[2.0, 3.0]])
    assert domain_codes(times, fs, field_times, deviation, mask).tolist() == [1, 0, 2, 2]


def test_unit_and_temporal_metrics_are_vectorized_and_exhaustive():
    labels = np.asarray([1, 1, 2, 2, -1])
    domains = np.asarray([0, 1, 1, 2, 0], dtype=np.int8)
    keep = labels >= 0
    summary, table, _ = unit_domain_metrics(
        labels, domains, keep, np.asarray([1.0, 1.0, 1.0]), minimum_flat_events=1
    )
    assert summary["units"] == 2
    assert summary["events"] == 4
    assert summary["events_by_domain"] == {
        "negative_excursion": 1,
        "outside_mask_flat": 2,
        "catalogue_outside_remainder": 1,
    }
    assert table.eligible_flat.all()
    blocks = temporal_blocks(
        np.asarray([0, 1, 5, 8, 9]), labels, keep, 1.0, 10, 5.0
    )
    assert blocks.events.tolist() == [2, 2]
    assert blocks.active_units.tolist() == [1, 1]


def test_contract_keeps_cross_sorter_conclusion_descriptive():
    contract = json.loads((ROOT / "configs/em2h_full_session_descriptive.v1.json").read_text())
    assert contract["status"] == "frozen_before_full_session_outcome"
    assert contract["scope"]["rf"] is False
    assert contract["resources"]["voltage_bytes_read"] == 0
    assert "do not establish" in contract["interpretation"]


def test_segment_safe_isi_never_crosses_domain_boundaries():
    intervals = pd.DataFrame(
        {
            "start_s": [0.0, 5.0],
            "end_s": [5.0, 10.0],
            "domain": ["outside_mask_flat", "negative_excursion"],
        }
    )
    rows = segment_safe_isi(
        np.asarray([1, 2, 5, 6], dtype=np.int64),
        np.asarray([7, 7, 7, 7], dtype=np.int64),
        np.ones(4, dtype=bool),
        intervals,
        1.0,
    )
    by_domain = {row["domain"]: row for row in rows}
    assert by_domain["outside_mask_flat"]["denominator"] == 1
    assert by_domain["negative_excursion"]["denominator"] == 1
    assert by_domain["catalogue_outside_remainder"]["denominator"] == 0


def test_whole_session_audit_end_to_end_on_synthetic_outputs(tmp_path, monkeypatch):
    fs = 10.0
    end_frame = 100
    run = tmp_path / "candidate"
    (run / "sort").mkdir(parents=True)
    config_path = run / "config.json"
    config_path.write_text('{"synthetic": true}\n')
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    times = np.asarray([5, 10, 25, 30, 45, 55, 65, 75, 85, 95], dtype=np.int64)
    labels = np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 1, -1], dtype=np.int32)
    candidate_order = np.asarray([0, 2, 1, 3, 4, 5, 6, 7, 8, 9])
    candidate_sort = run / "sort/dartsort_sorting.npz"
    np.savez(
        candidate_sort,
        times_samples=times[candidate_order],
        labels=labels[candidate_order],
        channels=np.zeros(len(times), dtype=np.int16),
        sampling_frequency=np.asarray(fs),
        geom=np.asarray([[0.0, 0.0]]),
    )
    (run / "receipt.json").write_text(
        json.dumps({"status": "complete", "exit_status": 0, "config_sha256": config_sha})
    )

    reference = tmp_path / "reference"
    reference.mkdir()
    np.save(reference / "spike_times.npy", times[:-1])
    np.save(reference / "spike_clusters.npy", np.asarray([2, 2, 2, 3, 3, 3, 2, 2, 3]))
    np.save(reference / "spike_positions.npy", np.zeros((len(times) - 1, 2)))
    (reference / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n2\tgood\n3\tmua\n")

    field = tmp_path / "field.npz"
    field_times = np.linspace(0.0, 10.0, 21)
    displacement = np.r_[np.full(5, -200.0), np.zeros(8), np.full(8, 100.0)][:, None]
    np.savez(field, time_s=field_times, displacement_um=displacement)
    mask = tmp_path / "mask.csv"
    mask.write_text("start_s,end_s\n")

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    reference_files = {
        name: digest(reference / name)
        for name in (
            "spike_times.npy",
            "spike_clusters.npy",
            "spike_positions.npy",
            "cluster_KSLabel.tsv",
        )
    }
    contract = {
        "scope": {
            "frames_half_open": [0, end_frame],
            "sampling_frequency_hz": fs,
        },
        "candidate": {
            "run": str(run),
            "sorting": str(candidate_sort),
            "resolved_config_sha256": config_sha,
        },
        "reference": {"folder": str(reference), "files": reference_files},
        "domain_inputs": {
            "field": {"path": str(field), "sha256": digest(field)},
            "mask": {"path": str(mask), "sha256": digest(mask)},
        },
        "metrics": {"minimum_flat_events": 1, "time_block_seconds": 5.0},
        "resources": {"cpu_seconds_max": 1800, "memory_gib_max": 20},
        "interpretation": "synthetic descriptive audit",
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["em2h", "--contract", str(contract_path), "--output", str(output)],
    )
    main()
    result = json.loads((output / "RESULT.json").read_text())
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert result["status"] == "complete"
    assert result["resource_limits_pass"] is True
    assert result["candidate_time_ordering"]["normalization"] == "stable_argsort_times_samples"
    assert len(result["summaries"]) == 3
    assert complete["status"] == "complete"
    assert (output / "UNIT_DOMAIN_METRICS.csv").is_file()
    assert (output / "TEMPORAL_BLOCKS.csv").is_file()
