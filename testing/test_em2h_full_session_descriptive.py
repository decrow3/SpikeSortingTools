import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.em2h_full_session_descriptive import (
    domain_codes,
    mask_membership,
    temporal_blocks,
    unit_domain_metrics,
    segment_safe_isi,
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
