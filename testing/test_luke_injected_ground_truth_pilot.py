import numpy as np
import pandas as pd
import pytest

from testing.luke_injected_ground_truth_pilot import (
    N_CHANNELS,
    build_discovery_pairs,
    build_schedule,
    nearest_artifact_distance,
    prepare_template,
    retention_row,
)
from testing.luke_injected_ground_truth_benchmark import InjectionEvent


def reviewed_tables():
    rows = []
    labels = []
    for unit in (1, 2):
        for index in range(6):
            review_id = f"E{unit}{index}"
            rows.append(
                {
                    "review_id": review_id,
                    "unit_id": unit,
                    "window": "discovery",
                    "peak_snr": 5.0 + index,
                    "sample_index": unit * 1000 + index * 10,
                    "aligned_sample_index": unit * 1000 + index * 10,
                    "peak_channel": 20 + unit,
                }
            )
            labels.append(
                {
                    "review_id": review_id,
                    "review_label": "neural",
                    "review_confidence": "medium",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_discovery_pairs_are_deterministic_and_disjoint():
    key, labels = reviewed_tables()
    first = build_discovery_pairs(key, labels, 4)
    second = build_discovery_pairs(key, labels, 4)
    pd.testing.assert_frame_equal(first, second)
    assert first["split_is_disjoint"].all()
    assert set(first["donor_review_id"]).isdisjoint(first["qualifier_review_id"])


def test_prepare_template_removes_baseline_restricts_channels_and_zeros_edges():
    snippet = np.full((21, N_CHANNELS), 7.0, dtype=np.float32)
    snippet[10, 99:102] -= [2.0, 8.0, 2.0]
    template = prepare_template(snippet, 100, channel_radius=2, edge_guard_samples=4)
    assert template.dtype == np.float32
    assert np.all(template[[0, -1]] == 0)
    assert np.all(template[:, :98] == 0)
    assert np.all(template[:, 103:] == 0)
    assert template[10, 100] == -8.0


def test_nearest_artifact_distance_handles_empty_and_neighbors():
    assert nearest_artifact_distance(10, np.array([], dtype=int)) is None
    assert nearest_artifact_distance(10, np.array([2, 14, 20])) == 4


def test_schedule_reserves_near_artifact_events_when_available():
    pairs = pd.DataFrame({"template_id": [f"T{i:02d}" for i in range(10)]})
    mask = np.zeros((15_000, N_CHANNELS), dtype=bool)
    mask[5_000, 3] = True
    mask[9_000, 7] = True
    events = build_schedule(pairs, len(mask), 30_000.0, mask)
    distances = [event.artifact_distance_samples for event in events]
    assert distances.count(30) == 2


def test_retention_identity_has_unit_amplitude_cosine_and_zero_location_error():
    template = np.zeros((121, N_CHANNELS), dtype=np.float32)
    template[60, 10] = -5
    delta = np.zeros((300, N_CHANNELS), dtype=np.float32)
    delta[100:221] = template
    event = InjectionEvent("one", "T01", 160)
    row = retention_row(
        event,
        template,
        "raw",
        delta,
        delta,
        np.ones(N_CHANNELS, dtype=float),
        30_000.0,
    )
    assert row["amplitude_retention"] == 1.0
    assert row["cosine_to_reference"] == pytest.approx(1.0)
    assert row["localization_error_channels"] == 0
    assert row["polarity"] == "negative"
