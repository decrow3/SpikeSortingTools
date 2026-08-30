import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from testing.luke_draw_prospective_holdout_events import (
    amplitude_stratum,
    greedy_deduplicate,
    hash_rank,
    make_blinded_outputs,
    select_events,
    torch_local_median_reference,
    validate_parent_seal,
)


def test_amplitude_strata_boundaries():
    assert amplitude_stratum(50) == "50_to_75"
    assert amplitude_stratum(75) == "75_to_100"
    assert amplitude_stratum(100) == "at_least_100"


def test_greedy_dedup_keeps_larger_neighbor():
    samples = np.array([100, 102, 200])
    channels = np.array([0, 1, 0])
    amplitudes = np.array([60.0, 90.0, 70.0])
    kept = greedy_deduplicate(samples, channels, amplitudes, [np.array([0, 1]), np.array([0, 1])], 3)
    assert set(kept.tolist()) == {1, 2}


def test_rank_is_deterministic():
    assert hash_rank("imec1", "T1", 123, 4) == hash_rank("imec1", "T1", 123, 4)


def test_torch_reference_matches_numpy_even_and_odd_neighborhoods():
    values = np.array([[1, 4, 8], [3, 7, 9]], dtype=np.float32)
    neighbors = [np.array([0, 1]), np.array([0, 1, 2]), np.array([1, 2])]
    expected = np.column_stack(
        [
            values[:, channel] - np.median(values[:, index], axis=1)
            for channel, index in enumerate(neighbors)
        ]
    )
    observed = torch_local_median_reference(values, neighbors, "cpu")
    np.testing.assert_allclose(observed, expected)


def test_sparse_cells_are_enumerated_without_borrowing():
    windows = [
        {
            "probe": "imec0",
            "window_id": "T1_quiet",
            "motion_stratum": "relative_quiet",
        }
    ]
    candidates = pd.DataFrame(
        [
            {
                "probe": "imec0",
                "window_id": "T1_quiet",
                "depth_third": 1,
                "polarity": "negative",
                "amplitude_stratum": "50_to_75",
                "rank_sha256": f"{index:064x}",
            }
            for index in range(2)
        ]
    )
    selected, deficits = select_events(candidates, windows)
    assert len(deficits) == 18
    assert len(selected) == 2
    populated = deficits[deficits.candidate_count > 0].iloc[0]
    assert populated.selected_count == 2
    assert populated.deficit == 2
    assert (deficits[deficits.candidate_count == 0].deficit == 4).all()


def test_blinded_candidates_and_coordinate_key_are_separate():
    chosen = pd.DataFrame(
        [
            {
                "probe": "imec1",
                "window_id": "T1_quiet",
                "motion_stratum": "relative_quiet",
                "depth_third": 2,
                "polarity": "positive",
                "amplitude_stratum": "75_to_100",
                "sample_index": 123,
                "time_s": 0.0041,
                "physical_channel": 17,
                "depth_um": 320.0,
                "amplitude_uv": 80.0,
                "rank_sha256": "a" * 64,
            }
        ]
    )
    blinded, key = make_blinded_outputs(chosen)
    assert list(blinded) == ["candidate_id"]
    assert set(("sample_index", "physical_channel", "amplitude_uv")) <= set(key)
    assert set(("probe", "window_id", "depth_third", "polarity", "amplitude_stratum")) <= set(key)
    assert blinded.candidate_id.iloc[0] == key.candidate_id.iloc[0]


def test_parent_seal_validation_fails_on_manifest_mutation(tmp_path):
    manifest = tmp_path / "manifest.json"
    windows = tmp_path / "windows.csv"
    seal = tmp_path / "seal.json"
    payload = {
        "windows": [
            {
                "probe": probe,
                "window_id": f"W{index}",
                "start_s": float(index),
                "stop_s": float(index + 1),
                "motion_stratum": "relative_quiet",
            }
            for index in range(6)
            for probe in ("imec0", "imec1")
        ]
    }
    manifest.write_text(json.dumps(payload))
    pd.DataFrame(payload["windows"]).drop_duplicates("window_id").to_csv(windows, index=False)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    seal.write_text(json.dumps({"sealed": True, "event_indices_drawn": False, "sha256": digest}))
    assert validate_parent_seal(manifest, seal, windows)["manifest_sha256"] == digest
    manifest.write_text(json.dumps({**payload, "changed": True}))
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        validate_parent_seal(manifest, seal, windows)
