import numpy as np
import pandas as pd

from testing import luke_conditioning_stage_audit as audit


def test_ks_center_car_removes_channel_means_and_common_median():
    values = np.array([[1, 3, 5], [2, 4, 6], [3, 5, 7]], dtype=float)
    result = audit.ks_center_car(values)
    assert np.allclose(result.mean(axis=0), 0)
    assert np.allclose(np.median(result, axis=1), 0)


def test_clip_saturation_changes_only_threshold_exceedances():
    values = np.array([[-5.0, 0.0], [1.0, 6.0]])
    result = audit.clip_saturation(values, 5.0)
    assert result[0, 0] == np.median(values)
    assert result[1, 1] == np.median(values)
    assert result[1, 0] == 1.0


def test_postfilter_blank_dilates_only_in_time():
    values = np.arange(21, dtype=float).reshape(7, 3)
    mask = np.zeros_like(values, dtype=bool)
    mask[3, 1] = True
    result = audit.apply_postfilter_blank(values, mask, 1)
    assert np.all(result[2:5, 1] == np.median(values, axis=0)[1])
    assert np.array_equal(result[:, 0], values[:, 0])


def test_interpolate_191_uses_saved_good_channel_order():
    values = np.zeros((2, audit.N_CHANNELS), dtype=float)
    values[:, 190] = 2.0
    weights = np.zeros((audit.N_CHANNELS - 1, 1), dtype=float)
    # Good-channel index 190 is physical channel 190; index 191 is physical 192.
    weights[190, 0] = 1.0
    result = audit.interpolate_191(values, weights)
    assert np.all(result[:, 191] == 2.0)


def test_materialize_int16_rounds_and_clips():
    values = np.array([-40000.0, -1.6, 1.6, 40000.0])
    result = audit.materialize_int16(values)
    assert result.tolist() == [-32768.0, -2.0, 2.0, 32767.0]


def test_batch_starts_are_within_window():
    window = audit.AuditWindow("x", 10.0, 20.0)
    starts = audit.choose_batch_starts(window, 1000.0, 2.0, 3)
    assert starts.tolist() == [10000, 19000, 28000]


def test_event_counts_collapses_a_shared_event():
    values = np.zeros((101, 3), dtype=float)
    values[50, :] = -10.0
    # Provide nonzero background MAD without creating threshold crossings.
    values[::2] += np.array([1.0, -1.0, 1.0])
    positions = np.array([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]])
    counts = audit.event_counts(values, positions, 1000.0)
    assert counts["negative_6sigma_events_per_s"] == 1000.0 / 101.0


def test_saturation_enrichment_selects_broad_and_focal(tmp_path):
    path = tmp_path / "index.csv"
    import pandas as pd

    pd.DataFrame(
        {
            "dataset": ["Luke imec1"] * 3,
            "window_kind": ["x"] * 3,
            "batch_index": [0, 1, 2],
            "batch_start_s": [10.0, 20.0, 30.0],
            "channel": [0, 0, 0],
            "fraction_abs_raw_over_500uv": [0.1, 0.3, 0.2],
        }
    ).to_csv(path, index=False)
    selected = audit.saturation_enriched_starts(path, 2)
    assert len(selected) == 2
    assert {kind for kind, _ in selected} == {"saturation_broad", "saturation_focal"}


def test_motion_is_explicitly_disabled_in_plan(tmp_path):
    args = type(
        "Args",
        (),
        {"smoke": True, "batches_per_window": 12, "batch_duration_s": 2.0},
    )()
    plan = audit.build_plan(args, 30000.0)
    assert plan["motion_enabled"] is False
    assert all("motion" not in stage for stage in plan["stages"])


def test_kilosort_whitening_handles_exact_channel_dependency():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(500, 4)).astype("float32")
    values[:, 3] = values[:, 1] + values[:, 2]
    positions = np.c_[np.zeros(4), np.arange(4) * 20.0]
    metrics = audit.kilosort_whitening_metrics(values, values, positions)
    assert all(np.isfinite(value) for value in metrics.values())


def test_review_channel_indices_are_depth_local_and_exclude_191():
    depths = np.arange(audit.N_CHANNELS, dtype=float) * 10.0
    event = pd.Series({"peak_depth_um": 1920.0})
    indices = audit.review_channel_indices(event, depths, radius_um=30.0)
    assert indices.tolist() == [189, 190, 192, 193, 194, 195]
    assert 191 not in indices


def test_review_channel_indices_use_review_depth_not_candidate_peak():
    depths = np.arange(10, dtype=float) * 20.0
    event = pd.Series({"peak_depth_um": 80.0})
    indices = audit.review_channel_indices(
        event, depths, radius_um=20.0, excluded_channels=()
    )
    assert indices.tolist() == [3, 4, 5]
