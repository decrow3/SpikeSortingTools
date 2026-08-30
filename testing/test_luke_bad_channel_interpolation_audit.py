import pandas as pd

from testing.luke_bad_channel_interpolation_audit import BASELINE, INTERPOLATED, run


def test_interpolation_audit_pairs_events_and_tracks_bad_channel(tmp_path):
    rows = []
    events = [
        ("E1", "neural", "unmatched", 191, 25, 80.0, 30.0),
        ("E2", "neural", "matched", 20, 20, 50.0, 50.0),
        ("E3", "artifact", "matched", 30, 30, 40.0, 40.0),
    ]
    for stage in (BASELINE, INTERPOLATED):
        for review_id, label, status, blank_ch, interp_ch, blank_amp, interp_amp in events:
            is_interpolated = stage == INTERPOLATED
            rows.append(
                {
                    "review_id": review_id,
                    "sample_index": len(rows),
                    "time_seconds": 1.0,
                    "window": "test",
                    "status": status,
                    "review_label": label,
                    "automatic_neural_like": False,
                    "stage": stage,
                    "peak_channel": interp_ch if is_interpolated else blank_ch,
                    "peak_depth_um": 0.0,
                    "peak_amplitude_counts": interp_amp if is_interpolated else blank_amp,
                    "peak_snr": 5.0,
                    "active_channels": 2,
                    "local_energy_fraction": 0.8,
                    "footprint_depth_sd_um": 30.0,
                    "extra_temporal_extrema": 1,
                    "sidelobe_to_core_energy": 0.1,
                    "spatial_peak_count_4sigma": 1,
                }
            )
    # sample_index is part of the pairing identity and must match across stages.
    frame = pd.DataFrame(rows)
    frame.loc[frame["stage"] == INTERPOLATED, "sample_index"] = [0, 1, 2]
    frame.loc[frame["stage"] == BASELINE, "sample_index"] = [0, 1, 2]
    source = tmp_path / "metrics.csv"
    frame.to_csv(source, index=False)
    result = run(source, tmp_path / "out")
    assert result["channel_191_peak_events"]["all_reviewed"] == 1
    assert result["channel_191_peak_events"]["remain_on_191_after_interpolation"] == 0
    assert result["population_summary"][0]["exact_peak_amplitude_fraction"] == 2 / 3
