import numpy as np
import pandas as pd

from testing.luke_adc_spatial_robustness import run_audit


def test_spatial_audit_separates_adc_identity_and_sampling_phase(tmp_path):
    mapping = []
    geometry = []
    rows = []
    for channel in range(384):
        adc = (channel // 24) * 2 + channel % 2
        phase = (channel % 24) // 2
        x = (16, 48, 0, 32)[channel % 4]
        y = (channel // 2) * 20.0
        mapping.append({"channel": channel, "electrical_bank": f"adc{adc}", "adc_index": adc, "mapping_kind": "NP1_ADC_identity"})
        geometry.append({"row": channel, "raw_x_um": x, "raw_y_um": y})
        ratio = np.exp(0.15 * np.sin(phase * 2 * np.pi / 12))
        for dataset, window, stage in (("d", "w", "s"),):
            rows.extend([
                {"dataset": dataset, "window_kind": window, "stage": stage, "channel": channel, "y_um": y, "polarity": "negative", "event_count": 100},
                {"dataset": dataset, "window_kind": window, "stage": stage, "channel": channel, "y_um": y, "polarity": "positive", "event_count": int(round(100 * ratio))},
            ])
    event_csv, map_csv, geom_csv = tmp_path / "events.csv", tmp_path / "map.csv", tmp_path / "geom.csv"
    pd.DataFrame(rows).to_csv(event_csv, index=False)
    pd.DataFrame(mapping).to_csv(map_csv, index=False)
    pd.DataFrame(geometry).to_csv(geom_csv, index=False)
    result, receipt = run_audit(event_csv, map_csv, geom_csv)
    assert len(result) == 2
    assert receipt["stratum_count"] == 1
    stratum = result.loc[result.scope == "stratum"].iloc[0]
    assert stratum.sampling_phase_partial_r2 > 0.9
    assert stratum.sampling_phase_cyclic_p <= 0.02
    assert receipt["safety"] == {"raw_recording_read": False, "gpu_used": False}
