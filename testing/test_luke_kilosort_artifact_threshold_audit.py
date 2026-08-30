import numpy as np
import pandas as pd

from testing import luke_kilosort_artifact_threshold_audit as audit


def test_padded_traces_repeats_edges():
    class Recording:
        def get_num_samples(self):
            return 5

        def get_traces(self, start_frame, end_frame):
            return np.arange(start_frame, end_frame, dtype=np.float32)[:, None]

    result = audit.padded_traces(Recording(), 0, 3, 2)
    assert result[:, 0].tolist() == [0, 0, 0, 1, 2, 3, 4]


def test_threshold_summary_counts_erased_neural_events():
    batches = pd.DataFrame(
        {"window": ["neutral_template"] * 2, "batch_index": [0, 1], "max_abs_post_car_highpass": [99, 201]}
    )
    events = pd.DataFrame(
        {"window": ["neutral_template"] * 2, "batch_index": [0, 1], "review_label": ["neural", "neural"]}
    )
    # Supply an empty second window so the shared summarizer can run.
    batches = pd.concat([batches, pd.DataFrame({"window": ["pathological"], "batch_index": [0], "max_abs_post_car_highpass": [0]})])
    events = pd.concat([events, pd.DataFrame({"window": ["pathological"], "batch_index": [0], "review_label": ["artifact"]})])
    result = audit.threshold_summary(batches, events)
    row = result[(result.window == "neutral_template") & (result.artifact_threshold_counts == 200)].iloc[0]
    assert row.rejected_batches == 1
    assert row.reviewed_neural_events_erased == 1
