import numpy as np

from testing.luke_cluster553_local_template_rescue import (
    deduplicate_proposals,
    waveform_scores,
)


def test_deduplicate_proposals_keeps_strongest_then_time_orders():
    result = deduplicate_proposals(
        np.array([100, 105, 200]), np.array([3.0, 5.0, 4.0]), refractory=10
    )
    assert result.tolist() == [105, 200]


def test_waveform_scores_prefers_matching_shape():
    template = np.array([[-1.0, 0.0], [-3.0, 1.0], [-1.0, 0.0]])
    waves = np.stack([template, -template])
    cosine, score = waveform_scores(waves, template, np.ones(2), max_lag=0)
    assert cosine[0] > 0.99
    assert cosine[1] < -0.99
    assert score[0] > score[1]
