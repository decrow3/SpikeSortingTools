import pandas as pd

from testing.luke_imec1_two_bank_proposal_audit_v1 import exact_overlap, p90_span


def test_p90_span():
    values = pd.Series(range(101))
    assert p90_span(values) == 90


def test_exact_overlap_counts_source_events_once():
    assert exact_overlap([1.0, 2.0, 3.0], [1.0, 1.000001, 4.0]) == 1
