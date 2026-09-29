from testing.luke_cluster553_template_alignment_audit import alignment_verdict


def test_alignment_verdict_requires_frozen_target_acceptance():
    rows = [
        {"maximum_lag_samples": 2, "target_acceptance": 0.2},
        {"maximum_lag_samples": 5, "target_acceptance": 0.8},
    ]
    assert alignment_verdict(rows, 0.8) == ("alignment_followup_supported", [5])
    assert alignment_verdict(rows, 0.81) == ("alignment_followup_not_supported", [])
