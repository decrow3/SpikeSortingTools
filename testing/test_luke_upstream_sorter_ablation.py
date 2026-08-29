from testing.luke_upstream_sorter_ablation import CONDITIONS, parse_extraction_counts


def test_parse_extraction_counts_uses_last_complete_detection_pair():
    text = """
    [INFO] - 304775 spikes extracted in 10s
    [INFO] - 1298260 spikes extracted in 20s
    """
    assert parse_extraction_counts(text) == (304775, 1298260)


def test_parse_extraction_counts_returns_none_for_incomplete_log():
    assert parse_extraction_counts("100 spikes extracted") == (None, None)


def test_conditioning_ablation_contains_single_factor_controls():
    by_name = {condition.name: condition for condition in CONDITIONS}
    assert by_name["bandpass_no_reference"].stage == "interpolated_bandpass"
    assert by_name["global_reference"].stage == "global_reference_control"
    assert by_name["local_reference_no_ks_car"].do_ks_car is False
    assert by_name["single_ks_preprocessing"].stage == "interpolated_unfiltered"
