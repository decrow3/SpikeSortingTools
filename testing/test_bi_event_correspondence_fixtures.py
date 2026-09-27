from testing.bi_event_correspondence_fixtures import fixture_contract, run_contract


def test_bi_event_correspondence_fixture_contract():
    result = run_contract(fixture_contract())
    assert result["all_pass"]
    assert len(result["fixtures"]) == 5
    assert result["shifted_common_support"]["pass"]
