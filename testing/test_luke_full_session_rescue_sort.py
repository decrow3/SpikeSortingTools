from testing import luke_full_session_rescue_sort as rescue


def test_full_session_candidate_is_single_pass_claim_off():
    assert rescue.CLAIM_OFF.claim_ms == 0.0
    assert rescue.CLAIM_OFF.claim_um == 0.0
    assert rescue.RECORDING_PATH.name == "recording"
    assert rescue.SORT_PATH.name == "sort"
