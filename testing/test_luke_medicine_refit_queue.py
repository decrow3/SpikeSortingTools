from testing.luke_medicine_rigid_queue import dependency_decision


def test_refit_waits_for_live_sort_even_with_stale_success():
    state = {"ActiveState": "active", "process_exists": True}
    assert dependency_decision(state, {"state": "complete", "returncode": 0},
                               {"SERVICE_RESULT": "success", "EXIT_STATUS": "0"}) == "wait"


def test_refit_releases_only_after_successful_terminal_receipts():
    state = {"ActiveState": "inactive", "process_exists": False}
    assert dependency_decision(state, {"state": "complete", "returncode": 0},
                               {"SERVICE_RESULT": "success", "EXIT_STATUS": "0"}) == "ready"
