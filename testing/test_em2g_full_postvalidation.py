from pathlib import Path

from testing.em2g_full_postvalidation import ROOT, step_commands


def test_postvalidation_plan_is_complete_sequential_and_nonwaiting():
    commands = step_commands(Path("/venv/python"), Path("/output"))
    names = [name for name, _ in commands]
    assert names == [
        "slice_full_sort",
        "prepare_window_packets",
        "w2_scorecard",
        "w2_event_overlap",
        "w3_scorecard",
        "w3_event_overlap",
        "full_session_descriptive",
    ]
    flattened = " ".join(part for _, command in commands for part in command)
    assert "sleep" not in flattened
    assert "wait" not in flattened
    assert str(ROOT / "testing/em2g_slice_full_sort.py") in flattened
    assert str(ROOT / "testing/em2h_full_session_descriptive.py") in flattened
    assert all(command[0] == "/venv/python" for _, command in commands)
