import pytest


def test_tracked_hold_blocks_full_session_entrypoint(tmp_path):
    from testing.luke_full_session_rigid import assert_not_on_hold
    hold = tmp_path / "configs/luke_full_session_rigid.HOLD.json"
    hold.parent.mkdir(parents=True)
    hold.write_text("{}\n")
    with pytest.raises(RuntimeError, match="user-requested hold"):
        assert_not_on_hold(tmp_path)


def test_full_session_entrypoint_allows_absent_hold(tmp_path):
    from testing.luke_full_session_rigid import assert_not_on_hold
    assert_not_on_hold(tmp_path)
