import math

import pytest

from testing.ladder_sorter import (
    LEGACY_STYLE,
    NAMED_CONFIGS,
    NONRIGID,
    RESCUE,
    SorterConfig,
    run_sorter_config,
)


def test_rescue_config_is_the_unmodified_baseline():
    assert RESCUE.overrides == {}
    base = RESCUE.params()
    assert base["do_correction"] is False
    assert math.isinf(base["artifact_threshold"])


def test_legacy_style_flips_only_motion_and_thresholds():
    p = LEGACY_STYLE.params()
    assert p["do_correction"] is True and p["nblocks"] == 1
    assert p["Th_universal"] == 9 and p["Th_learned"] == 8
    # everything else stays at the rescue baseline
    assert p["do_CAR"] == RESCUE.params()["do_CAR"]
    assert p["duplicate_spike_ms"] == RESCUE.params()["duplicate_spike_ms"]


def test_digest_is_stable_and_config_sensitive():
    assert LEGACY_STYLE.digest == SorterConfig("legacy_style", dict(LEGACY_STYLE.overrides)).digest
    assert LEGACY_STYLE.digest != RESCUE.digest
    assert SorterConfig("x", {"Th_learned": 8}).digest != SorterConfig("x", {"Th_learned": 7}).digest


def test_named_configs_registry():
    assert set(NAMED_CONFIGS) == {"rescue", "legacy_style", "rescue_rigid", "nonrigid"}


def test_nonrigid_turns_on_datashift_without_touching_thresholds():
    p = NONRIGID.params()
    assert p["do_correction"] is True and p["nblocks"] == 6
    assert p["Th_universal"] == RESCUE.params()["Th_universal"]
    assert p["Th_learned"] == RESCUE.params()["Th_learned"]
    assert NONRIGID.digest not in {RESCUE.digest, LEGACY_STYLE.digest}


def test_run_sorter_config_refuses_mnt():
    with pytest.raises(ValueError, match="/mnt"):
        run_sorter_config("/some/recording", "/mnt/out", RESCUE)


def test_rescue_rigid_isolates_correction_from_thresholds():
    """LEGACY_STYLE moves correction and thresholds together; this must not."""
    from testing.ladder_sorter import LEGACY_STYLE, RESCUE, RESCUE_RIGID

    rigid, rescue = RESCUE_RIGID.params(), RESCUE.params()
    assert rigid["do_correction"] is True and rigid["nblocks"] == 1
    for threshold in ("Th_universal", "Th_learned"):
        assert rigid[threshold] == rescue[threshold]      # unchanged vs rescue
        assert rigid[threshold] != LEGACY_STYLE.params()[threshold]
    assert RESCUE_RIGID.digest != RESCUE.digest != LEGACY_STYLE.digest


def test_rescue_requests_nblocks_1_but_must_resolve_to_0():
    """The requested dict proves nothing: KS4 zeroes nblocks when do_correction is off."""
    from testing.ladder_sorter import EXPECTED_EFFECTIVE, RESCUE

    assert RESCUE.params()["nblocks"] == 1
    assert RESCUE.params()["do_correction"] is False
    assert EXPECTED_EFFECTIVE["rescue"]["effective_nblocks"] == 0


def test_check_effective_settings_fails_closed_on_a_silent_downgrade():
    import pytest

    from testing.ladder_sorter import EXPECTED_EFFECTIVE, check_effective_settings

    good = dict(EXPECTED_EFFECTIVE["rescue_rigid"])
    assert check_effective_settings("rescue_rigid", good) == good
    downgraded = {**good, "effective_nblocks": 0}   # correction silently off
    with pytest.raises(RuntimeError, match="effective_nblocks"):
        check_effective_settings("rescue_rigid", downgraded)
