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
    assert set(NAMED_CONFIGS) == {"rescue", "legacy_style", "nonrigid"}


def test_nonrigid_turns_on_datashift_without_touching_thresholds():
    p = NONRIGID.params()
    assert p["do_correction"] is True and p["nblocks"] == 6
    assert p["Th_universal"] == RESCUE.params()["Th_universal"]
    assert p["Th_learned"] == RESCUE.params()["Th_learned"]
    assert NONRIGID.digest not in {RESCUE.digest, LEGACY_STYLE.digest}


def test_run_sorter_config_refuses_mnt():
    with pytest.raises(ValueError, match="/mnt"):
        run_sorter_config("/some/recording", "/mnt/out", RESCUE)
