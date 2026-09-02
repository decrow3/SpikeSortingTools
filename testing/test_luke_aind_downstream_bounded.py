import json
from pathlib import Path

from testing.luke_aind_downstream_bounded import (
    canonical_digest,
    load_config,
    plan_payload,
    presence_fraction,
    valid_sample_mask,
)

import numpy as np


CONFIG = Path(__file__).resolve().parents[1] / "testing/configs/luke_aind_downstream_bounded_v1.json"


def test_frozen_plan_has_three_conditions_across_both_probes_and_three_windows():
    plan = plan_payload(load_config(CONFIG))
    assert plan["n_sorts"] == 18
    assert plan["n_prepared_recordings"] == 12
    assert plan["sealed_events_per_probe"] == 360
    assert {job["kilosort_do_CAR"] for job in plan["jobs"]} == {True, False}


def test_config_digest_is_canonical_and_stable_to_key_order():
    config = load_config(CONFIG)
    reversed_config = dict(reversed(list(config.items())))
    assert canonical_digest(config) == canonical_digest(reversed_config)
    assert canonical_digest(config) == canonical_digest(json.loads(CONFIG.read_text()))


def test_valid_sample_mask_uses_half_open_recording_bounds():
    times = np.array([-1, 0, 9, 10])
    assert valid_sample_mask(times, 10).tolist() == [False, True, True, False]


def test_presence_fraction_uses_equal_bins_without_terminal_sliver():
    # Eight samples spanning a nominal eight-bin window remain fully present
    # even when the extracted recording has one extra boundary frame.
    times = np.arange(8, dtype=np.int64) * 900_000 + 10
    assert presence_fraction(times, 7_200_001, 8) == 1.0
