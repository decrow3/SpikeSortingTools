import numpy as np
import pytest

from testing.ladder_score import truth_digest
from testing.luke_c2_train_sentinel import CANDIDATES, SENTINEL, output_root, realisations

FS = 29999.759166666667


def test_only_the_reference_changes_the_event_count():
    """Count must vary on exactly one axis, or count and composition confound."""
    trains = realisations(FS)
    assert trains["full_708"].size == 708
    others = {n: t.size for n, t in trains.items() if n != "full_708"}
    assert set(others.values()) == {687}


def test_phase_variant_removes_the_same_events_at_shifted_times():
    """boundary vs phase isolates absolute timing from which events were cut."""
    trains = realisations(FS)
    boundary, phase = trains["boundary_687"], trains["phase_687"]
    assert boundary.size == phase.size
    offsets = np.unique(phase - boundary)
    assert offsets.size == 1 and offsets[0] > 0        # a pure shift
    assert not np.array_equal(boundary, phase)


def test_deletion_variants_are_genuinely_different_trains():
    trains = realisations(FS)
    names = ["boundary_687", "random_687_s1", "random_687_s2", "random_687_s3",
             "uniform_687"]
    digests = {truth_digest({"inj0": trains[n]}) for n in names}
    assert len(digests) == len(names)


def test_uniform_deletion_is_spread_not_clustered():
    """The point of the uniform arm: contrast with the clustered boundary cut."""
    trains = realisations(FS)
    full = set(trains["full_708"].tolist())
    removed = np.array(sorted(full - set(trains["uniform_687"].tolist())))
    gaps = np.diff(removed)
    assert gaps.std() / gaps.mean() < 0.1              # evenly spread
    boundary_removed = np.array(sorted(full - set(trains["boundary_687"].tolist())))
    assert np.diff(boundary_removed).std() > gaps.std()  # clustered by comparison


def test_realisations_are_frozen_and_reproducible():
    assert truth_digest({"inj0": realisations(FS)["random_687_s2"]}) == \
           truth_digest({"inj0": realisations(FS)["random_687_s2"]})


def test_sentinels_are_the_donors_that_actually_moved():
    assert set(SENTINEL["donors"]) == {"D02", "D04", "D07", "D10", "D14"}
    assert set(SENTINEL["why_these"]) == set(SENTINEL["donors"])


def test_correction_is_off_and_this_is_labelled_a_diagnostic():
    assert all(c.overrides["do_correction"] is False for c in CANDIDATES)
    assert "not a ranking" in SENTINEL["status"]


def test_never_writes_under_mnt():
    with pytest.raises(ValueError, match="under /mnt"):
        output_root("/mnt/NPX/nope")
