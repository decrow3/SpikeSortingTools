import numpy as np

from testing.luke_aw_full_probe_extract import (
    centered_cosine,
    evenly_spaced,
    exact_map,
    remap,
    support_rows,
    trace_frame_chunk,
)


def test_evenly_spaced_is_deterministic_unique():
    source = np.arange(900, dtype=np.int64)
    got = evenly_spaced(source, 500)
    assert got.size == np.unique(got).size == 500
    np.testing.assert_array_equal(got, evenly_spaced(source, 500))


def test_exact_map_and_remap():
    geom = np.c_[np.zeros(6), np.arange(6) * 40.0]
    mapping = exact_map(geom, -40.0)
    template = np.arange(18, dtype=np.float32).reshape(3, 6)
    moved = remap(template, mapping)
    np.testing.assert_array_equal(moved[:, :5], template[:, 1:])
    np.testing.assert_array_equal(moved[:, 5], 0)


def test_support_denominator_is_total_full_probe_energy():
    geom = np.c_[np.zeros(8), np.arange(8) * 40.0]
    template = np.zeros((3, 8), dtype=np.float32)
    template[1, 1] = 1
    template[1, 6] = 3
    rows = support_rows(template[None], geom, np.arange(4, 8), np.array([7]))
    zero = next(r for r in rows if r["state_um"] == 0)
    assert np.isclose(zero["retained_energy_fraction"], 0.9)
    assert zero["support_pass"] is False


def test_centered_cosine_identical():
    x = np.arange(20, dtype=np.float32).reshape(4, 5)
    assert np.isclose(centered_cosine(x, x), 1.0)


def test_trace_frame_chunk_uses_named_frame_bounds():
    class Recorder:
        def get_traces(self, *args, **kwargs):
            assert args == ()
            assert kwargs == {
                "start_frame": 11,
                "end_frame": 23,
                "channel_ids": ["AP0", "AP1"],
            }
            return np.zeros((12, 2), dtype=np.float32)

    got = trace_frame_chunk(Recorder(), 11, 23, channel_ids=["AP0", "AP1"])
    assert got.shape == (12, 2)
