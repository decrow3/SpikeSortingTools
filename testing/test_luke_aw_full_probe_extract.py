import numpy as np

from testing.luke_aw_full_probe_extract import (
    bounded_read_preflight,
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
        def get_num_channels(self):
            return 2

        def get_num_frames(self):
            return 100

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


def test_trace_frame_chunk_rejects_large_request_before_recording_read():
    class Recorder:
        called = False

        def get_num_channels(self):
            return 384

        def get_num_frames(self):
            return 100_000

        def get_traces(self, *args, **kwargs):
            self.called = True
            raise AssertionError("guard did not fail before voltage read")

    rec = Recorder()
    with np.testing.assert_raises(MemoryError):
        trace_frame_chunk(rec, 0, 30_001)
    assert rec.called is False


def test_trace_frame_chunk_rejects_wrong_real_shape():
    class Recorder:
        def get_num_channels(self):
            return 2

        def get_num_frames(self):
            return 100

        def get_traces(self, *args, **kwargs):
            return np.zeros((11, 2), dtype=np.float32)

    with np.testing.assert_raises(RuntimeError):
        trace_frame_chunk(Recorder(), 11, 23, channel_ids=["AP0", "AP1"])


def test_bounded_read_preflight_reports_forbidden_full_allocation():
    receipt = bounded_read_preflight(
        {
            "sampling_frequency": 29999.759166666667,
            "start_frame": 26999783,
            "end_frame": 37199701,
            "source_channel_ids": [f"AP{i}" for i in range(384)],
        }
    )
    assert receipt["status"] == "pass_before_voltage_open"
    assert receipt["window_frames"] == 10_199_918
    assert receipt["maximum_explicit_audit_request_frames"] == 30_000
    assert receipt["maximum_explicit_audit_request_float32_bytes"] == 46_080_000
    assert receipt["full_float32_window_bytes_forbidden"] == 15_667_074_048
