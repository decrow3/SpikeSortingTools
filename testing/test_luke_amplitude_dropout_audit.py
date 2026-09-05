import dataclasses
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from testing.luke_amplitude_dropout_audit import (
    CASE_STATUS_STABLE,
    CASE_STATUS_UNSTABLE,
    CASE_WINDOWS_COLUMNS,
    CASE_WINDOW_REPRODUCED,
    CASE_WINDOW_REPRODUCTION_MISMATCH,
    PRODUCTION_MAX_ISI_S,
    REPRODUCTION_ATOL_PP,
    SCHEMA,
    SELECTION_CONSTANT_KEYS,
    AuditConfig,
    CachedTruncationQC,
    CuratedArrays,
    SelectionConstants,
    SortConfig,
    STATUS_BOUNDARY_PINNED,
    STATUS_FINITE_INTERIOR,
    STATUS_INVALID_INPUT,
    STATUS_NONFINITE_FIT,
    STATUS_NO_FIT,
    WINDOWS_COLUMNS,
    _reject_unsafe_out_root,
    _validate_kept_spikes,
    audit_source_files,
    build_windows_table,
    canonical_selection_digest,
    classify_control_run,
    classify_failure_run,
    cluster_amplitude_sequence,
    exact_indexing_eligibility,
    historical_exact_counts,
    historical_exact_fit,
    load_attested_selection,
    load_cached_truncation_qc,
    load_config,
    load_curated_arrays,
    main,
    _parse_windows_bytes,
    parse_selection_constants,
    read_attested_windows,
    run_inspect,
    run_inventory,
    run_select,
    select_cases,
    sha256_file,
    validate_recording_metadata,
    window_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = REPO_ROOT / "testing/configs/luke_amplitude_dropout_audit_v1.json"

# The frozen selection block every valid config must carry. Tests read it from
# the real config so a drift between the shipped constants and the ones the
# selection tests pin down cannot go unnoticed.
SELECTION_BLOCK = json.loads(REAL_CONFIG.read_text())["selection"]
CONSTANTS = parse_selection_constants(SELECTION_BLOCK)


# --------------------------------------------------------------------------- #
# fixtures: a synthetic curated sorter-output directory
# --------------------------------------------------------------------------- #
def _kept_positions(kept: np.ndarray, n: int) -> np.ndarray:
    """Row positions `full_st[kept]` actually selects, for building a fixture's
    spike_times/spike_clusters to match. Falls back to identity for a
    deliberately-malformed `kept` (wrong-length bool / out-of-range int) --
    those fixtures only need to not crash; the loader is expected to reject
    `kept` itself before ever comparing array lengths."""
    kept = np.asarray(kept)
    if kept.dtype == np.bool_ and kept.shape[0] == n:
        return np.flatnonzero(kept)
    if np.issubdtype(kept.dtype, np.integer) and kept.size and kept.min() >= 0 and kept.max() < n:
        return kept
    return np.arange(n)


def _write_curated(tmp_path: Path, *, n: int = 20, amplitude_offset: float = 0.0,
                    times_dtype=np.int64, kept=None, corrupt_times: bool = False) -> Path:
    """Write a minimal curated directory with n full-table spikes across 2
    clusters, `kept` of them retained in `spike_times.npy`/`spike_clusters.npy`
    (mirroring `full_st.npy` + `kept_spikes.npy` -> `spike_times.npy` in a real
    curated output).

    full_st columns are [sample, template, amplitude]; amplitudes.npy is
    written with a DIFFERENT value so a loader bug that reads it instead of
    full_st[:, 2] is caught (prescription clause 1).
    """
    d = tmp_path / "curated"
    d.mkdir(parents=True, exist_ok=True)
    full_times = np.arange(n, dtype=np.int64) * 10
    full_clusters = np.array([0, 1] * (n // 2) + [0] * (n % 2), dtype=np.int64)[:n]
    st_amplitude = np.arange(n, dtype=np.float64) + 100.0 + amplitude_offset
    full_st = np.stack([full_times.astype(np.float64), np.zeros(n), st_amplitude], axis=1)

    if kept is None:
        kept = np.ones(n, dtype=bool)
    np.save(d / "full_st.npy", full_st)
    np.save(d / "kept_spikes.npy", kept)

    positions = _kept_positions(kept, n)
    stored_times = full_times[positions].copy()
    if corrupt_times:
        stored_times = stored_times.astype(np.float64) + 0.5  # non-integer
    np.save(d / "spike_times.npy", stored_times.astype(times_dtype) if not corrupt_times else stored_times)
    np.save(d / "spike_clusters.npy", full_clusters[positions])
    # deliberately wrong: production must never read this column for amplitude
    np.save(d / "amplitudes.npy", st_amplitude[positions] * -1.0 - 999.0)
    return d


def _write_cached_qc(qc_dir: Path, *, cid: np.ndarray, window_blocks: np.ndarray,
                      popts: np.ndarray, mpcts: np.ndarray) -> Path:
    out = qc_dir / "amp_truncation"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "truncation_qc.npz", cid=cid, window_blocks=window_blocks, popts=popts, mpcts=mpcts)
    return qc_dir


# --------------------------------------------------------------------------- #
# layer 1: loader / window-table normalization
# --------------------------------------------------------------------------- #
def test_load_curated_arrays_uses_full_st_amplitude_not_amplitudes_npy(tmp_path):
    d = _write_curated(tmp_path)
    curated = load_curated_arrays("s1", d)
    # amplitudes.npy was written as -(st_amplitude) - 999; loader must not match that.
    assert np.all(curated.amplitudes > 0)
    assert curated.amplitudes.min() >= 100.0


def test_load_curated_arrays_rejects_equal_length_misaligned_times(tmp_path):
    d = _write_curated(tmp_path)
    # Corrupt spike_times.npy so it no longer matches full_st[kept][:, 0].
    bad_times = np.load(d / "spike_times.npy")
    bad_times = bad_times + 1
    np.save(d / "spike_times.npy", bad_times)
    with pytest.raises(ValueError, match="do not match"):
        load_curated_arrays("s1", d)


def test_load_curated_arrays_rejects_boolean_kept_spikes_wrong_length(tmp_path):
    n = 20
    kept = np.ones(n - 1, dtype=bool)  # wrong length
    d = _write_curated(tmp_path, n=n, kept=kept)
    with pytest.raises(ValueError, match="kept_spikes"):
        load_curated_arrays("s1", d)


def test_load_curated_arrays_rejects_out_of_range_integer_kept_spikes(tmp_path):
    n = 20
    kept = np.arange(n, dtype=np.int64)
    kept[-1] = n  # out of range
    d = _write_curated(tmp_path, n=n, kept=kept)
    with pytest.raises(ValueError, match="out of range"):
        load_curated_arrays("s1", d)


def test_load_curated_arrays_accepts_integer_index_kept_spikes(tmp_path):
    n = 20
    kept_idx = np.arange(0, n, 2, dtype=np.int64)  # keep every other spike
    d = _write_curated(tmp_path, n=n, kept=kept_idx)
    curated = load_curated_arrays("s1", d)
    assert curated.times.size == kept_idx.size


def test_load_curated_arrays_rejects_sample_second_confusion(tmp_path):
    d = _write_curated(tmp_path, corrupt_times=True)
    with pytest.raises(ValueError, match="non-integer"):
        load_curated_arrays("s1", d)


def test_load_curated_arrays_rejects_large_magnitude_fractional_times(tmp_path):
    """np.allclose's relative tolerance would accept 100000.25 ~ 100000 for
    large sample counts; exact integrality must not (prescription: reject
    invalid arrays rather than silently rounding)."""
    d = _write_curated(tmp_path, n=10)
    times = np.load(d / "spike_times.npy").astype(np.float64)
    full_st = np.load(d / "full_st.npy")
    times[0] = 100000.25
    full_st[0, 0] = 100000.25
    np.save(d / "spike_times.npy", times)
    np.save(d / "full_st.npy", full_st)
    with pytest.raises(ValueError, match="non-integer"):
        load_curated_arrays("s1", d)


def test_load_curated_arrays_rejects_mismatched_full_st_time_that_truncates_to_match(tmp_path):
    """full_st[:, 0] = 10.9 must not be accepted just because int64 truncation
    happens to equal spike_times.npy's 10."""
    d = _write_curated(tmp_path, n=10)
    full_st = np.load(d / "full_st.npy")
    full_st[0, 0] += 0.9
    np.save(d / "full_st.npy", full_st)
    with pytest.raises(ValueError, match="non-integer"):
        load_curated_arrays("s1", d)


def test_load_curated_arrays_rejects_fractional_cluster_labels(tmp_path):
    """Fractional cluster labels must be rejected, not silently collapsed by
    a later int(c) cast (two labels 1.2 and 1.8 must not both become id 1)."""
    d = _write_curated(tmp_path, n=4)
    np.save(d / "spike_clusters.npy", np.array([1.2, 1.2, 1.8, 1.8]))
    with pytest.raises(ValueError, match="non-integer"):
        load_curated_arrays("s1", d)


def test_load_curated_arrays_rejects_negative_offset_samples(tmp_path):
    d = _write_curated(tmp_path)
    times = np.load(d / "spike_times.npy")
    times[0] = -5
    np.save(d / "spike_times.npy", times)
    full_st = np.load(d / "full_st.npy")
    full_st[0, 0] = -5
    np.save(d / "full_st.npy", full_st)
    with pytest.raises(ValueError, match="negative"):
        load_curated_arrays("s1", d)


def test_load_curated_arrays_flags_unordered_source(tmp_path):
    d = _write_curated(tmp_path, n=10)
    times = np.load(d / "spike_times.npy")
    full_st = np.load(d / "full_st.npy")
    # swap two rows to break monotonicity while keeping times<->full_st aligned
    times[3], times[4] = times[4], times[3]
    full_st[3, 0], full_st[4, 0] = full_st[4, 0], full_st[3, 0]
    np.save(d / "spike_times.npy", times)
    np.save(d / "full_st.npy", full_st)
    curated = load_curated_arrays("s1", d)
    assert curated.was_time_ordered is False
    # the loader still stable-sorts for general use...
    assert np.all(np.diff(curated.times) >= 0)


def test_load_curated_arrays_noncontiguous_cluster_ids(tmp_path):
    d = _write_curated(tmp_path, n=10)
    clusters = np.array([5, 5, 5, 5, 5, 12, 12, 12, 12, 12], dtype=np.int64)
    np.save(d / "spike_clusters.npy", clusters)
    curated = load_curated_arrays("s1", d)
    assert set(np.unique(curated.clusters).tolist()) == {5, 12}


def test_build_windows_table_status_classification(tmp_path):
    d = _write_curated(tmp_path, n=10)
    curated = load_curated_arrays("s1", d)
    # cluster 0 has 5 spikes (positions within cluster 0..4); cluster 1 has 5.
    cid = np.array([0, 0, 0, 1], dtype=np.float64)
    window_blocks = np.array([
        [0, 3],   # finite interior (valid, i1 < cluster_len for cluster 0 which has 5 members)
        [0, 4],   # invalid_input: i1 == cluster_len (5 members -> valid indices 0..4, so this is OK actually
        [2, 1],   # invalid_input: i1 <= i0
        [0, 4],   # cluster 1 has only 5 members too; boundary-pinned example
    ], dtype=np.int64)
    popts = np.array([
        [10.0, 1.0, 1.0],
        [10.0, 1.0, 1.0],
        [10.0, 1.0, 1.0],
        [8.0, 0.0, 1.0],
    ])
    mpcts = np.array([12.5, float("nan"), 30.0, 50.0])
    cached_qc = CachedTruncationQC(sort_id="s1", cid=cid.astype(np.int64), window_blocks=window_blocks,
                                   popts=popts, mpcts=mpcts)
    table = build_windows_table(curated, cached_qc, fs=1000.0)

    by_row = {(int(r.cluster_id), int(r.source_row)): r for r in table.itertuples()}
    assert by_row[(0, 0)].status == STATUS_FINITE_INTERIOR
    assert by_row[(0, 1)].status == STATUS_NONFINITE_FIT  # nan mpct
    assert by_row[(0, 2)].status == STATUS_INVALID_INPUT  # i1 <= i0
    assert by_row[(1, 3)].status == STATUS_BOUNDARY_PINNED


def test_build_windows_table_missing_cluster_is_no_fit(tmp_path):
    d = _write_curated(tmp_path, n=10)
    curated = load_curated_arrays("s1", d)
    # cached QC only covers cluster 0; cluster 1 has zero cached windows.
    cached_qc = CachedTruncationQC(
        sort_id="s1",
        cid=np.array([0], dtype=np.int64),
        window_blocks=np.array([[0, 3]], dtype=np.int64),
        popts=np.array([[10.0, 1.0, 1.0]]),
        mpcts=np.array([12.5]),
    )
    table = build_windows_table(curated, cached_qc, fs=1000.0)
    row1 = table[table.cluster_id == 1].iloc[0]
    assert row1.status == STATUS_NO_FIT
    assert np.isnan(row1.missing_pct)


def test_build_windows_table_unknown_cached_cluster_id_is_not_dropped(tmp_path):
    """A cached row whose cluster_id is absent from the curated arrays must
    still appear in the table (as invalid_input), not silently vanish because
    the row-emission loop only visits curated cluster ids."""
    d = _write_curated(tmp_path, n=10)
    curated = load_curated_arrays("s1", d)
    cached_qc = CachedTruncationQC(
        sort_id="s1",
        cid=np.array([0, 99], dtype=np.int64),  # 99 does not exist in curated clusters
        window_blocks=np.array([[0, 3], [0, 3]], dtype=np.int64),
        popts=np.array([[10.0, 1.0, 1.0], [10.0, 1.0, 1.0]]),
        mpcts=np.array([12.5, 12.5]),
    )
    table = build_windows_table(curated, cached_qc, fs=1000.0)
    unknown_rows = table[table.cluster_id == 99]
    assert len(unknown_rows) == 1
    assert unknown_rows.iloc[0].status == STATUS_INVALID_INPUT
    assert "absent from curated" in unknown_rows.iloc[0].invalid_reason


def test_build_windows_table_rejects_window_spanning_a_gap(tmp_path):
    """A cached window whose stored [i0, i1] spans a >10s gap between
    consecutive spikes (e.g. because the curated arrays changed since the
    cache was written) must not be silently fit across that gap."""
    d = _write_curated(tmp_path, n=4)
    fs = 1000.0
    # cluster 0 spikes far enough apart in TIME (not index) to exceed 10s at fs=1000
    times = np.array([0, 1, 10_000 * fs + 2000, 10_000 * fs + 3000], dtype=np.int64)
    full_st = np.load(d / "full_st.npy")
    full_st[:, 0] = times.astype(np.float64)
    np.save(d / "full_st.npy", full_st)
    np.save(d / "spike_times.npy", times)
    np.save(d / "spike_clusters.npy", np.array([0, 0, 0, 0], dtype=np.int64))
    curated = load_curated_arrays("s1", d)

    cached_qc = CachedTruncationQC(
        sort_id="s1", cid=np.array([0], dtype=np.int64),
        window_blocks=np.array([[0, 3]], dtype=np.int64),
        popts=np.array([[10.0, 1.0, 1.0]]), mpcts=np.array([12.5]),
    )
    table = build_windows_table(curated, cached_qc, fs=fs, max_isi_s=10.0)
    row = table.iloc[0]
    assert row.status == STATUS_INVALID_INPUT
    assert "gap" in row.invalid_reason


def test_build_windows_table_rejects_unordered_source(tmp_path):
    d = _write_curated(tmp_path, n=10)
    times = np.load(d / "spike_times.npy")
    full_st = np.load(d / "full_st.npy")
    times[3], times[4] = times[4], times[3]
    full_st[3, 0], full_st[4, 0] = full_st[4, 0], full_st[3, 0]
    np.save(d / "spike_times.npy", times)
    np.save(d / "full_st.npy", full_st)
    curated = load_curated_arrays("s1", d)
    cached_qc = CachedTruncationQC(
        sort_id="s1", cid=np.array([0], dtype=np.int64),
        window_blocks=np.array([[0, 3]], dtype=np.int64),
        popts=np.array([[10.0, 1.0, 1.0]]), mpcts=np.array([12.5]),
    )
    with pytest.raises(ValueError, match="not already time-ordered"):
        build_windows_table(curated, cached_qc, fs=1000.0)


def test_load_cached_truncation_qc_rejects_mismatched_row_counts(tmp_path):
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir,
        cid=np.array([0.0, 1.0]),
        window_blocks=np.array([[0, 3], [0, 3]]),
        popts=np.array([[1.0, 1.0, 1.0]]),  # wrong row count
        mpcts=np.array([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="popts shape"):
        load_cached_truncation_qc("s1", qc_dir)


def test_load_cached_truncation_qc_rejects_non_integer_cid(tmp_path):
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir,
        cid=np.array([0.5]),
        window_blocks=np.array([[0, 3]]),
        popts=np.array([[1.0, 1.0, 1.0]]),
        mpcts=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="non-integer"):
        load_cached_truncation_qc("s1", qc_dir)


def test_load_cached_truncation_qc_does_not_reject_semantically_invalid_bounds(tmp_path):
    """i1 <= i0 is a per-row `invalid_input` case for build_windows_table to
    classify (with all rows kept), not a reason to abort loading the whole
    cache -- see test_build_windows_table_status_classification."""
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir,
        cid=np.array([0.0]),
        window_blocks=np.array([[5, 5]]),
        popts=np.array([[1.0, 1.0, 1.0]]),
        mpcts=np.array([1.0]),
    )
    cached = load_cached_truncation_qc("s1", qc_dir)
    assert cached.window_blocks.tolist() == [[5, 5]]


def test_load_cached_truncation_qc_rejects_large_magnitude_fractional_cid(tmp_path):
    """np.allclose's relative tolerance would accept 100000.25 ~ 100000; exact
    integrality must not."""
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir,
        cid=np.array([100000.25]),
        window_blocks=np.array([[0, 3]]),
        popts=np.array([[1.0, 1.0, 1.0]]),
        mpcts=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="non-integer"):
        load_cached_truncation_qc("s1", qc_dir)


def test_load_cached_truncation_qc_rejects_large_magnitude_fractional_window_blocks(tmp_path):
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir,
        cid=np.array([0.0]),
        window_blocks=np.array([[0, 100000.25]]),
        popts=np.array([[1.0, 1.0, 1.0]]),
        mpcts=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="non-integer"):
        load_cached_truncation_qc("s1", qc_dir)


# --------------------------------------------------------------------------- #
# layer 2: historical / exact replay
# --------------------------------------------------------------------------- #
def test_historical_exact_counts_1000_window():
    hist, exact = historical_exact_counts(0, 999)  # nominal 1000-spike window
    assert hist == 999
    assert exact == 1000


def test_historical_exact_counts_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        historical_exact_counts(5, 5)
    with pytest.raises(ValueError):
        historical_exact_counts(-1, 3)


def test_historical_exact_fit_final_amplitude_only_in_exact():
    rng = np.random.default_rng(0)
    amps = np.sort(rng.normal(20.0, 3.0, size=1001))
    result = historical_exact_fit(amps, 0, 999)
    assert result["historical_count"] == 999
    assert result["exact_count"] == 1000
    # the historical slice must exclude amps[999]; the exact slice must include it
    assert not np.array_equal(amps[0:999], amps[0:1000])


def test_historical_exact_fit_calls_fit_amp_cdf_with_the_exact_slices(monkeypatch):
    """Spy on fit_amp_cdf directly: a bug that fits the historical slice twice
    (while still reporting the right counts) must fail this, unlike the
    input-agnostic size-based assertion above."""
    import testing.luke_amplitude_dropout_audit as mod

    amps = np.arange(1001, dtype=np.float64) + 1000.0  # amps[999] is distinctive
    seen = []
    real_fit_amp_cdf = mod.fit_amp_cdf

    def spy(a, x_min=None):
        seen.append(np.asarray(a).copy())
        return real_fit_amp_cdf(a, x_min=x_min)

    monkeypatch.setattr(mod, "fit_amp_cdf", spy)
    mod.historical_exact_fit(amps, 0, 999)

    assert len(seen) == 2
    hist_call, exact_call = seen
    assert np.array_equal(hist_call, amps[0:999])
    assert np.array_equal(exact_call, amps[0:1000])
    assert amps[999] not in hist_call
    assert amps[999] in exact_call


def test_historical_exact_fit_accepts_boundary_i1_at_last_valid_index():
    amps = np.arange(100, dtype=np.float64)  # valid indices 0..99
    result = historical_exact_fit(amps, 90, 99)  # exact slice [90:100) fits exactly
    assert result["exact_count"] == 10


def test_historical_exact_fit_rejects_overrun():
    amps = np.arange(100, dtype=np.float64)  # valid indices 0..99
    with pytest.raises(ValueError, match="exceeds cluster length"):
        historical_exact_fit(amps, 50, 100)  # i1 == size: exact slice would need index 100


def test_construct_windows_gap_over_10s_splits_the_block():
    from pipeline.truncation import construct_windows

    # block_a has 1500 spikes (one 1000-window fits, 500 leftover); block_b has
    # only 600 (below the 1000 threshold on its own). A >10s gap must split
    # them, so only block_a's window survives.
    block_a = np.arange(0, 1500) * 0.001
    block_b = block_a[-1] + 10.001 + np.arange(0, 600) * 0.001
    ts = np.concatenate([block_a, block_b])
    window_blocks, _ = construct_windows(ts, max_isi=10, spikes_per_window=1000)
    assert len(window_blocks) == 1


def test_construct_windows_exactly_10s_separation_does_not_split():
    from pipeline.truncation import construct_windows

    # Same sizes as above, but the gap is exactly 10s (not > max_isi), so the
    # two spans stay one 2100-spike block and yield two windows.
    block_a = np.arange(0, 1500) * 0.001
    block_b = block_a[-1] + 10.0 + np.arange(0, 600) * 0.001
    ts = np.concatenate([block_a, block_b])
    window_blocks, _ = construct_windows(ts, max_isi=10, spikes_per_window=1000)
    assert len(window_blocks) == 2


def test_construct_windows_leftover_tail_and_empty_input():
    from pipeline.truncation import construct_windows

    # 1999 spikes -> one 1000-window fits, 999 leftover (no second window)
    ts = np.arange(1999) * 0.001
    window_blocks, _ = construct_windows(ts, max_isi=10, spikes_per_window=1000)
    assert len(window_blocks) == 1

    # empty input
    window_blocks, _ = construct_windows(np.array([]), max_isi=10, spikes_per_window=1000)
    assert len(window_blocks) == 0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_load_config_real_file_has_two_sorts():
    cfg = load_config(REAL_CONFIG)
    assert cfg.schema == SCHEMA
    assert len(cfg.sorts) == 2
    ids = {s.sort_id for s in cfg.sorts}
    assert ids == {"rescue_luke0804_v2v1_g0_imec0", "legacy_luke0804_v2v1_g0_imec0"}


def test_audit_config_rejects_wrong_schema():
    with pytest.raises(ValueError, match="schema"):
        AuditConfig(schema="wrong", sorts=(
            SortConfig(sort_id="a", curated=Path("."), qc_dir=Path("."),
                       source_recording=Path("."), sampling_frequency_hz=1.0,
                       selected_start_sample=0, duration_s=1.0),
        ), selection=CONSTANTS)


def test_audit_config_rejects_duplicate_sort_id():
    entry = dict(curated=Path("."), qc_dir=Path("."), source_recording=Path("."),
                 sampling_frequency_hz=1.0, selected_start_sample=0, duration_s=1.0)
    with pytest.raises(ValueError, match="duplicate"):
        AuditConfig(schema=SCHEMA, sorts=(
            SortConfig(sort_id="a", **entry),
            SortConfig(sort_id="a", **entry),
        ), selection=CONSTANTS)


def test_load_config_rejects_non_integral_selected_start_sample(tmp_path):
    payload = {
        "schema": SCHEMA,
        "selection": SELECTION_BLOCK,
        "sorts": [{
            "sort_id": "a", "curated": ".", "qc_dir": ".", "source_recording": ".",
            "sampling_frequency_hz": 1000.0, "selected_start_sample": 123.7, "duration_s": 1.0,
        }],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be an integer"):
        load_config(config_path)


def test_load_config_accepts_large_int64_representable_selected_start_sample(tmp_path):
    """A round-trip through float() would lose precision on this value
    (9007199254740993 > 2**53); _exact_int_scalar must not do that."""
    big = 9007199254740993
    payload = {
        "schema": SCHEMA,
        "selection": SELECTION_BLOCK,
        "sorts": [{
            "sort_id": "a", "curated": ".", "qc_dir": ".", "source_recording": ".",
            "sampling_frequency_hz": 1000.0, "selected_start_sample": big, "duration_s": 1.0,
        }],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload))
    cfg = load_config(config_path)
    assert cfg.sorts[0].selected_start_sample == big


def test_sort_config_rejects_nan_duration():
    """A NaN duration_s must not silently pass `duration_s <= 0` (NaN
    comparisons are always False) and later be reported as validated."""
    with pytest.raises(ValueError, match="finite and positive"):
        SortConfig(sort_id="a", curated=Path("."), qc_dir=Path("."), source_recording=Path("."),
                   sampling_frequency_hz=1000.0, selected_start_sample=0, duration_s=float("nan"))


def test_sort_config_rejects_nan_sampling_frequency():
    with pytest.raises(ValueError, match="finite and positive"):
        SortConfig(sort_id="a", curated=Path("."), qc_dir=Path("."), source_recording=Path("."),
                   sampling_frequency_hz=float("nan"), selected_start_sample=0, duration_s=1.0)


def test_load_config_rejects_nan_duration_string(tmp_path):
    """`float("NaN")` succeeds where a numeric parse should have been
    rejected outright; SortConfig's isfinite check is the actual backstop."""
    payload = {
        "schema": SCHEMA,
        "selection": SELECTION_BLOCK,
        "sorts": [{
            "sort_id": "a", "curated": ".", "qc_dir": ".", "source_recording": ".",
            "sampling_frequency_hz": 1000.0, "selected_start_sample": 0, "duration_s": "NaN",
        }],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite and positive"):
        load_config(config_path)


def test_exact_int_array_rejects_complex_dtype():
    from testing.luke_amplitude_dropout_audit import _exact_int_array

    with pytest.raises(ValueError, match="complex"):
        _exact_int_array(np.array([1 + 1j]), "probe")


def test_exact_int_array_rejects_uint64_overflowing_int64():
    from testing.luke_amplitude_dropout_audit import _exact_int_array

    with pytest.raises(ValueError, match="int64 range"):
        _exact_int_array(np.array([2**64 - 1], dtype=np.uint64), "probe")


def test_exact_int_array_rejects_float_beyond_int64_range():
    from testing.luke_amplitude_dropout_audit import _exact_int_array

    with pytest.raises(ValueError, match="int64 range"):
        _exact_int_array(np.array([float(2**63)]), "probe")


def test_recording_metadata_rejects_fractional_manifest_frames(tmp_path):
    from testing.luke_amplitude_dropout_audit import _recording_metadata

    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "rescue_recording_manifest.json").write_text(json.dumps({
        "sampling_frequency_hz": 1000.0,
        "selected_start_frame": 0.9,
        "selected_end_frame": 5000.9,
    }))
    with pytest.raises(ValueError, match="must be an integer"):
        _recording_metadata(recording)


def test_validate_recording_metadata_checks_duration_against_rescue_manifest(tmp_path):
    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "rescue_recording_manifest.json").write_text(json.dumps({
        "sampling_frequency_hz": 1000.0,
        "selected_start_frame": 0,
        "selected_end_frame": 5000,
    }))
    good = SortConfig(sort_id="a", curated=Path("."), qc_dir=Path("."), source_recording=recording,
                      sampling_frequency_hz=1000.0, selected_start_sample=0, duration_s=5.0)
    check = validate_recording_metadata(good)
    assert check["duration_validated"] is True

    bad = SortConfig(sort_id="a", curated=Path("."), qc_dir=Path("."), source_recording=recording,
                     sampling_frequency_hz=1000.0, selected_start_sample=0, duration_s=999.0)
    with pytest.raises(ValueError, match="duration_s"):
        validate_recording_metadata(bad)

    bad_start = SortConfig(sort_id="a", curated=Path("."), qc_dir=Path("."), source_recording=recording,
                           sampling_frequency_hz=1000.0, selected_start_sample=7, duration_s=5.0)
    with pytest.raises(ValueError, match="selected_start_sample"):
        validate_recording_metadata(bad_start)


def test_validate_recording_metadata_real_rescue_config_is_authoritative():
    cfg = load_config(REAL_CONFIG)
    rescue = cfg.by_id("rescue_luke0804_v2v1_g0_imec0")
    if not (rescue.source_recording / "rescue_recording_manifest.json").exists():
        pytest.skip("Luke 20250804 /mnt recording manifest not available")
    check = validate_recording_metadata(rescue)
    assert check["duration_validated"] is True


# --------------------------------------------------------------------------- #
# path safety
# --------------------------------------------------------------------------- #
def test_reject_unsafe_out_root_under_mnt():
    with pytest.raises(ValueError, match="/mnt"):
        _reject_unsafe_out_root(Path("/mnt/NPX/scratch"), [])


def test_reject_unsafe_out_root_under_input_dir(tmp_path):
    input_dir = tmp_path / "curated"
    input_dir.mkdir()
    with pytest.raises(ValueError, match="input directory"):
        _reject_unsafe_out_root(input_dir / "nested_output", [input_dir])


def test_reject_unsafe_out_root_allows_sibling(tmp_path):
    input_dir = tmp_path / "curated"
    input_dir.mkdir()
    out_root = tmp_path / "audit_output"
    resolved = _reject_unsafe_out_root(out_root, [input_dir])
    assert resolved == out_root.resolve()


# --------------------------------------------------------------------------- #
# end-to-end inventory on synthetic fixtures
# --------------------------------------------------------------------------- #
def _write_config(tmp_path: Path, curated: Path, qc_dir: Path, source_recording: Path, *,
                  sampling_frequency_hz: float = 1000.0, duration_s: float = 1.0) -> Path:
    source_recording.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "selection": SELECTION_BLOCK,
        "sorts": [{
            "sort_id": "synthetic",
            "curated": str(curated),
            "qc_dir": str(qc_dir),
            "source_recording": str(source_recording),
            "sampling_frequency_hz": sampling_frequency_hz,
            "selected_start_sample": 0,
            "duration_s": duration_s,
        }],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload))
    return config_path


def test_run_inventory_end_to_end(tmp_path):
    curated = _write_curated(tmp_path, n=10)
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir,
        cid=np.array([0.0]),
        window_blocks=np.array([[0, 3]]),
        popts=np.array([[10.0, 1.0, 1.0]]),
        mpcts=np.array([12.5]),
    )
    config_path = _write_config(tmp_path, curated, qc_dir, tmp_path / "recording")
    out_root = tmp_path / "out"

    manifest = run_inventory(config_path, out_root)
    assert manifest["status"] == "complete"
    assert (out_root / "windows.csv").exists()
    table = pd.read_csv(out_root / "windows.csv")
    assert len(table) == 2  # cluster 0 (1 cached row) + cluster 1 (no_fit)


def test_run_inventory_refuses_existing_output(tmp_path):
    curated = _write_curated(tmp_path, n=10)
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir, cid=np.array([0.0]), window_blocks=np.array([[0, 3]]),
        popts=np.array([[10.0, 1.0, 1.0]]), mpcts=np.array([12.5]),
    )
    config_path = _write_config(tmp_path, curated, qc_dir, tmp_path / "recording")
    out_root = tmp_path / "out"
    run_inventory(config_path, out_root)
    with pytest.raises(RuntimeError, match="non-empty output root"):
        run_inventory(config_path, out_root)


def test_run_inventory_refuses_orphaned_output_without_a_manifest(tmp_path):
    """A non-empty output root must be refused even without manifest.json --
    e.g. an orphaned windows.csv left by an incompatible/partial prior run
    must not be silently overwritten."""
    curated = _write_curated(tmp_path, n=10)
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir, cid=np.array([0.0]), window_blocks=np.array([[0, 3]]),
        popts=np.array([[10.0, 1.0, 1.0]]), mpcts=np.array([12.5]),
    )
    config_path = _write_config(tmp_path, curated, qc_dir, tmp_path / "recording")
    out_root = tmp_path / "out"
    out_root.mkdir()
    (out_root / "windows.csv").write_text("stale,data\n1,2\n")
    with pytest.raises(RuntimeError, match="non-empty output root"):
        run_inventory(config_path, out_root)


def test_run_inventory_writes_failed_manifest_on_error(tmp_path):
    curated = _write_curated(tmp_path, n=10)
    qc_dir = tmp_path / "qc"  # deliberately never populated -> load_cached_truncation_qc raises
    config_path = _write_config(tmp_path, curated, qc_dir, tmp_path / "recording")
    out_root = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        run_inventory(config_path, out_root)
    manifest = json.loads((out_root / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "failure_reason" in manifest


# --------------------------------------------------------------------------- #
# layer 3: deterministic case selection
#
# Every table below is hand-constructed. Nothing here reads real data, and --
# the prescription's hard rule -- no candidate, waveform, voltage or
# intervention outcome is ever an input to selection.
# --------------------------------------------------------------------------- #
def _wrow(*, sort_id="s1", cluster_id=0, source_row=0, i0=0, nominal=1000,
          start_s=0.0, dur_s=100.0, missing_pct=1.0, status=STATUS_FINITE_INTERIOR):
    """One normalized `windows.csv` row, with production's inclusive [i0, i1]."""
    i1 = i0 + nominal - 1
    return {
        "sort_id": sort_id, "cluster_id": cluster_id, "source_row": source_row,
        "i0": i0, "i1": i1,
        "first_sample": int(round(start_s * 30000.0)),
        "last_sample": int(round((start_s + dur_s) * 30000.0)),
        "start_s": start_s, "end_s": start_s + dur_s,
        "historical_count": nominal - 1, "nominal_count": nominal,
        "missing_pct": missing_pct,
        "fit_x0": 1.0, "fit_k": 1.0, "fit_A": 1.0,
        "status": status, "invalid_reason": None,
    }


def _run_rows(mpcts, *, sort_id="s1", cluster_id=0, i0=0, start_s=0.0, dur_s=100.0,
              inter_window_gap_s=0.0, nominals=None, statuses=None, base_row=0):
    """A run of windows tiling one gap-free block: `i0_next == i1_prev + 1`."""
    rows = []
    idx = i0
    t = start_s
    for k, mpct in enumerate(mpcts):
        nominal = 1000 if nominals is None else nominals[k]
        status = STATUS_FINITE_INTERIOR if statuses is None else statuses[k]
        rows.append(_wrow(sort_id=sort_id, cluster_id=cluster_id, source_row=base_row + k,
                          i0=idx, nominal=nominal, start_s=t, dur_s=dur_s,
                          missing_pct=mpct, status=status))
        idx += nominal
        t += dur_s + inter_window_gap_s
    return rows


def _table(*groups):
    return pd.DataFrame([row for group in groups for row in group])


def _ids(result):
    return [c["case_id"] for c in result["cases"]]


def _failing_run(**kw):
    """The canonical qualifying transition: 1/2 % -> 20/22 %, 400 s span."""
    return _run_rows([1.0, 2.0, 20.0, 22.0], **kw)


def test_select_freezes_one_failure_case_from_a_qualifying_transition():
    result = select_cases(_table(_failing_run(sort_id="s1", cluster_id=7)), CONSTANTS)
    assert _ids(result) == ["s1__c7__failure1"]
    case = result["cases"][0]
    assert case["role"] == "failure" and case["rank"] == 1
    assert [w["window_role"] for w in case["windows"]] == [
        "reference", "reference", "failing", "failing"
    ]
    assert [w["i0"] for w in case["windows"]] == [0, 1000, 2000, 3000]
    assert [w["i1"] for w in case["windows"]] == [999, 1999, 2999, 3999]
    assert [w["nominal_count"] for w in case["windows"]] == [1000] * 4
    assert [w["historical_count"] for w in case["windows"]] == [999] * 4
    assert [w["missing_pct"] for w in case["windows"]] == [1.0, 2.0, 20.0, 22.0]
    assert case["reference_median_missing_pct"] == pytest.approx(1.5)
    assert case["failing_median_missing_pct"] == pytest.approx(21.0)
    assert case["difference_pp"] == pytest.approx(19.5)
    assert case["span_s"] == pytest.approx(400.0)
    assert "gap-free block" in case["reason"] and "rank 1" in case["reason"]


def test_reference_threshold_is_inclusive_at_5_pct():
    """Both reference windows at exactly 5.0 % qualify; a hair above does not."""
    ok = select_cases(_table(_run_rows([5.0, 5.0, 20.0, 20.0], cluster_id=1)), CONSTANTS)
    assert _ids(ok) == ["s1__c1__failure1"]

    rows = _run_rows([5.0, 5.0 + 1e-9, 20.0, 20.0], cluster_id=1)
    bad = select_cases(_table(rows), CONSTANTS)
    assert bad["cases"] == []
    assert bad["exclusion_counts"]["failure_runs_by_reason"][
        "reference_above_max_missing_pct"] == 1


def test_failing_threshold_is_inclusive_at_15_pct():
    """Both failing windows at exactly 15.0 % qualify; a hair below does not."""
    ok = select_cases(_table(_run_rows([1.0, 1.0, 15.0, 15.0], cluster_id=2)), CONSTANTS)
    assert _ids(ok) == ["s1__c2__failure1"]

    bad = select_cases(_table(_run_rows([1.0, 1.0, 15.0, 15.0 - 1e-9], cluster_id=2)), CONSTANTS)
    assert bad["cases"] == []
    assert bad["exclusion_counts"]["failure_runs_by_reason"]["failing_below_min_missing_pct"] == 1


def test_median_difference_is_inclusive_at_10_pp_and_is_actually_applied():
    """5/5 -> 15/15 is the only configuration that can reach the 10.0 pp bound
    exactly, and it must qualify.

    With the shipped constants the median rule can never bind below equality
    (median(failing) >= 15 and median(reference) <= 5 already force >= 10 pp),
    so a raised constant is used to prove the rule is live rather than dead
    code that happens to agree.
    """
    ok = select_cases(_table(_run_rows([5.0, 5.0, 15.0, 15.0], cluster_id=3)), CONSTANTS)
    assert _ids(ok) == ["s1__c3__failure1"]
    assert ok["cases"][0]["difference_pp"] == pytest.approx(10.0)

    stricter = dataclasses.replace(CONSTANTS, min_median_difference_pp=12.0)
    bad = select_cases(_table(_run_rows([5.0, 5.0, 15.0, 15.0], cluster_id=3)), stricter)
    assert bad["cases"] == []
    assert bad["exclusion_counts"]["failure_runs_by_reason"][
        "median_difference_below_min_pp"] == 1


def test_span_cap_is_inclusive_at_600_s_and_exclusions_are_reported():
    """A 600.0 s span qualifies; a longer one is excluded and COUNTED, so slow
    units stay visible in the inventory instead of vanishing."""
    ok = select_cases(_table(_failing_run(cluster_id=4, dur_s=150.0)), CONSTANTS)
    assert _ids(ok) == ["s1__c4__failure1"]
    assert ok["cases"][0]["span_s"] == pytest.approx(600.0)

    slow = select_cases(
        _table(_failing_run(cluster_id=4, dur_s=150.0, inter_window_gap_s=1.0)), CONSTANTS
    )
    assert slow["cases"] == []
    assert slow["exclusion_counts"]["failure_runs_excluded_by_span_cap"] == 1
    assert slow["exclusion_counts"]["failure_runs_by_reason"]["span_over_max_s"] == 1


@pytest.mark.parametrize("bad_status", [STATUS_BOUNDARY_PINNED, STATUS_INVALID_INPUT,
                                        STATUS_NONFINITE_FIT])
def test_an_intervening_invalid_or_boundary_pinned_window_breaks_the_run(bad_status):
    """A censored/invalid window sits inside the interval, so it must break the
    run rather than being filtered out and letting its neighbours look
    adjacent. A 50 % boundary-pinned fit is never a numeric change score."""
    rows = _run_rows(
        [1.0, 2.0, 50.0, 20.0, 22.0], cluster_id=5,
        statuses=[STATUS_FINITE_INTERIOR, STATUS_FINITE_INTERIOR, bad_status,
                  STATUS_FINITE_INTERIOR, STATUS_FINITE_INTERIOR],
    )
    result = select_cases(_table(rows), CONSTANTS)
    assert result["cases"] == []
    assert result["exclusion_counts"]["failure_runs_by_reason"][
        "status_not_finite_interior"] == 2  # both length-4 runs contain it


def test_a_non_contiguous_i0_jump_breaks_the_run():
    rows = _failing_run(cluster_id=6)
    for row in rows[2:]:            # shift the second half one index later
        row["i0"] += 1
        row["i1"] += 1
    result = select_cases(_table(rows), CONSTANTS)
    assert result["cases"] == []
    assert result["exclusion_counts"]["failure_runs_by_reason"]["non_contiguous_index"] == 1


def test_a_wrong_nominal_count_breaks_the_run():
    rows = _run_rows([1.0, 2.0, 20.0, 22.0], cluster_id=8, nominals=[1000, 1000, 999, 1000])
    assert [r["i0"] for r in rows] == [0, 1000, 2000, 2999]   # still contiguous
    result = select_cases(_table(rows), CONSTANTS)
    assert result["cases"] == []
    assert result["exclusion_counts"]["failure_runs_by_reason"]["nominal_count_not_required"] == 1


def test_a_production_gap_between_two_index_contiguous_windows_breaks_the_run():
    """Index contiguity alone can bridge two blocks whose lengths are exact
    multiples of the window size. The prescription requires the four windows to
    lie inside ONE gap-free block, so a >10 s separation between window k's
    last spike and window k+1's first spike -- already known to be adjacent
    spikes by index -- splits the run."""
    over = select_cases(
        _table(_failing_run(cluster_id=9, inter_window_gap_s=PRODUCTION_MAX_ISI_S + 0.5)),
        CONSTANTS,
    )
    assert over["cases"] == []
    assert over["exclusion_counts"]["failure_runs_by_reason"]["gap_between_windows"] == 1

    exact = select_cases(
        _table(_failing_run(cluster_id=9, inter_window_gap_s=PRODUCTION_MAX_ISI_S)), CONSTANTS
    )
    assert _ids(exact) == ["s1__c9__failure1"]   # exactly 10 s does not split


def test_ties_are_broken_by_start_then_by_numeric_cluster_id():
    same = [1.0, 1.0, 21.0, 21.0]
    later_but_lower_id = _run_rows(same, cluster_id=3, start_s=1000.0)
    earlier_but_higher_id = _run_rows(same, cluster_id=7, start_s=0.0)
    result = select_cases(_table(later_but_lower_id, earlier_but_higher_id), CONSTANTS)
    assert _ids(result) == ["s1__c7__failure1", "s1__c3__failure2"]  # start beats ID

    tie_on_start = select_cases(
        _table(_run_rows(same, cluster_id=9, start_s=0.0),
               _run_rows(same, cluster_id=4, start_s=0.0)),
        CONSTANTS,
    )
    assert _ids(tie_on_start) == ["s1__c4__failure1", "s1__c9__failure2"]  # then ID


def test_largest_difference_per_cluster_is_kept_and_cluster_ties_go_to_earliest_start():
    early_small = _run_rows([1.0, 1.0, 16.0, 16.0], cluster_id=2, i0=0, start_s=0.0)
    late_large = _run_rows([1.0, 1.0, 26.0, 26.0], cluster_id=2, i0=50000, start_s=5000.0)
    result = select_cases(_table(early_small, late_large), CONSTANTS)
    assert _ids(result) == ["s1__c2__failure1"]
    assert result["cases"][0]["difference_pp"] == pytest.approx(25.0)
    assert result["cases"][0]["windows"][0]["i0"] == 50000

    same = [1.0, 1.0, 21.0, 21.0]
    tie = select_cases(
        _table(_run_rows(same, cluster_id=2, i0=0, start_s=0.0),
               _run_rows(same, cluster_id=2, i0=50000, start_s=5000.0)),
        CONSTANTS,
    )
    assert tie["cases"][0]["windows"][0]["i0"] == 0


def test_only_two_failure_cases_per_sort_are_kept():
    result = select_cases(
        _table(_run_rows([1.0, 1.0, 30.0, 30.0], cluster_id=1),
               _run_rows([1.0, 1.0, 25.0, 25.0], cluster_id=2),
               _run_rows([1.0, 1.0, 20.0, 20.0], cluster_id=3)),
        CONSTANTS,
    )
    failures = [c for c in result["cases"] if c["role"] == "failure"]
    assert [c["case_id"] for c in failures] == ["s1__c1__failure1", "s1__c2__failure2"]
    assert result["per_sort"]["s1"]["n_failure_eligible_clusters"] == 3
    assert result["per_sort"]["s1"]["n_failure_cases_selected"] == 2


def test_fewer_cases_than_the_caps_is_a_valid_result_without_relaxation():
    """One sort yields a case, the other yields nothing at all. Nothing is
    backfilled and no threshold moves."""
    result = select_cases(
        _table(_failing_run(sort_id="rescue", cluster_id=1),
               _run_rows([9.0, 9.0, 9.0, 9.0], sort_id="legacy", cluster_id=1)),
        CONSTANTS,
    )
    assert _ids(result) == ["rescue__c1__failure1"]
    assert result["per_sort"]["legacy"]["n_failure_cases_selected"] == 0
    assert result["per_sort"]["legacy"]["n_control_cases_selected"] == 0
    assert result["per_sort"]["legacy"]["first_failure_span_s"] is None


def test_control_range_is_inclusive_at_3_pp():
    ok = select_cases(_table(_run_rows([1.0, 4.0, 2.0, 3.0], cluster_id=1)), CONSTANTS)
    assert _ids(ok) == ["s1__c1__control1"]
    assert ok["cases"][0]["range_pp"] == pytest.approx(3.0)

    bad = select_cases(_table(_run_rows([1.0, 4.5, 2.0, 3.0], cluster_id=1)), CONSTANTS)
    assert bad["cases"] == []
    assert bad["exclusion_counts"]["control_runs_by_reason"]["range_above_max_pp"] == 1


def test_control_missing_pct_is_inclusive_at_5_pct():
    ok = select_cases(_table(_run_rows([5.0, 5.0, 5.0, 5.0], cluster_id=1)), CONSTANTS)
    assert _ids(ok) == ["s1__c1__control1"]

    bad = select_cases(_table(_run_rows([5.0, 5.0, 5.0, 5.0 + 1e-9], cluster_id=1)), CONSTANTS)
    assert bad["cases"] == []
    assert bad["exclusion_counts"]["control_runs_by_reason"]["above_max_missing_pct"] == 1


def test_a_cluster_selected_as_a_failure_cannot_also_be_the_control():
    failure = _failing_run(cluster_id=1, i0=0, start_s=0.0)
    same_cluster_control = _run_rows([1.0, 1.0, 1.0, 1.0], cluster_id=1, i0=50000,
                                     start_s=5000.0, base_row=4)
    other_cluster_control = _run_rows([2.0, 2.0, 2.0, 2.0], cluster_id=2, i0=0, start_s=0.0)
    result = select_cases(
        _table(failure, same_cluster_control, other_cluster_control), CONSTANTS
    )
    assert _ids(result) == ["s1__c1__failure1", "s1__c2__control1"]
    assert result["exclusion_counts"]["control_runs_by_reason"]["cluster_selected_as_failure"] > 0


def test_no_control_is_invented_when_the_only_candidate_is_the_failure_cluster():
    result = select_cases(
        _table(_failing_run(cluster_id=1, i0=0, start_s=0.0),
               _run_rows([1.0, 1.0, 1.0, 1.0], cluster_id=1, i0=50000, start_s=5000.0,
                         base_row=4)),
        CONSTANTS,
    )
    assert _ids(result) == ["s1__c1__failure1"]
    assert result["per_sort"]["s1"]["n_control_cases_selected"] == 0


def test_control_is_chosen_by_the_minimal_absolute_log_span_ratio():
    """Failure span 400 s. Candidate spans 200 s (|log 0.5| = 0.693) and 600 s
    (|log 1.5| = 0.405): the longer one wins even though it is further away in
    absolute seconds."""
    result = select_cases(
        _table(_failing_run(cluster_id=1, dur_s=100.0),
               _run_rows([1.0, 1.0, 1.0, 1.0], cluster_id=2, dur_s=50.0),
               _run_rows([1.0, 1.0, 1.0, 1.0], cluster_id=3, dur_s=150.0)),
        CONSTANTS,
    )
    assert _ids(result) == ["s1__c1__failure1", "s1__c3__control1"]
    control = result["cases"][1]
    assert control["span_s"] == pytest.approx(600.0)
    assert control["abs_log_span_ratio"] == pytest.approx(abs(math.log(600.0 / 400.0)))
    assert "minimal |log(control span" in control["reason"]


def test_control_log_ratio_ties_break_by_start_then_cluster_id():
    stable = [1.0, 1.0, 1.0, 1.0]
    result = select_cases(
        _table(_failing_run(cluster_id=1, dur_s=100.0),
               _run_rows(stable, cluster_id=3, dur_s=100.0, start_s=0.0),
               _run_rows(stable, cluster_id=2, dur_s=100.0, start_s=9000.0)),
        CONSTANTS,
    )
    assert _ids(result) == ["s1__c1__failure1", "s1__c3__control1"]  # start beats ID

    tied = select_cases(
        _table(_failing_run(cluster_id=1, dur_s=100.0),
               _run_rows(stable, cluster_id=5, dur_s=100.0, start_s=9000.0),
               _run_rows(stable, cluster_id=4, dur_s=100.0, start_s=9000.0)),
        CONSTANTS,
    )
    assert _ids(tied) == ["s1__c1__failure1", "s1__c4__control1"]


def test_with_no_failure_the_earliest_eligible_control_is_taken():
    stable = [1.0, 1.0, 1.0, 1.0]
    result = select_cases(
        _table(_run_rows(stable, cluster_id=2, dur_s=150.0, start_s=5000.0),
               _run_rows(stable, cluster_id=8, dur_s=50.0, start_s=10.0)),
        CONSTANTS,
    )
    assert _ids(result) == ["s1__c8__control1"]
    assert result["cases"][0]["abs_log_span_ratio"] is None
    assert "no selected failure case" in result["cases"][0]["reason"]


def test_controls_are_selected_independently_per_sort():
    result = select_cases(
        _table(_failing_run(sort_id="rescue", cluster_id=1),
               _run_rows([1.0, 1.0, 1.0, 1.0], sort_id="rescue", cluster_id=2),
               _failing_run(sort_id="legacy", cluster_id=1),
               _run_rows([1.0, 1.0, 1.0, 1.0], sort_id="legacy", cluster_id=3)),
        CONSTANTS,
    )
    assert _ids(result) == [
        "legacy__c1__failure1", "legacy__c3__control1",
        "rescue__c1__failure1", "rescue__c2__control1",
    ]
    assert len([c for c in result["cases"] if c["role"] == "failure"]) <= 4
    assert len([c for c in result["cases"] if c["role"] == "control"]) <= 2


def test_window_records_keeps_censored_rows_but_drops_no_fit_placeholders():
    rows = _failing_run(cluster_id=1)
    rows[2]["status"] = STATUS_BOUNDARY_PINNED
    no_fit = _wrow(cluster_id=2, source_row=-1)
    no_fit.update({"i0": np.nan, "i1": np.nan, "first_sample": np.nan, "last_sample": np.nan,
                   "start_s": np.nan, "end_s": np.nan, "historical_count": np.nan,
                   "nominal_count": np.nan, "missing_pct": np.nan, "status": STATUS_NO_FIT})
    grouped = window_records(_table(rows, [no_fit]))
    assert set(grouped) == {("s1", 1)}
    assert [r["status"] for r in grouped[("s1", 1)]][2] == STATUS_BOUNDARY_PINNED
    assert [r["i0"] for r in grouped[("s1", 1)]] == [0, 1000, 2000, 3000]


def test_classifiers_are_pure_functions_of_windows_and_constants():
    rows = window_records(_table(_failing_run(cluster_id=1)))[("s1", 1)]
    assert classify_failure_run(rows, CONSTANTS)[0] == "qualified"
    assert classify_control_run(rows, CONSTANTS)[0] == "above_max_missing_pct"


def test_no_candidate_or_intervention_column_can_influence_selection():
    """CRITICAL contract: selection consumes cached historical QC rows only."""
    base = _table(_failing_run(cluster_id=1), _run_rows([1.0, 1.0, 1.0, 1.0], cluster_id=2))
    poisoned = base.copy()
    poisoned["intervention_outcome"] = "improved"
    poisoned["ks_label"] = ["mua"] * len(poisoned)
    poisoned["waveform_score"] = np.arange(len(poisoned), dtype=float)
    assert select_cases(poisoned, CONSTANTS)["cases"] == select_cases(base, CONSTANTS)["cases"]


def test_ks_good_mua_status_does_not_filter_eligibility():
    table = _table(_failing_run(cluster_id=1))
    table["ks_label"] = "mua"
    assert _ids(select_cases(table, CONSTANTS)) == ["s1__c1__failure1"]


# --------------------------------------------------------------------------- #
# layer 3: frozen selection constants live in CONFIG, nowhere else
# --------------------------------------------------------------------------- #
def test_real_config_carries_the_prescribed_frozen_selection_constants():
    cfg = load_config(REAL_CONFIG)
    assert cfg.selection.windows_per_case == 4
    assert cfg.selection.required_nominal_count == 1000
    assert cfg.selection.reference_max_missing_pct == 5.0
    assert cfg.selection.failing_min_missing_pct == 15.0
    assert cfg.selection.min_median_difference_pp == 10.0
    assert cfg.selection.max_span_s == 600.0
    assert cfg.selection.control_max_missing_pct == 5.0
    assert cfg.selection.control_max_range_pp == 3.0
    assert cfg.selection.max_failure_cases_per_sort == 2
    assert cfg.selection.max_control_cases_per_sort == 1
    # the prescription requires the units alongside the constants
    for key in SELECTION_CONSTANT_KEYS:
        assert cfg.selection.units[key].strip()


def test_load_config_refuses_a_config_without_a_selection_block(tmp_path):
    payload = json.loads(REAL_CONFIG.read_text())
    payload.pop("selection")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no 'selection' block"):
        load_config(path)


def test_load_config_refuses_a_selection_block_missing_one_constant(tmp_path):
    payload = json.loads(REAL_CONFIG.read_text())
    payload["selection"].pop("max_span_s")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing required key"):
        load_config(path)


def test_load_config_refuses_an_unknown_selection_key(tmp_path):
    """A typo'd constant that silently fell back to a built-in default would
    defeat the freeze, so unknown keys are refused."""
    payload = json.loads(REAL_CONFIG.read_text())
    payload["selection"]["max_span_seconds"] = 600.0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_load_config_refuses_a_selection_block_missing_units(tmp_path):
    payload = json.loads(REAL_CONFIG.read_text())
    payload["selection"]["units"].pop("max_span_s")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing a unit"):
        load_config(path)


def test_selection_constants_reject_incoherent_values():
    with pytest.raises(ValueError, match="windows_per_case"):
        dataclasses.replace(CONSTANTS, windows_per_case=3)
    with pytest.raises(ValueError, match="max_span_s"):
        dataclasses.replace(CONSTANTS, max_span_s=0.0)
    with pytest.raises(ValueError, match="failing_min_missing_pct"):
        dataclasses.replace(CONSTANTS, failing_min_missing_pct=1.0)
    with pytest.raises(ValueError, match="finite"):
        dataclasses.replace(CONSTANTS, max_span_s=float("nan"))


def test_selection_constants_cannot_be_overridden_on_the_command_line(tmp_path):
    """The prescription freezes them in CONFIG before any ranking is read, so
    no CLI flag may move one."""
    for flag in ("--reference-max-missing-pct", "--max-span-s", "--failing-min-missing-pct"):
        with pytest.raises(SystemExit):
            main(["select", "--config", str(REAL_CONFIG), "--out-root", str(tmp_path / "o"),
                  flag, "1.0"])


# --------------------------------------------------------------------------- #
# layer 3: `select` writes into the inventory's own out-root, once
# --------------------------------------------------------------------------- #
def _completed_inventory(tmp_path: Path, windows: pd.DataFrame | None = None):
    """Run a real `inventory` on synthetic fixtures, then (optionally) replace
    windows.csv with a hand-built table so selection can be exercised without
    fabricating a 4,000-spike cluster.

    The manifest's recorded windows.csv hash is re-stamped to match, i.e. the
    result stands in for an inventory that genuinely produced that table. The
    guard against a windows.csv changing *behind* the manifest is exercised
    separately, by the tests that tamper without re-stamping.
    """
    curated = _write_curated(tmp_path, n=10)
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir, cid=np.array([0.0]), window_blocks=np.array([[0, 3]]),
        popts=np.array([[10.0, 1.0, 1.0]]), mpcts=np.array([12.5]),
    )
    config_path = _write_config(tmp_path, curated, qc_dir, tmp_path / "recording")
    out_root = tmp_path / "out"
    run_inventory(config_path, out_root)
    if windows is not None:
        windows.to_csv(out_root / "windows.csv", index=False)
        _restamp_windows_hash(out_root)
    return config_path, out_root


def _restamp_windows_hash(out_root: Path) -> None:
    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["windows_csv_sha256"] = sha256_file(out_root / "windows.csv")
    manifest_path.write_text(json.dumps(manifest))


def test_run_select_end_to_end_freezes_cases_and_updates_the_manifest(tmp_path):
    table = _table(_failing_run(sort_id="synthetic", cluster_id=1),
                   _run_rows([1.0, 1.0, 1.0, 1.0], sort_id="synthetic", cluster_id=2))
    config_path, out_root = _completed_inventory(tmp_path, table)

    payload = run_select(config_path, out_root)
    on_disk = json.loads((out_root / "selection.json").read_text())
    assert on_disk == payload
    assert [c["case_id"] for c in payload["cases"]] == [
        "synthetic__c1__failure1", "synthetic__c2__control1"
    ]

    # every input that could change the answer is recorded
    assert payload["schema"] == SCHEMA
    assert payload["selection_constants"] == CONSTANTS.to_dict()
    assert payload["windows_csv_sha256"] == sha256_file(out_root / "windows.csv")
    assert payload["config_sha256"] == sha256_file(config_path)
    assert payload["source_sha256"]["module"] == sha256_file(
        REPO_ROOT / "testing/luke_amplitude_dropout_audit.py")
    assert payload["git_commit"]
    assert payload["production_constants"]["max_isi_s"] == PRODUCTION_MAX_ISI_S
    assert "failure_runs_excluded_by_span_cap" in payload["exclusion_counts"]

    manifest = json.loads((out_root / "manifest.json").read_text())
    assert manifest["stage"] == "select" and manifest["status"] == "complete"
    assert manifest["select"]["selection_sha256"] == payload["selection_sha256"]
    assert manifest["select"]["case_ids"] == [c["case_id"] for c in payload["cases"]]


def test_selection_sha256_covers_the_frozen_content(tmp_path):
    table = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    config_path, out_root = _completed_inventory(tmp_path, table)
    payload = run_select(config_path, out_root)

    assert payload["selection_sha256"] == canonical_selection_digest(payload)
    tampered = json.loads(json.dumps(payload))
    tampered["cases"][0]["windows"][0]["missing_pct"] = 99.0
    assert canonical_selection_digest(tampered) != payload["selection_sha256"]


def test_select_refuses_a_missing_inventory_manifest(tmp_path):
    config_path, out_root = _completed_inventory(tmp_path)
    (out_root / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="no manifest.json"):
        run_select(config_path, out_root)


def test_select_refuses_an_incomplete_inventory(tmp_path):
    config_path, out_root = _completed_inventory(tmp_path)
    manifest = json.loads((out_root / "manifest.json").read_text())
    manifest["status"] = "running"
    (out_root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="inventory status"):
        run_select(config_path, out_root)


def test_select_refuses_a_manifest_at_the_wrong_stage(tmp_path):
    config_path, out_root = _completed_inventory(tmp_path)
    manifest = json.loads((out_root / "manifest.json").read_text())
    manifest["stage"] = "inspect"
    (out_root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="expected 'inventory'"):
        run_select(config_path, out_root)


def test_select_refuses_a_foreign_schema(tmp_path):
    config_path, out_root = _completed_inventory(tmp_path)
    manifest = json.loads((out_root / "manifest.json").read_text())
    manifest["schema"] = "some-other-audit-v9"
    (out_root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="schema"):
        run_select(config_path, out_root)


def test_select_refuses_a_config_the_inventory_was_not_built_with(tmp_path):
    """Mismatch is refused with a reason; the inventory is never regenerated."""
    table = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    config_path, out_root = _completed_inventory(tmp_path, table)
    payload = json.loads(config_path.read_text())
    payload["sorts"][0]["duration_s"] = 2.0        # any change moves the hash
    config_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="config_sha256 mismatch"):
        run_select(config_path, out_root)
    assert not (out_root / "selection.json").exists()


def test_select_refuses_an_out_root_without_windows_csv(tmp_path):
    config_path, out_root = _completed_inventory(tmp_path)
    (out_root / "windows.csv").unlink()
    with pytest.raises(RuntimeError, match="windows.csv"):
        run_select(config_path, out_root)


def test_select_twice_is_refused(tmp_path):
    table = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    config_path, out_root = _completed_inventory(tmp_path, table)
    first = run_select(config_path, out_root)
    with pytest.raises(RuntimeError, match="existing frozen selection"):
        run_select(config_path, out_root)
    assert json.loads((out_root / "selection.json").read_text()) == first


def test_select_still_refuses_an_unsafe_out_root():
    with pytest.raises(ValueError, match="/mnt"):
        run_select(REAL_CONFIG, Path("/mnt/NPX/scratch_selection"))


def test_select_writes_a_failed_manifest_and_leaves_no_selection(tmp_path):
    bad = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    bad["cluster_id"] = np.nan          # unusable inventory row
    config_path, out_root = _completed_inventory(tmp_path, bad)
    with pytest.raises(ValueError, match="cluster_id"):
        run_select(config_path, out_root)
    manifest = json.loads((out_root / "manifest.json").read_text())
    assert manifest["stage"] == "select" and manifest["status"] == "failed"
    assert "failure_reason" in manifest
    assert not (out_root / "selection.json").exists()


def test_select_cli_reports_the_frozen_case_ids(tmp_path, capsys):
    table = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    config_path, out_root = _completed_inventory(tmp_path, table)
    assert main(["select", "--config", str(config_path), "--out-root", str(out_root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["case_ids"] == ["synthetic__c1__failure1"]
    assert printed["selection_sha256"] == json.loads(
        (out_root / "selection.json").read_text())["selection_sha256"]


# --------------------------------------------------------------------------- #
# layer 3 regressions: windows.csv provenance between inventory and select
# --------------------------------------------------------------------------- #
def test_run_inventory_records_the_windows_csv_hash(tmp_path):
    """The inventory must attest the table it produced, not just the arrays it
    read: otherwise select can only re-hash whatever windows.csv it finds."""
    config_path, out_root = _completed_inventory(tmp_path)
    manifest = json.loads((out_root / "manifest.json").read_text())
    assert manifest["windows_csv_sha256"] == sha256_file(out_root / "windows.csv")


def test_select_refuses_a_windows_csv_altered_since_the_inventory(tmp_path):
    """A windows.csv modified, corrupted or replaced between the two stages
    must be refused. Without the inventory-time hash, select would rank the
    substituted rows and selection.json's own hash would merely authenticate
    the replacement, letting arbitrary case IDs claim to come from a completed
    inventory."""
    table = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    config_path, out_root = _completed_inventory(tmp_path, table)

    swapped = _table(_failing_run(sort_id="synthetic", cluster_id=4242))
    swapped.to_csv(out_root / "windows.csv", index=False)   # manifest NOT re-stamped

    with pytest.raises(RuntimeError, match="windows.csv sha256 mismatch"):
        run_select(config_path, out_root)
    assert not (out_root / "selection.json").exists()


def test_select_refuses_a_windows_csv_edited_in_place_since_the_inventory(tmp_path):
    """Even a one-value edit that leaves the row count identical is refused."""
    table = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    config_path, out_root = _completed_inventory(tmp_path, table)

    edited = pd.read_csv(out_root / "windows.csv")
    edited.loc[0, "missing_pct"] = 4.9
    edited.to_csv(out_root / "windows.csv", index=False)

    with pytest.raises(RuntimeError, match="windows.csv sha256 mismatch"):
        run_select(config_path, out_root)


def test_select_refuses_a_manifest_that_never_attested_windows_csv(tmp_path):
    """An inventory manifest predating the attestation cannot be trusted to
    have produced the windows.csv sitting next to it."""
    table = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    config_path, out_root = _completed_inventory(tmp_path, table)
    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("windows_csv_sha256")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="records no windows_csv_sha256"):
        run_select(config_path, out_root)


# --------------------------------------------------------------------------- #
# layer 3 regressions: a sort that yields nothing must still be reported
# --------------------------------------------------------------------------- #
def test_a_sort_whose_inventory_is_entirely_no_fit_still_reports_zero_cases():
    """`window_records` drops no_fit placeholders, so a sort with nothing but
    no_fit clusters has no entry there. It must not therefore vanish from
    per_sort -- that would hide exactly the insufficient-data condition this
    audit exists to report."""
    no_fit_rows = []
    for cluster_id in (11, 12):
        row = _wrow(sort_id="legacy", cluster_id=cluster_id, source_row=-1)
        row.update({"i0": np.nan, "i1": np.nan, "first_sample": np.nan,
                    "last_sample": np.nan, "start_s": np.nan, "end_s": np.nan,
                    "historical_count": np.nan, "nominal_count": np.nan,
                    "missing_pct": np.nan, "status": STATUS_NO_FIT})
        no_fit_rows.append(row)

    result = select_cases(
        _table(_failing_run(sort_id="rescue", cluster_id=1), no_fit_rows), CONSTANTS
    )
    assert _ids(result) == ["rescue__c1__failure1"]
    assert set(result["per_sort"]) == {"rescue", "legacy"}
    legacy = result["per_sort"]["legacy"]
    assert legacy["n_clusters_with_stored_windows"] == 0
    assert legacy["n_failure_eligible_clusters"] == 0
    assert legacy["n_failure_cases_selected"] == 0
    assert legacy["n_eligible_control_runs"] == 0
    assert legacy["n_control_cases_selected"] == 0
    assert legacy["first_failure_span_s"] is None
    assert legacy["failure_cap"] == CONSTANTS.max_failure_cases_per_sort


def test_a_configured_sort_with_no_inventory_rows_at_all_still_reports_zero_cases():
    result = select_cases(
        _table(_failing_run(sort_id="rescue", cluster_id=1)), CONSTANTS,
        configured_sort_ids=["rescue", "legacy"],
    )
    assert set(result["per_sort"]) == {"rescue", "legacy"}
    assert result["per_sort"]["legacy"]["n_failure_cases_selected"] == 0


def test_run_select_reports_every_configured_sort_even_when_it_yields_nothing(tmp_path):
    """End to end: the configured sort id survives into selection.json."""
    no_fit = _wrow(sort_id="synthetic", cluster_id=3, source_row=-1)
    no_fit.update({"i0": np.nan, "i1": np.nan, "first_sample": np.nan, "last_sample": np.nan,
                   "start_s": np.nan, "end_s": np.nan, "historical_count": np.nan,
                   "nominal_count": np.nan, "missing_pct": np.nan, "status": STATUS_NO_FIT})
    config_path, out_root = _completed_inventory(tmp_path, _table([no_fit]))

    payload = run_select(config_path, out_root)
    assert payload["cases"] == []
    assert set(payload["per_sort"]) == {"synthetic"}   # the only configured sort
    assert payload["per_sort"]["synthetic"]["n_failure_cases_selected"] == 0
    assert payload["per_sort"]["synthetic"]["n_control_cases_selected"] == 0


# --------------------------------------------------------------------------- #
# layer 3 regressions: the attestation must survive a race, and an empty
# inventory must still be reportable
# --------------------------------------------------------------------------- #
def test_select_cannot_be_raced_between_hashing_and_parsing_windows_csv(tmp_path, monkeypatch):
    """Hashing the path and separately parsing the path is two reads: a file
    swapped in between would be ranked although the inventory never attested
    it, and selection.json's hash would not catch it either. The parse must
    come from the very bytes that were hashed."""
    attested = _table(_failing_run(sort_id="synthetic", cluster_id=1))
    config_path, out_root = _completed_inventory(tmp_path, attested)
    recorded = json.loads((out_root / "manifest.json").read_text())["windows_csv_sha256"]
    swapped = _table(_failing_run(sort_id="synthetic", cluster_id=4242))

    real_read_csv = pd.read_csv
    races = {"n": 0}

    def racing_read_csv(source, *args, **kwargs):
        # fires in the gap the old code left between hashing and parsing
        if races["n"] == 0:
            races["n"] += 1
            swapped.to_csv(out_root / "windows.csv", index=False)
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", racing_read_csv)
    payload = run_select(config_path, out_root)

    assert races["n"] == 1                                        # the race really fired
    assert sha256_file(out_root / "windows.csv") != recorded      # the file really changed
    # the attested rows were ranked, not the substituted ones
    assert [c["case_id"] for c in payload["cases"]] == ["synthetic__c1__failure1"]
    assert payload["windows_csv_sha256"] == recorded


def test_an_all_empty_inventory_reports_every_configured_sort_with_zero_cases(tmp_path):
    """The most complete form of the insufficient-data condition: no spikes, no
    clusters, no cached QC rows. It must reach a zero-case selection, not crash
    on an unparseable windows.csv before `select_cases` is ever called."""
    curated = _write_curated(tmp_path, n=0)
    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir, cid=np.zeros(0), window_blocks=np.zeros((0, 2)),
        popts=np.zeros((0, 3)), mpcts=np.zeros(0),
    )
    config_path = _write_config(tmp_path, curated, qc_dir, tmp_path / "recording")
    out_root = tmp_path / "out"

    manifest = run_inventory(config_path, out_root)
    assert manifest["status"] == "complete"
    assert manifest["row_counts"]["windows_total"] == 0
    # the empty table still carries its header, so it round-trips as a table
    assert list(pd.read_csv(out_root / "windows.csv").columns) == list(WINDOWS_COLUMNS)

    payload = run_select(config_path, out_root)
    assert payload["cases"] == []
    assert set(payload["per_sort"]) == {"synthetic"}
    zero = payload["per_sort"]["synthetic"]
    assert zero["n_clusters_with_stored_windows"] == 0
    assert zero["n_failure_eligible_clusters"] == 0
    assert zero["n_failure_cases_selected"] == 0
    assert zero["n_eligible_control_runs"] == 0
    assert zero["n_control_cases_selected"] == 0
    assert zero["first_failure_span_s"] is None
    assert payload["exclusion_counts"]["failure_runs_excluded_by_span_cap"] == 0


def test_select_survives_a_headerless_empty_windows_csv(tmp_path):
    """An inventory written before windows.csv carried a header leaves a
    zero-byte table; it must still parse as an empty table rather than raise."""
    config_path, out_root = _completed_inventory(tmp_path)
    (out_root / "windows.csv").write_bytes(b"")
    _restamp_windows_hash(out_root)

    payload = run_select(config_path, out_root)
    assert payload["cases"] == []
    assert set(payload["per_sort"]) == {"synthetic"}
    assert payload["per_sort"]["synthetic"]["n_failure_cases_selected"] == 0


# --------------------------------------------------------------------------- #
# layer 3 regression: a sort_id must survive windows.csv byte-for-byte
# --------------------------------------------------------------------------- #
def _write_two_sort_config(tmp_path: Path, sort_ids: list[str]) -> Path:
    """A config whose sort IDs are the awkward ones CSV inference rewrites."""
    recording = tmp_path / "recording"
    recording.mkdir(parents=True, exist_ok=True)
    sorts = []
    for n, sort_id in enumerate(sort_ids):
        curated = _write_curated(tmp_path / f"sort{n}", n=10)
        qc_dir = tmp_path / f"qc{n}"
        _write_cached_qc(
            qc_dir, cid=np.array([0.0]), window_blocks=np.array([[0, 3]]),
            popts=np.array([[10.0, 1.0, 1.0]]), mpcts=np.array([12.5]),
        )
        sorts.append({
            "sort_id": sort_id, "curated": str(curated), "qc_dir": str(qc_dir),
            "source_recording": str(recording), "sampling_frequency_hz": 1000.0,
            "selected_start_sample": 0, "duration_s": 1.0,
        })
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(
        {"schema": SCHEMA, "selection": SELECTION_BLOCK, "sorts": sorts}
    ))
    return config_path


def test_sort_ids_survive_the_windows_csv_round_trip_without_type_inference(tmp_path):
    """`pd.read_csv` infers types, so "001" comes back as 1 and "NA" as nan.
    The rows would then be keyed by a mangled sort while the configured
    identifier is unioned in as its original string -- wrong case IDs plus a
    phantom zero-case per_sort entry for the same sort."""
    config_path = _write_two_sort_config(tmp_path, ["001", "NA"])
    out_root = tmp_path / "out"
    manifest = run_inventory(config_path, out_root)

    windows, _ = read_attested_windows(
        out_root / "windows.csv", manifest["windows_csv_sha256"]
    )
    assert sorted(set(windows["sort_id"])) == ["001", "NA"]     # "NA" is not nan
    assert all(isinstance(v, str) for v in windows["sort_id"])
    # cluster 1 has no cached QC row, so it is a no_fit placeholder that
    # window_records drops -- but its row still carries the exact sort_id
    assert set(window_records(windows)) == {("001", 0), ("NA", 0)}
    assert set(windows.loc[windows["status"] == STATUS_NO_FIT, "sort_id"]) == {"001", "NA"}

    payload = run_select(config_path, out_root)
    assert set(payload["per_sort"]) == {"001", "NA"}            # no "1"/nan phantoms
    assert len(payload["per_sort"]) == len(json.loads(config_path.read_text())["sorts"])


def test_a_wholly_numeric_looking_sort_id_column_is_not_coerced_to_numbers():
    """The pure case the mixed column hides: every value CSV-coercible."""
    table = _table(_failing_run(sort_id="001", cluster_id=1),
                   _failing_run(sort_id="002", cluster_id=1))
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    parsed = _parse_windows_bytes(buffer.getvalue().encode())

    assert list(parsed["sort_id"].unique()) == ["001", "002"]
    assert _ids(select_cases(parsed, CONSTANTS)) == ["001__c1__failure1", "002__c1__failure1"]


def test_numeric_columns_stay_nullable_across_the_round_trip():
    """Disabling NA conversion must not spill onto the numeric columns: the
    no_fit placeholder rows depend on blank i0/i1/times reading as missing."""
    no_fit = _wrow(sort_id="s1", cluster_id=2, source_row=-1)
    no_fit.update({"i0": np.nan, "i1": np.nan, "first_sample": np.nan, "last_sample": np.nan,
                   "start_s": np.nan, "end_s": np.nan, "historical_count": np.nan,
                   "nominal_count": np.nan, "missing_pct": np.nan, "status": STATUS_NO_FIT,
                   "invalid_reason": None})
    buffer = io.StringIO()
    _table(_failing_run(sort_id="s1", cluster_id=1), [no_fit]).to_csv(buffer, index=False)
    parsed = _parse_windows_bytes(buffer.getvalue().encode())

    placeholder = parsed[parsed["status"] == STATUS_NO_FIT].iloc[0]
    for column in ("i0", "i1", "first_sample", "last_sample", "start_s", "end_s",
                   "historical_count", "nominal_count", "missing_pct"):
        assert pd.isna(placeholder[column]), column
    assert placeholder["invalid_reason"] is None
    assert parsed.loc[0, "i0"] == 0 and parsed.loc[0, "nominal_count"] == 1000
    # the placeholder is dropped by window_records but its sort still reports
    assert set(window_records(parsed)) == {("s1", 1)}
    assert set(select_cases(parsed, CONSTANTS)["per_sort"]) == {"s1"}


def test_junk_in_a_numeric_column_is_rejected_not_silently_read_as_missing():
    table = _table(_failing_run(sort_id="s1", cluster_id=1))
    table.loc[0, "i0"] = "not-a-number"
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    parsed = _parse_windows_bytes(buffer.getvalue().encode())
    with pytest.raises(ValueError, match="must be a number"):
        select_cases(parsed, CONSTANTS)


# --------------------------------------------------------------------------- #
# layer 4a: `inspect` -- gate, case_windows.csv, historical/exact replay
#
# Every fixture below is synthetic. The cached QC is generated by running
# production's own `analyze_amplitude_truncation` over the fixture arrays, so
# the cache these tests replay is the cache production would have written for
# them -- not a hand-typed number the replay is then asked to match.
# --------------------------------------------------------------------------- #
REPLAY_FS = 30000.0
REPLAY_ISI_SAMPLES = 3000          # 0.1 s, far inside the 10 s production gap rule
REPLAY_SPIKES_PER_WINDOW = 1000    # production's own window size


def _truncated_logistic_amps(n: int, missing_fraction: float, x0=20.0, k=1.0) -> np.ndarray:
    """`n` deterministic amplitudes on the fitter's own model.

    `fit_amp_cdf` fits a sigmoid CDF truncated at `min(amps)` and reports
    `100 * sigmoid(x_min)`. Drawing evenly spaced quantiles of a logistic(x0,
    k) truncated at the `missing_fraction` quantile therefore produces a window
    whose fitted missing percentage lands close to `100 * missing_fraction`,
    with no RNG anywhere.

    Values come out ascending, so the amplitude at a window's inclusive
    endpoint i1 -- the one production's `amps[i0:i1]` slice never sees -- is
    that window's largest, unless a test overrides it.
    """
    f = float(missing_fraction)
    u = f + (1.0 - f) * (np.arange(n) + 0.5) / n
    return x0 + np.log(u / (1.0 - u)) / k


def _write_replay_fixture(
    tmp_path: Path, spec: dict[int, list[float]], *,
    final_amp_overrides: dict[tuple[int, int], float] | None = None,
    mpct_perturbation: dict[tuple[int, int], float] | None = None,
    gap_after_window: dict[int, set[int]] | None = None,
) -> tuple[Path, Path]:
    """A curated directory plus the cache production QC would have written.

    `spec` maps cluster_id -> one target missing fraction per nominal
    1,000-spike window. `final_amp_overrides` replaces the single amplitude at
    a window's inclusive endpoint (cluster_id, window ordinal); that value is
    excluded from every historical fit by construction, so it can only reach
    the exact-1,000 fit. `mpct_perturbation` corrupts a cached missing
    percentage *after* the fit, standing in for a cache that no longer matches
    its own inputs. `gap_after_window` inserts a >10 s silence after a window,
    splitting production's block there.
    """
    from pipeline.truncation import analyze_amplitude_truncation

    final_amp_overrides = final_amp_overrides or {}
    mpct_perturbation = mpct_perturbation or {}
    gap_after_window = gap_after_window or {}

    curated_dir = tmp_path / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)

    all_times: list[np.ndarray] = []
    all_clusters: list[np.ndarray] = []
    all_amps: list[np.ndarray] = []
    cid_chunks: list[np.ndarray] = []
    block_chunks: list[np.ndarray] = []
    popt_chunks: list[np.ndarray] = []
    mpct_chunks: list[np.ndarray] = []

    next_sample = 0
    for cluster_id in sorted(spec):
        fractions = spec[cluster_id]
        amps = np.concatenate([
            _truncated_logistic_amps(REPLAY_SPIKES_PER_WINDOW, f) for f in fractions
        ])
        for (cid, ordinal), value in final_amp_overrides.items():
            if cid == cluster_id:
                amps[(ordinal + 1) * REPLAY_SPIKES_PER_WINDOW - 1] = float(value)

        n = amps.size
        isis = np.full(n - 1, REPLAY_ISI_SAMPLES, dtype=np.int64)
        for ordinal in gap_after_window.get(cluster_id, set()):
            isis[(ordinal + 1) * REPLAY_SPIKES_PER_WINDOW - 1] = int(20.0 * REPLAY_FS)
        times = next_sample + np.concatenate([[0], np.cumsum(isis)]).astype(np.int64)
        next_sample = int(times[-1]) + REPLAY_ISI_SAMPLES

        # exactly what pipeline.qc.truncation_qc does per cluster, on seconds
        blocks, _, popts, mpcts = analyze_amplitude_truncation(
            times / REPLAY_FS, amps, max_isi=10, spikes_per_window=REPLAY_SPIKES_PER_WINDOW,
        )
        for (cid, ordinal), delta in mpct_perturbation.items():
            if cid == cluster_id:
                mpcts[ordinal] = mpcts[ordinal] + float(delta)

        all_times.append(times)
        all_clusters.append(np.full(n, cluster_id, dtype=np.int64))
        all_amps.append(amps)
        if len(blocks):
            cid_chunks.append(np.full(len(blocks), cluster_id, dtype=float))
            block_chunks.append(np.asarray(blocks))
            popt_chunks.append(np.asarray(popts))
            mpct_chunks.append(np.asarray(mpcts))

    times = np.concatenate(all_times)
    clusters = np.concatenate(all_clusters)
    amplitudes = np.concatenate(all_amps)
    full_st = np.stack([times.astype(np.float64), np.zeros(times.size), amplitudes], axis=1)

    np.save(curated_dir / "full_st.npy", full_st)
    np.save(curated_dir / "kept_spikes.npy", np.ones(times.size, dtype=bool))
    np.save(curated_dir / "spike_times.npy", times)
    np.save(curated_dir / "spike_clusters.npy", clusters)
    # deliberately wrong: the loader must never read this column for amplitude
    np.save(curated_dir / "amplitudes.npy", -amplitudes - 999.0)

    qc_dir = tmp_path / "qc"
    _write_cached_qc(
        qc_dir,
        cid=np.concatenate(cid_chunks) if cid_chunks else np.empty((0,), dtype=float),
        window_blocks=(
            np.concatenate(block_chunks, axis=0) if block_chunks else np.empty((0, 2), dtype=int)
        ),
        popts=np.concatenate(popt_chunks, axis=0) if popt_chunks else np.empty((0, 3)),
        mpcts=np.concatenate(mpct_chunks) if mpct_chunks else np.empty((0,)),
    )
    return curated_dir, qc_dir


#: One qualifying failure transition (cluster 1) and one stable control
#: (cluster 2), under the shipped constants: references at ~2%, failing at
#: ~25%, four index-contiguous nominal-1,000 windows spanning 399.9 s.
FAILURE_AND_CONTROL_SPEC = {1: [0.02, 0.02, 0.25, 0.25], 2: [0.02, 0.025, 0.03, 0.02]}


def _inspect_ready_root(tmp_path: Path, spec=None, **fixture_kwargs):
    """Run a real `inventory` then `select` on a synthetic fixture."""
    curated, qc_dir = _write_replay_fixture(
        tmp_path, FAILURE_AND_CONTROL_SPEC if spec is None else spec, **fixture_kwargs
    )
    config_path = _write_config(
        tmp_path, curated, qc_dir, tmp_path / "recording",
        sampling_frequency_hz=REPLAY_FS, duration_s=2000.0,
    )
    out_root = tmp_path / "out"
    run_inventory(config_path, out_root)
    selection = run_select(config_path, out_root)
    return config_path, out_root, selection


def _case_windows_table(out_root: Path) -> pd.DataFrame:
    return pd.read_csv(out_root / "case_windows.csv")


def test_inspect_replays_every_frozen_case_window_keeping_both_counts(tmp_path):
    """End-to-end: the frozen cases, and only those, are replayed -- with the
    999 historical / 1,000 exact sample counts preserved on every row."""
    _, out_root, selection = _inspect_ready_root(tmp_path)
    assert [c["case_id"] for c in selection["cases"]] == [
        "synthetic__c1__failure1", "synthetic__c2__control1"
    ]

    manifest = run_inspect(out_root / "selection.json", out_root)
    assert manifest["stage"] == "inspect" and manifest["status"] == "complete"

    table = _case_windows_table(out_root)
    assert list(table.columns) == list(CASE_WINDOWS_COLUMNS)
    assert len(table) == 8
    assert (table["historical_count"] == 999).all()
    assert (table["exact_count"] == 1000).all()
    assert (table["i1"] - table["i0"] == 999).all()
    assert list(table["window_ordinal"]) == [0, 1, 2, 3] * 2
    assert list(table["window_role"]) == (
        ["reference", "reference", "failing", "failing"] + ["control"] * 4
    )
    # frozen order, nothing added, dropped or re-ranked
    assert list(dict.fromkeys(table["case_id"])) == [c["case_id"] for c in selection["cases"]]

    assert manifest["inspect"]["n_cases"] == 2
    assert manifest["inspect"]["n_case_windows"] == 8
    assert manifest["inspect"]["case_windows_sha256"] == sha256_file(
        out_root / "case_windows.csv")
    assert manifest["inspect"]["selection_sha256"] == selection["selection_sha256"]


def test_inspect_reproduces_every_cached_missing_pct_within_the_frozen_tolerance(tmp_path):
    """The replayed historical estimate must reproduce the cached one; the
    cache here was produced by production's own fitter over these arrays."""
    _, out_root, _ = _inspect_ready_root(tmp_path)
    manifest = run_inspect(out_root / "selection.json", out_root)
    table = _case_windows_table(out_root)

    assert (table["status"] == CASE_WINDOW_REPRODUCED).all()
    assert table["status_reason"].isna().all()
    assert (table["reproduction_abs_diff_pp"] <= REPRODUCTION_ATOL_PP).all()
    np.testing.assert_allclose(
        table["replayed_historical_missing_pct"], table["cached_missing_pct"],
        rtol=1e-6, atol=1e-6,
    )
    assert (table["case_status"] == CASE_STATUS_STABLE).all()
    assert not table["unstable_reproduction_mismatch"].any()
    assert not table["unstable_under_exact_indexing"].any()
    assert manifest["inspect"]["reproduction_mismatch_cases"] == []
    assert manifest["inspect"]["unstable_under_exact_indexing"] == []
    # the cached values are the ones the audit is about: a real transition
    assert table.loc[table["window_role"] == "reference", "cached_missing_pct"].max() <= 5.0
    assert table.loc[table["window_role"] == "failing", "cached_missing_pct"].min() >= 15.0


def test_the_final_window_amplitude_reaches_only_the_exact_fit_through_inspect(tmp_path):
    """Layer 2's slice contract, carried end-to-end.

    The amplitude at a window's inclusive endpoint i1 is fit by nothing
    historically: `amps[i0:i1]` stops before it and the next window starts at
    i1+1. Replacing it with a value far below the window's own minimum must
    therefore leave the historical fit reproducing the cache exactly, while
    moving the exact-1,000 fit (whose x_min is that new value) to near zero.
    """
    _, out_root, _ = _inspect_ready_root(
        tmp_path, final_amp_overrides={(1, 0): 5.0},   # reference window 0 of the failure case
    )
    run_inspect(out_root / "selection.json", out_root)
    table = _case_windows_table(out_root)

    touched = table[(table["cluster_id"] == 1) & (table["window_ordinal"] == 0)].iloc[0]
    assert touched["status"] == CASE_WINDOW_REPRODUCED
    assert touched["reproduction_abs_diff_pp"] <= REPRODUCTION_ATOL_PP
    assert touched["exact_missing_pct"] < 0.01
    assert touched["replayed_historical_missing_pct"] > 1.5
    assert touched["historical_count"] == 999 and touched["exact_count"] == 1000

    # untouched windows still differ between the two fits -- the extra sample
    # is genuinely in the exact fit everywhere, not only where it was rigged
    others = table[~((table["cluster_id"] == 1) & (table["window_ordinal"] == 0))]
    assert (others["exact_missing_pct"] != others["replayed_historical_missing_pct"]).all()
    # a ~2% reference window staying under 5% keeps the case eligible
    assert (table["case_status"] == CASE_STATUS_STABLE).all()


def test_exact_indexing_sensitivity_is_flagged_without_touching_the_selection(tmp_path):
    """The exact-1,000 estimate would disqualify a failing window; the case is
    flagged and reported, and the frozen selection is left exactly as it was."""
    _, out_root, selection = _inspect_ready_root(
        tmp_path, final_amp_overrides={(1, 3): 5.0},   # failing window 3 of the failure case
    )
    frozen_bytes = (out_root / "selection.json").read_bytes()

    manifest = run_inspect(out_root / "selection.json", out_root)
    table = _case_windows_table(out_root)

    failure = table[table["case_id"] == "synthetic__c1__failure1"]
    assert failure["unstable_under_exact_indexing"].all()
    assert (failure["case_status"] == CASE_STATUS_UNSTABLE).all()
    assert (failure["exact_eligibility_reason"] == "failing_below_min_missing_pct").all()
    # ...but every historical value still reproduced: this is a sensitivity,
    # not an input mismatch
    assert (failure["status"] == CASE_WINDOW_REPRODUCED).all()
    assert not failure["unstable_reproduction_mismatch"].any()

    control = table[table["case_id"] == "synthetic__c2__control1"]
    assert not control["unstable_under_exact_indexing"].any()
    assert (control["case_status"] == CASE_STATUS_STABLE).all()

    assert manifest["inspect"]["unstable_under_exact_indexing"] == ["synthetic__c1__failure1"]

    # nothing re-selected, dropped or re-ranked, and selection.json untouched
    assert list(dict.fromkeys(table["case_id"])) == [c["case_id"] for c in selection["cases"]]
    assert (out_root / "selection.json").read_bytes() == frozen_bytes


def test_a_perturbed_cache_is_recorded_as_a_reproduction_mismatch(tmp_path):
    """A cached value its own inputs no longer produce must be recorded as an
    input/runtime mismatch and mark the case unstable -- never absorbed by a
    wider tolerance, and never a reason to drop the case."""
    delta = 1e-3   # far under any selection threshold, far over the 1e-6 tolerance
    _, out_root, selection = _inspect_ready_root(
        tmp_path, mpct_perturbation={(1, 1): delta},
    )
    manifest = run_inspect(out_root / "selection.json", out_root)
    table = _case_windows_table(out_root)

    bad = table[(table["cluster_id"] == 1) & (table["window_ordinal"] == 1)].iloc[0]
    assert bad["status"] == CASE_WINDOW_REPRODUCTION_MISMATCH
    assert "input/runtime mismatch" in bad["status_reason"]
    assert bad["reproduction_abs_diff_pp"] == pytest.approx(delta, rel=1e-9)
    assert bad["replayed_historical_missing_pct"] != bad["cached_missing_pct"]

    failure = table[table["case_id"] == "synthetic__c1__failure1"]
    assert failure["unstable_reproduction_mismatch"].all()
    assert (failure["case_status"] == CASE_STATUS_UNSTABLE).all()
    # only the perturbed window is a mismatch; its three siblings reproduce
    assert list(failure["status"]).count(CASE_WINDOW_REPRODUCED) == 3

    assert manifest["inspect"]["reproduction_mismatch_cases"] == ["synthetic__c1__failure1"]
    assert manifest["inspect"]["cases"][0]["mismatched_window_ordinals"] == [1]
    assert manifest["inspect"]["reproduction_tolerance"] == {"rtol": 1e-6, "atol_pp": 1e-6}
    # the case is still present, in its frozen position
    assert list(dict.fromkeys(table["case_id"])) == [c["case_id"] for c in selection["cases"]]
    assert manifest["status"] == "complete"


def test_inspect_writes_a_headed_empty_table_when_a_gap_left_nothing_to_select(tmp_path):
    """A >10 s gap splits production's block, so no four consecutive windows
    tile one gap-free block and nothing is selected. `inspect` must still
    complete and still write case_windows.csv with its header."""
    spec = {1: [0.02, 0.02, 0.25, 0.25]}
    _, out_root, selection = _inspect_ready_root(
        tmp_path, spec=spec, gap_after_window={1: {1}},
    )
    assert selection["cases"] == []
    assert selection["exclusion_counts"]["failure_runs_by_reason"]["gap_between_windows"] == 1

    manifest = run_inspect(out_root / "selection.json", out_root)
    assert manifest["status"] == "complete"
    assert manifest["inspect"]["n_cases"] == 0
    assert manifest["inspect"]["n_case_windows"] == 0
    assert manifest["inspect"]["unstable_under_exact_indexing"] == []

    text = (out_root / "case_windows.csv").read_text()
    assert text.strip() == ",".join(CASE_WINDOWS_COLUMNS)
    assert len(_case_windows_table(out_root)) == 0


def test_inspect_cli_replays_and_reports(tmp_path, capsys):
    _, out_root, selection = _inspect_ready_root(tmp_path)
    assert main([
        "inspect", "--selection", str(out_root / "selection.json"),
        "--out-root", str(out_root),
    ]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["selection_sha256"] == selection["selection_sha256"]
    assert [c["case_id"] for c in printed["cases"]] == [
        c["case_id"] for c in selection["cases"]
    ]
    assert (out_root / "case_windows.csv").exists()


# --------------------------------------------------------------------------- #
# layer 4a: the gate. Every refusal below must leave `select` as it was.
# --------------------------------------------------------------------------- #
def test_inspect_refuses_a_tampered_selection(tmp_path):
    _, out_root, _ = _inspect_ready_root(tmp_path)
    selection_path = out_root / "selection.json"
    payload = json.loads(selection_path.read_text())
    payload["cases"][0]["windows"][0]["missing_pct"] = 99.0     # digest NOT re-stamped
    selection_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="selection_sha256 mismatch"):
        run_inspect(selection_path, out_root)
    assert not (out_root / "case_windows.csv").exists()
    assert json.loads((out_root / "manifest.json").read_text())["stage"] == "select"


def test_inspect_refuses_a_self_consistent_selection_that_contradicts_windows_csv(tmp_path):
    """selection.json carries its own digest, so a forged one can be made
    self-consistent -- and the manifest re-stamped to match. The inventory's
    attested windows.csv cannot be, so it stays the authority for every cached
    value the replay reproduces."""
    _, out_root, _ = _inspect_ready_root(tmp_path)
    selection_path = out_root / "selection.json"
    manifest_path = out_root / "manifest.json"

    payload = json.loads(selection_path.read_text())
    payload["cases"][0]["windows"][0]["missing_pct"] = 4.75
    payload["selection_sha256"] = canonical_selection_digest(payload)
    selection_path.write_text(json.dumps(payload))
    manifest = json.loads(manifest_path.read_text())
    manifest["select"]["selection_sha256"] = payload["selection_sha256"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="disagrees with the attested windows.csv"):
        run_inspect(selection_path, out_root)
    assert not (out_root / "case_windows.csv").exists()


def test_inspect_refuses_a_windows_csv_that_moved_since_the_freeze(tmp_path):
    _, out_root, _ = _inspect_ready_root(tmp_path)
    edited = pd.read_csv(out_root / "windows.csv")
    edited.loc[0, "missing_pct"] = 4.9
    edited.to_csv(out_root / "windows.csv", index=False)

    with pytest.raises(RuntimeError, match="windows.csv sha256 mismatch"):
        run_inspect(out_root / "selection.json", out_root)
    assert not (out_root / "case_windows.csv").exists()


def test_inspect_names_the_config_when_it_is_the_input_that_moved(tmp_path):
    config_path, out_root, _ = _inspect_ready_root(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["sorts"][0]["duration_s"] = 2222.0      # any change moves the hash
    config_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match=r"recorded input\(s\) moved") as exc:
        run_inspect(out_root / "selection.json", out_root)
    message = str(exc.value)
    assert str(config_path) in message
    # only the input that actually moved is named
    assert "pipeline.truncation" not in message
    assert str(REPO_ROOT / "testing/luke_amplitude_dropout_audit.py") not in message
    assert not (out_root / "case_windows.csv").exists()


def test_inspect_names_the_source_file_when_the_working_tree_code_moved(tmp_path, monkeypatch):
    """A dirty working tree is expected, so the Git commit proves nothing: the
    recorded working-tree source hashes are what must still hold."""
    import testing.luke_amplitude_dropout_audit as mod

    stand_in = tmp_path / "module_under_audit.py"
    stand_in.write_text("# the code that froze the selection\n")
    monkeypatch.setattr(mod, "audit_source_files", lambda: {"module": stand_in})

    _, out_root, _ = _inspect_ready_root(tmp_path)
    stand_in.write_text("# edited after the freeze\n")

    with pytest.raises(RuntimeError, match=r"recorded input\(s\) moved.*module") as exc:
        run_inspect(out_root / "selection.json", out_root)
    assert str(stand_in) in str(exc.value)
    assert not (out_root / "case_windows.csv").exists()


def test_inspect_refuses_a_curated_array_that_moved_since_the_inventory(tmp_path):
    _, out_root, _ = _inspect_ready_root(tmp_path)
    curated = tmp_path / "curated"
    full_st = np.load(curated / "full_st.npy")
    full_st[0, 2] += 1.0                     # one amplitude, nothing else
    np.save(curated / "full_st.npy", full_st)

    with pytest.raises(RuntimeError, match="curated array"):
        run_inspect(out_root / "selection.json", out_root)
    assert not (out_root / "case_windows.csv").exists()


def test_inspect_twice_is_refused(tmp_path):
    _, out_root, _ = _inspect_ready_root(tmp_path)
    first = run_inspect(out_root / "selection.json", out_root)
    before = (out_root / "case_windows.csv").read_bytes()
    with pytest.raises(RuntimeError, match="existing replay"):
        run_inspect(out_root / "selection.json", out_root)
    assert (out_root / "case_windows.csv").read_bytes() == before
    assert json.loads((out_root / "manifest.json").read_text())["inspect"] == first["inspect"]


def test_inspect_refuses_a_selection_from_another_output_root(tmp_path):
    _, out_root, _ = _inspect_ready_root(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    copied = elsewhere / "selection.json"
    copied.write_bytes((out_root / "selection.json").read_bytes())

    with pytest.raises(RuntimeError, match="does not live in --out-root"):
        run_inspect(copied, out_root)


def test_inspect_refuses_an_out_root_that_never_ran_select(tmp_path):
    """Only `inventory` has run here, so there is no frozen selection to
    verify and no completed select stage to extend."""
    curated, qc_dir = _write_replay_fixture(tmp_path, FAILURE_AND_CONTROL_SPEC)
    config_path = _write_config(
        tmp_path, curated, qc_dir, tmp_path / "recording",
        sampling_frequency_hz=REPLAY_FS, duration_s=2000.0,
    )
    out_root = tmp_path / "out"
    run_inventory(config_path, out_root)
    with pytest.raises(RuntimeError, match="no frozen selection"):
        run_inspect(out_root / "selection.json", out_root)


def test_inspect_refuses_a_manifest_still_at_the_inventory_stage(tmp_path):
    _, out_root, _ = _inspect_ready_root(tmp_path)
    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stage"] = "inventory"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="expected 'select'"):
        run_inspect(out_root / "selection.json", out_root)


def test_inspect_still_refuses_an_unsafe_out_root(tmp_path):
    _, out_root, _ = _inspect_ready_root(tmp_path)
    with pytest.raises(ValueError, match="/mnt"):
        run_inspect(out_root / "selection.json", Path("/mnt/NPX/scratch_inspect"))


def test_inspect_writes_a_failed_manifest_and_no_table_when_the_replay_raises(tmp_path):
    import testing.luke_amplitude_dropout_audit as mod

    _, out_root, _ = _inspect_ready_root(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("fitter unavailable")

    original = mod.historical_exact_fit
    mod.historical_exact_fit = boom
    try:
        with pytest.raises(RuntimeError, match="fitter unavailable"):
            run_inspect(out_root / "selection.json", out_root)
    finally:
        mod.historical_exact_fit = original

    manifest = json.loads((out_root / "manifest.json").read_text())
    assert manifest["stage"] == "inspect" and manifest["status"] == "failed"
    assert "fitter unavailable" in manifest["failure_reason"]
    assert not (out_root / "case_windows.csv").exists()


# --------------------------------------------------------------------------- #
# layer 4a: unit contracts underneath the stage
# --------------------------------------------------------------------------- #
def test_cluster_amplitude_sequence_refuses_a_source_that_was_not_time_ordered():
    """Cached window_blocks index that cluster's own time-ordered sequence.
    Replaying them against a defensively re-sorted array would silently
    reinterpret the indices, so an unordered source is refused instead."""
    curated = CuratedArrays(
        sort_id="s1",
        times=np.array([0, 10, 20], dtype=np.int64),
        clusters=np.array([7, 7, 7], dtype=np.int64),
        amplitudes=np.array([1.0, 2.0, 3.0]),
        row_id=np.array([0, 1, 2], dtype=np.int64),
        was_time_ordered=False,
    )
    with pytest.raises(ValueError, match="not already time-ordered"):
        cluster_amplitude_sequence(curated, 7)


def test_cluster_amplitude_sequence_selects_by_id_value_not_array_position():
    curated = CuratedArrays(
        sort_id="s1",
        times=np.array([0, 10, 20, 30], dtype=np.int64),
        clusters=np.array([41, 7, 41, 7], dtype=np.int64),   # noncontiguous ids
        amplitudes=np.array([1.0, 2.0, 3.0, 4.0]),
        row_id=np.arange(4, dtype=np.int64),
        was_time_ordered=True,
    )
    times, amps = cluster_amplitude_sequence(curated, 41)
    assert list(times) == [0, 20]
    assert list(amps) == [1.0, 3.0]
    with pytest.raises(ValueError, match="no spikes in the curated arrays"):
        cluster_amplitude_sequence(curated, 999)


def _sensitivity_case(role: str, cached_mpcts: list[float]) -> dict:
    run = _run_rows(cached_mpcts)
    roles = (["reference"] * 2 + ["failing"] * 2) if role == "failure" else ["control"] * 4
    return {
        "case_id": "case", "sort_id": "s1", "cluster_id": 0, "role": role,
        "windows": [
            {**row, "position": i, "window_role": roles[i]} for i, row in enumerate(run)
        ],
    }


def _replays(exact_mpcts: list[float], popt=(20.0, 1.0, 1.0)) -> list[dict]:
    return [{"exact_missing_pct": v, "exact_popt": list(popt)} for v in exact_mpcts]


def test_exact_indexing_eligibility_reports_the_frozen_runs_own_verdict():
    case = _sensitivity_case("failure", [2.0, 2.0, 25.0, 25.0])
    assert exact_indexing_eligibility(case, _replays([2.0, 2.0, 25.0, 25.0]), CONSTANTS) \
        == "qualified"
    # the exact estimate drops one failing window under the 15% gate
    assert exact_indexing_eligibility(case, _replays([2.0, 2.0, 25.0, 10.0]), CONSTANTS) \
        == "failing_below_min_missing_pct"
    # ...and a boundary-pinned exact fit is censored, not a 50% measurement
    assert exact_indexing_eligibility(case, _replays([2.0, 2.0, 25.0, 50.0]), CONSTANTS) \
        == "status_not_finite_interior"


def test_exact_indexing_eligibility_uses_the_control_test_for_a_control_case():
    case = _sensitivity_case("control", [2.0, 2.0, 2.0, 2.0])
    assert exact_indexing_eligibility(case, _replays([2.0, 2.5, 3.0, 2.0]), CONSTANTS) \
        == "qualified"
    assert exact_indexing_eligibility(case, _replays([2.0, 2.0, 2.0, 5.5]), CONSTANTS) \
        == "above_max_missing_pct"
    assert exact_indexing_eligibility(case, _replays([1.0, 1.0, 1.0, 4.5]), CONSTANTS) \
        == "range_above_max_pp"


def test_exact_indexing_eligibility_refuses_an_unknown_case_role():
    """Falling through to the control test for an unrecognised role would
    quietly report the wrong sensitivity verdict."""
    case = _sensitivity_case("failure", [2.0, 2.0, 25.0, 25.0])
    case["role"] = "something_else"
    with pytest.raises(ValueError, match="unknown case role"):
        exact_indexing_eligibility(case, _replays([2.0, 2.0, 25.0, 25.0]), CONSTANTS)


def test_load_attested_selection_refuses_a_foreign_or_unfrozen_file(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"schema": "other-v9", "stage": "select"}))
    with pytest.raises(RuntimeError, match="schema"):
        load_attested_selection(path)

    path.write_text(json.dumps({"schema": SCHEMA, "stage": "inventory"}))
    with pytest.raises(RuntimeError, match="expected 'select'"):
        load_attested_selection(path)

    payload = {"schema": SCHEMA, "stage": "select", "cases": [], "selection_sha256": "0" * 64}
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="selection_sha256 mismatch"):
        load_attested_selection(path)

    payload["selection_sha256"] = canonical_selection_digest(payload)
    path.write_text(json.dumps(payload))
    assert load_attested_selection(path)["selection_sha256"] == payload["selection_sha256"]


def test_audit_source_files_covers_the_fitter_and_this_module():
    files = audit_source_files()
    assert set(files) == {"module", "pipeline.truncation"}
    assert files["module"].name == "luke_amplitude_dropout_audit.py"
    assert all(p.exists() for p in files.values())


# --------------------------------------------------------------------------- #
# real-data smoke test (acceptance test 5's spirit) -- skipped if /mnt absent
# --------------------------------------------------------------------------- #
def _real_data_available() -> bool:
    if not REAL_CONFIG.exists():
        return False
    cfg = load_config(REAL_CONFIG)
    return all(
        (Path(s.curated) / "spike_times.npy").exists()
        and (Path(s.qc_dir) / "amp_truncation" / "truncation_qc.npz").exists()
        for s in cfg.sorts
    )


@pytest.mark.skipif(not _real_data_available(), reason="Luke 20250804 /mnt data not available")
def test_run_inventory_real_data_smoke(tmp_path):
    out_root = tmp_path / "real_inventory"
    manifest = run_inventory(REAL_CONFIG, out_root)
    assert manifest["status"] == "complete"
    for sort_id, info in manifest["sorts"].items():
        assert info["was_time_ordered"] is True
        assert info["n_cached_windows"] > 0
    table = pd.read_csv(out_root / "windows.csv")
    assert set(table["status"].unique()) <= {
        STATUS_INVALID_INPUT, STATUS_NO_FIT, STATUS_NONFINITE_FIT,
        STATUS_BOUNDARY_PINNED, STATUS_FINITE_INTERIOR,
    }
    assert (table["status"] == STATUS_FINITE_INTERIOR).any()
    assert (table["status"] == STATUS_BOUNDARY_PINNED).any()


@pytest.mark.skipif(not _real_data_available(), reason="Luke 20250804 /mnt data not available")
def test_historical_replay_reproduces_one_real_cached_window():
    cfg = load_config(REAL_CONFIG)
    s = cfg.by_id("rescue_luke0804_v2v1_g0_imec0")
    curated = load_curated_arrays(s.sort_id, s.curated)
    cached = load_cached_truncation_qc(s.sort_id, s.qc_dir)

    # find a finite-interior cached window to replay
    from testing.luke_amplitude_dropout_audit import is_saturated
    finite = np.flatnonzero(~is_saturated(cached.mpcts) & np.isfinite(cached.mpcts))
    assert finite.size > 0
    r = int(finite[0])
    cid = int(cached.cid[r])
    i0, i1 = int(cached.window_blocks[r, 0]), int(cached.window_blocks[r, 1])

    positions = np.flatnonzero(curated.clusters == cid)
    cluster_amps = curated.amplitudes[positions]

    result = historical_exact_fit(cluster_amps, i0, i1)
    assert result["historical_count"] == i1 - i0
    assert abs(result["historical_missing_pct"] - float(cached.mpcts[r])) < 1e-6
