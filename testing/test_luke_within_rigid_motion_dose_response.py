import json

import numpy as np
import pandas as pd
import pytest

import testing.luke_within_rigid_motion_dose_response as m
from testing.luke_within_rigid_motion_dose_response import (
    ALL_ESTIMATORS,
    CONCORDANT_ESTIMATORS,
    ENDPOINT_KEYS,
    N_WINDOWS,
    SelectedWindow,
    consensus_dose_table,
    dose_response,
    load_frozen_list,
    mann_kendall,
    partial_corr_spearman,
    qualify_units,
    select_windows,
    spearman_ci,
    write_frozen_list,
    _cluster_amp_and_depth,
    _context_event_metrics,
    _fragmentation,
    _similar_pairs,
    _spec_for,
    _waveform_stability_median,
)


# --------------------------------------------------------------------------- #
# synthetic increment-1 CSV
# --------------------------------------------------------------------------- #
def _fake_csv(tmp_path, n=87, seed=0, medicine_agrees=False):
    rng = np.random.default_rng(seed)
    origin = {"medicine": 3057.7, "ks-motion": 3058.7, "dredge-motion": 3058.2,
              "decentralized-motion": 3058.7}
    dt = {"medicine": 1.0, "ks-motion": 2.0, "dredge-motion": 1.0, "decentralized-motion": 2.0}
    # a shared "true" motion order; concordant estimators track it, medicine doesn't
    truth = np.linspace(2.0, 25.0, n)
    perm_med = truth if medicine_agrees else rng.permutation(truth)
    rows = []
    for est in ALL_ESTIMATORS:
        base = truth if est != "medicine" else perm_med
        exc = base + rng.normal(0, 0.8, n)
        spd = 0.02 * base + rng.normal(0, 0.03, n) + 0.2
        for k in range(n):
            rows.append(dict(
                dataset="Luke", probe="imec0", estimator=est,
                window_start_native_s=origin[est] + k * 120.0,
                window_start_recording_s=k * 120.0,
                time_origin_native_s=origin[est], window_duration_s=120.0,
                time_interval_id=k * 120,  # matches the real CSV's rounding on a 120 s stride
                rigid_excursion_um=float(exc[k]), nonrigid_grad_um_per_mm=1.5,
                p95_nonrigid_grad_um_per_mm=2.0, rigid_speed_um_s=float(max(spd[k], 0.01)),
                finite_fraction=1.0, n_time_bins=int(120 / dt[est]), n_depth_bins=2,
                depth_span_um=4074.0, dt_median_s=dt[est], max_time_gap_s=dt[est],
            ))
    # a QC-failing interval on ks-motion only -> the interval must be dropped
    rows.append({**rows[0], "estimator": "ks-motion", "time_interval_id": 999 * 120,
                 "window_start_native_s": origin["ks-motion"] + 999 * 120.0,
                 "window_start_recording_s": 999 * 120.0, "finite_fraction": 0.4})
    for est in ("dredge-motion", "decentralized-motion", "medicine"):
        rows.append({**rows[0], "estimator": est, "time_interval_id": 999 * 120,
                     "window_start_native_s": origin[est] + 999 * 120.0,
                     "window_start_recording_s": 999 * 120.0})
    rows.append({**rows[0], "dataset": "Yates", "probe": "shank1"})
    p = tmp_path / "window_signatures.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# phase 1 -- consensus dose
# --------------------------------------------------------------------------- #
def test_consensus_table_joins_on_interval_and_drops_incomplete(tmp_path):
    t = consensus_dose_table(_fake_csv(tmp_path))
    assert len(t) == 87  # the 999*120 interval fails ks QC -> dropped
    assert 999 * 120 not in set(t["time_interval_id"])
    assert {"exc_consensus_rank", "spd_consensus_rank"} <= set(t.columns)
    for e in ALL_ESTIMATORS:
        assert f"exc_{e}" in t.columns
    assert t["exc_consensus_rank"].between(0, 1).all()


def test_consensus_rank_tracks_concordant_not_medicine(tmp_path):
    from scipy.stats import spearmanr

    t = consensus_dose_table(_fake_csv(tmp_path, seed=1))
    for e in CONCORDANT_ESTIMATORS:
        assert spearmanr(t["exc_consensus_rank"], t[f"exc_{e}"]).statistic > 0.7
    assert abs(spearmanr(t["exc_consensus_rank"], t["exc_medicine"]).statistic) < 0.4


def test_select_windows_deterministic_and_spans_range(tmp_path):
    csv = _fake_csv(tmp_path)
    a, b = select_windows(csv), select_windows(csv)
    assert [w.time_interval_id for w in a] == [w.time_interval_id for w in b]
    assert len(a) == N_WINDOWS
    assert a[0].exc_consensus_rank < 0.1 and a[-1].exc_consensus_rank > 0.9
    assert a[0].snippet_start_s == a[0].time_interval_id  # recording-relative == interval id here


def test_select_windows_raises_when_too_few(tmp_path):
    with pytest.raises(RuntimeError, match="need"):
        select_windows(_fake_csv(tmp_path, n=10))


def test_frozen_list_records_provenance_and_validates(tmp_path):
    csv = _fake_csv(tmp_path)
    windows = select_windows(csv)
    path = tmp_path / "frozen.json"
    write_frozen_list(windows, csv, path)
    payload = json.loads(path.read_text())
    assert payload["source_csv_sha256"] and payload["git_commit"]
    assert payload["time_interval_ids"] == [w.time_interval_id for w in windows]

    loaded = load_frozen_list(path, csv)
    assert [w.time_interval_id for w in loaded] == [w.time_interval_id for w in windows]

    with pytest.raises(RuntimeError, match="frozen once"):
        write_frozen_list(windows, csv, path)

    csv.write_text(csv.read_text() + "\n")  # tamper
    with pytest.raises(RuntimeError, match="changed since"):
        load_frozen_list(path, csv)


# --------------------------------------------------------------------------- #
# phase 2 -- spec / dir resolution (the KeyError bug)
# --------------------------------------------------------------------------- #
def test_spec_and_dir_resolution(tmp_path, monkeypatch):
    w = SelectedWindow(time_interval_id=1200, snippet_start_s=1200.0,
                       exc_consensus_rank=0.5, spd_consensus_rank=0.5,
                       exc_by_estimator={e: 5.0 for e in ALL_ESTIMATORS},
                       spd_by_estimator={e: 0.3 for e in ALL_ESTIMATORS})
    spec = _spec_for(w)
    assert spec.start_s == 1200.0 and spec.channel_count == 384
    assert spec.duration_s == 120.0
    monkeypatch.setattr(m, "_snippet_dir_for", lambda s: tmp_path / s.directory_name)
    d = m._snippet_dir_for(spec)
    assert d.name.startswith("rigid_dose_iv1200-")


# --------------------------------------------------------------------------- #
# phase 3 -- endpoints
# --------------------------------------------------------------------------- #
def _train(rate_hz, dur_s, fs, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur_s, 1.0 / rate_hz)
    t = t + rng.normal(0, jitter, t.size)
    return np.sort((t * fs).astype(np.int64))


def test_qualify_units_applies_every_frozen_gate():
    fs, dur = 30000.0, 120.0
    clean = _train(5.0, dur, fs, jitter=0.05, seed=1)          # 0: ~600 spikes, clean
    few = _train(0.5, dur, fs, seed=2)                          # 1: ~60 spikes
    contaminated = np.sort(np.arange(0, int(dur * fs), int(fs / 1000)))  # 2: 1 kHz -> ISI 1 ms < 1.5
    early = _train(6.0, dur, fs, seed=4)
    early = early[early < 40 * fs]                              # 3: present only first 40 s
    st = np.concatenate([clean, few, contaminated, early])
    cl = np.concatenate([np.full(a.size, i) for i, a in enumerate([clean, few, contaminated, early])])
    amp = {i: 40.0 for i in range(4)}
    dep = {i: 1000.0 for i in range(4)}

    q = qualify_units(st, cl, amp, dep, fs, dur)
    assert q["qualified"] == [0]
    by_c = {u["cluster"]: u for u in q["per_unit"]}
    assert by_c[1]["n_spikes"] < m.QUAL_MIN_SPIKES
    assert by_c[2]["rv_fraction"] > m.QUAL_RV_CEILING
    assert by_c[3]["presence_bins"] < m.QUAL_PRESENCE_MIN_BINS

    amp[0] = 5.0
    assert qualify_units(st, cl, amp, dep, fs, dur)["qualified"] == []


def test_similar_pairs_and_fragmentation_are_label_free():
    templates = np.zeros((3, 40, 4))
    templates[0, 20, 0] = -10.0
    templates[1, 20, 0] = -9.9   # ~identical to 0
    templates[2, 20, 3] = -10.0  # different channel
    depth = {0: 1000.0, 1: 1000.0, 2: 1000.0}
    assert _similar_pairs([0, 1, 2], templates, depth) == 1

    fs = 30000.0
    a = np.arange(0, 3_000_000, 6000)          # 5 Hz
    b = a + 3000                                # interleaved, never coincident
    frag = _fragmentation([0, 1], {0: a, 1: b}, {0: 1000.0, 1: 1000.0}, fs)
    assert frag["n_fragment_pairs"] == 1 and frag["E8_fragmentation_index"] == 1.0


# --------------------------------------------------------------------------- #
# phase 3 -- trace-level endpoints on a small synthetic recording
# --------------------------------------------------------------------------- #
def _rec_with_spikes(fs=30000.0, n_ch=16, dur_s=8.0, spike_channels=(4,), rate_hz=20.0):
    from spikeinterface.core import NumpyRecording

    rng = np.random.default_rng(0)
    n = int(fs * dur_s)
    tr = rng.normal(0, 3.0, size=(n, n_ch)).astype("float32")
    times = np.arange(300, n - 300, int(fs / rate_hz))
    for t in times:
        for c in spike_channels:
            tr[t, c] -= 90.0
            tr[t - 1, c] -= 45.0
            tr[t + 1, c] -= 45.0
    rec = NumpyRecording([tr], sampling_frequency=fs)
    rec.set_dummy_probe_from_locations(
        np.column_stack([np.zeros(n_ch), np.arange(n_ch) * 20.0]).astype(float)
    )
    return rec, times, np.column_stack([np.zeros(n_ch), np.arange(n_ch) * 20.0])


def test_cluster_amplitude_is_microvolts_with_gain():
    rec, times, geom = _rec_with_spikes(spike_channels=(4,))
    templates = np.zeros((1, 60, 16))
    templates[0, 30, 4] = -1.0                       # peak on channel 4
    cl = np.zeros(times.size, dtype=int)
    gain = 2.0
    amp, depth, peak_ch = _cluster_amp_and_depth(
        cl, times, templates, None, geom, rec, gain, 30000.0
    )
    assert peak_ch[0] == 4 and depth[0] == 80.0
    # STA trough ~ -90 counts on the peak channel -> ~180 uV at gain 2
    assert 120.0 < amp[0] < 240.0


def test_context_c2_requires_spatial_proximity():
    rec, times, geom = _rec_with_spikes(spike_channels=(4,))
    fs = 30000.0
    # qualified unit near channel 4 (depth 80) vs one far away (depth 280)
    near = {0: times}
    depth_near = {0: 80.0}
    depth_far = {0: 280.0}
    _c1, c2_near = _context_event_metrics(rec, near, depth_near, geom, 0.34, fs)
    _c1, c2_far = _context_event_metrics(rec, near, depth_far, geom, 0.34, fs)
    assert c2_near > 0.3           # the real events (on/near channel 4) are counted
    assert c2_far < 0.08           # same spikes in time, but > 40 um away -> not counted
    assert c2_near > 4 * c2_far


def test_waveform_stability_uses_one_channel_set_per_unit():
    rec, times, geom = _rec_with_spikes(spike_channels=(4,), dur_s=12.0, rate_hz=30.0)
    peak_ch = {0: 4}
    stab = _waveform_stability_median([0], {0: times}, peak_ch, rec, geom, 30000.0)
    assert 0.8 < stab <= 1.0       # a stationary synthetic waveform is highly self-similar


# --------------------------------------------------------------------------- #
# phase 4 -- statistics
# --------------------------------------------------------------------------- #
def test_spearman_ci_monotone_and_degenerate():
    r = spearman_ci(np.arange(24.0), np.arange(24.0) ** 1.3, n_boot=200)
    assert r["rho"] == pytest.approx(1.0) and r["ci_lo"] > 0.8
    assert np.isnan(spearman_ci(np.ones(10), np.arange(10.0))["rho"])


def test_mann_kendall_tie_corrected_matches_reference():
    # y with a clear increasing trend and repeated values (ties)
    y = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], float)
    x = np.arange(y.size, dtype=float)
    mk = mann_kendall(x, y, n_perm=2000)
    # S for a perfectly ordered tied sequence: pairs concordant = C(12,2)-ties
    # 6 tie groups of 2 -> 6 ties; S = 66 - 6 = 60
    assert mk["S"] == 60
    assert 0.9 < mk["tau_b"] <= 1.0
    assert 0 < mk["p_perm"] < 0.01           # (k+1)/(n_perm+1) can never be 0
    assert mk["x_has_ties"] is False


def test_mann_kendall_handles_tied_dose():
    # tied x values must be treated as x ties, not arbitrarily ordered
    x = np.array([0, 0, 1, 1, 2, 2, 3, 3], float)
    y = np.array([1, 2, 2, 3, 3, 4, 4, 5], float)
    mk = mann_kendall(x, y, n_perm=1000)
    assert mk["x_has_ties"] is True
    assert mk["tau_b"] > 0
    # pairs within the same x group contribute sign(0)=0
    assert mk["S"] == sum(
        np.sign(x[j] - x[i]) * np.sign(y[j] - y[i])
        for i in range(8) for j in range(i + 1, 8)
    )


def test_partial_corr_spearman_matches_manual_formula():
    rng = np.random.default_rng(0)
    z = np.linspace(0, 1, 40)
    x = z + rng.normal(0, 0.1, 40)
    y = z + rng.normal(0, 0.1, 40)
    from scipy.stats import spearmanr

    rxy = spearmanr(x, y).statistic
    rxz = spearmanr(x, z).statistic
    ryz = spearmanr(y, z).statistic
    expected = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    assert partial_corr_spearman(y, x, z) == pytest.approx(expected, abs=0.03)


def test_partial_corr_removes_pure_confound():
    rng = np.random.default_rng(2)
    z = np.linspace(0, 1, 60)
    x = z + rng.normal(0, 0.03, 60)
    y_conf = 3 * z + rng.normal(0, 0.05, 60)   # driven by z only
    y_real = 3 * x + rng.normal(0, 0.05, 60)
    assert abs(partial_corr_spearman(y_conf, x, z)) < 0.4
    assert partial_corr_spearman(y_real, x, z) > 0.5


def _endpoint_frame(n=24, seed=0, e3_slope=-3.0, medicine_flips=False):
    rng = np.random.default_rng(seed)
    exc_rank = np.linspace(0.02, 1.0, n)
    rows = {
        "time_interval_id": np.arange(n) * 120,
        "snippet_start_s": np.arange(n) * 120.0,
        "exc_consensus_rank": exc_rank,
        "spd_consensus_rank": exc_rank + rng.normal(0, 0.05, n),
        "n_qualified": np.clip((30 + e3_slope * 10 * exc_rank).astype(int), 0, None),
        "E3_qualified_units_per_mm": 8 + e3_slope * exc_rank + rng.normal(0, 0.2, n),
        "E4_refractory_burden_median": 0.001 + 0.003 * exc_rank,
        "E5_similar_pairs_per_qualified_unit": 0.1 + 0.2 * exc_rank,
        "E6_waveform_stability_median": 0.99 - 0.1 * exc_rank,
        "E7_qualified_rate_hz_median": 5 + rng.normal(0, 0.5, n),
        "E8_fragmentation_index": 0.05 + 0.3 * exc_rank,
        "C1_detected_events_per_mm_per_s": 100 + rng.normal(0, 5, n),
        "C2_fraction_events_near_qualified": 0.8 - 0.2 * exc_rank,
    }
    for e in ALL_ESTIMATORS:
        if e == "medicine" and medicine_flips:
            rows[f"exc_{e}"] = -exc_rank + rng.normal(0, 0.02, n)
        else:
            rows[f"exc_{e}"] = exc_rank + rng.normal(0, 0.05, n)
        rows[f"spd_{e}"] = exc_rank + rng.normal(0, 0.05, n)
    return pd.DataFrame(rows)


def test_dose_response_primary_and_concordance():
    out = dose_response(_endpoint_frame(e3_slope=-3.0))
    assert out["primary_endpoint"] == "E3_qualified_units_per_mm"
    pb = out["primary"]
    assert pb["spearman_vs_excursion"]["rho"] < -0.8
    assert pb["exposure_validity"] == "resolved"
    assert pb["matches_prereg_direction"] is True
    # full pre-committed decision, not just the marginal sign
    r = pb["primary_decision_reasons"]
    assert set(r) == {"rho_negative", "ci_excludes_zero", "linear_partial_survives",
                      "quadratic_partial_survives", "exposure_resolved", "not_low_power"}
    assert pb["primary_supported"] == all(r.values())
    assert out["concordance_summary"].startswith("5/5")
    assert "dose_rank_cross_estimator" in out


def test_dose_response_exposure_unresolved_when_medicine_flips():
    out = dose_response(_endpoint_frame(medicine_flips=True))
    pb = out["primary"]
    # concordant three still agree, but MEDiCINe flips -> unresolved, per prespec
    assert pb["medicine_sign_agrees"] is False
    assert pb["exposure_validity"] == "unresolved"
    assert pb["primary_supported"] is False


def test_dose_response_no_association_when_flat():
    df = _endpoint_frame(e3_slope=0.0, seed=7)
    df["E3_qualified_units_per_mm"] = 8.0        # exactly flat -> no rank signal
    out = dose_response(df)
    assert out["primary"]["exposure_validity"] == "no_association"
    assert out["primary"]["primary_supported"] is False


def test_dose_response_low_power_blocks_primary():
    df = _endpoint_frame(e3_slope=-3.0)
    df.loc[0, "n_qualified"] = 2   # one underpowered window
    out = dose_response(df)
    assert out["low_power"] is True
    assert out["primary"]["primary_decision_reasons"]["not_low_power"] is False
    assert out["primary"]["primary_supported"] is False


def test_dose_response_validates_frozen_cohort():
    windows = [
        SelectedWindow(time_interval_id=i * 120, snippet_start_s=i * 120.0,
                       exc_consensus_rank=i / 23, spd_consensus_rank=i / 23,
                       exc_by_estimator={e: float(i) for e in ALL_ESTIMATORS},
                       spd_by_estimator={e: float(i) for e in ALL_ESTIMATORS})
        for i in range(24)
    ]
    df = _endpoint_frame()
    df["time_interval_id"] = [i * 120 for i in range(24)]
    df["snippet_start_s"] = [i * 120.0 for i in range(24)]
    df["exc_consensus_rank"] = [i / 23 for i in range(24)]
    dose_response(df, frozen_windows=windows)  # matches -> ok

    with pytest.raises(ValueError, match="!= frozen set|expected 24"):
        dose_response(df.iloc[:23], frozen_windows=windows)


def test_dose_response_requires_all_endpoints():
    df = _endpoint_frame().drop(columns=["E8_fragmentation_index"])
    with pytest.raises(ValueError, match="missing endpoint"):
        dose_response(df)


def test_dose_response_rejects_duplicate_intervals():
    df = _endpoint_frame()
    df.loc[1, "time_interval_id"] = df.loc[0, "time_interval_id"]
    with pytest.raises(ValueError, match="duplicate"):
        dose_response(df)
