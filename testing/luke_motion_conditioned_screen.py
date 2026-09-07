"""Exploratory fixed-time screen of existing off/rigid pairs; never sorts.

Selection is independent of sorted outcomes: 120 s bins, upper/lower motion
quartiles, nearest unused quiet bin within 600 s, chronological high-bin order.
Three previously accepted full-probe estimators are sensitivity arms. Both
within-bin excursion and displacement from session median are examined.
This is an exploratory screen, not a new promotion gate or identity validation.
"""
from pathlib import Path
import hashlib
import io
import json

import numpy as np
import pandas as pd

from testing.luke_amplitude_dropout_audit import read_curated_arrays, curated_arrays_from_raw

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = Path('/mnt/NPX/Luke/20250804/shared_analysis/luke_group1_handoff_v1')
MOTION = Path('/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion')
OUT = ROOT / 'testing/outputs/luke_motion_conditioned_screen_v1'


def select_pairs(dose, radius=5):
    low, high = np.quantile(dose, [.25, .75])
    if high <= low:
        return []
    available = set(np.flatnonzero(dose <= low).tolist())
    pairs = []
    for h in np.flatnonzero(dose >= high):
        choices = [q for q in available if abs(q - h) <= radius]
        if choices:
            q = min(choices, key=lambda q: (abs(q - h), q))
            pairs.append((int(h), q))
            available.remove(q)
    return pairs


def ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def data(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw

    manifest = json.loads(data(HANDOFF / 'recording/rescue_recording_manifest.json'))
    binary = json.loads(data(HANDOFF / 'recording/binary.json'))
    origin = float(binary['kwargs']['t_starts'][0])
    fs = float(manifest['sampling_frequency_hz'])
    duration = manifest['num_samples'] / fs
    nbin = int(duration // 120)
    path = ROOT / 'testing/outputs/luke_native_rigid_comparison_v1/matched_interior_unit_metrics.csv'
    families = pd.read_csv(io.BytesIO(data(path)))
    arrays = {}
    for arm, name in [('off', 'motion_off'), ('rigid', 'native_rigid')]:
        directory = HANDOFF / f'arms/rescue_12_9_{name}/cur/cur_output'
        raw, digest = read_curated_arrays(directory)
        hashes.update({str(directory / k): v for k, v in digest.items()})
        arrays[arm] = curated_arrays_from_raw(arm, raw)
    # Per-family bins include true zero-event intervals; amplitude is missing
    # for those intervals rather than filled with an artificial zero.
    metrics = {}
    for row in families.itertuples():
        key = int(row.baseline_cluster)
        for arm, cid in [('off', row.baseline_cluster), ('rigid', row.candidate_cluster)]:
            a = arrays[arm]
            keep = a.clusters == cid
            ts, amp = a.times[keep] / fs, a.amplitudes[keep]
            assert len(ts) == int(getattr(row, 'baseline_events' if arm == 'off' else 'candidate_events'))
            bins = np.floor(ts / 120).astype(int)
            counts = np.bincount(bins[bins < nbin], minlength=nbin)
            med = np.full(nbin, np.nan)
            q20 = med.copy()
            for b in range(nbin):
                values = amp[bins == b]
                if len(values):
                    q20[b], med[b] = np.quantile(values, [.2, .5])
            metrics[key, arm] = (counts, med, q20)
    selections, results = [], []
    for estimator in ['dredge-motion', 'decentralized-motion', 'ks-motion']:
        t = np.load(io.BytesIO(data(MOTION / estimator / 'time_bins.npy'))).reshape(-1) - origin
        m = np.load(io.BytesIO(data(MOTION / estimator / 'motion.npy')))
        if m.shape[0] != len(t):
            m = m.T
        assert m.shape[0] == len(t) and np.isfinite(m).all()
        assert np.all(np.diff(t) > 0) and 0 <= t[0] <= 2 and t[-1] >= nbin * 120 - 2
        rigid = np.median(m, axis=1)
        center = np.median(rigid)
        doses = {'excursion': [], 'absolute_displacement': []}
        for b in range(nbin):
            v = rigid[(t >= b * 120) & (t < (b + 1) * 120)]
            assert len(v) >= 50
            doses['excursion'].append(float(np.diff(np.quantile(v, [.05, .95]))[0]))
            doses['absolute_displacement'].append(float(np.median(np.abs(v - center))))
        for exposure, dose in doses.items():
            chosen = select_pairs(np.asarray(dose))
            hi = [h for h, q in chosen]
            lo = [q for h, q in chosen]
            for h, q in chosen:
                selections.append(dict(estimator=estimator, exposure=exposure,
                                       high_start_s=h*120, quiet_start_s=q*120,
                                       high_dose_um=dose[h], quiet_dose_um=dose[q]))
            for row in families.itertuples():
                key = int(row.baseline_cluster)
                result = dict(estimator=estimator, exposure=exposure,
                              baseline_cluster=key, candidate_cluster=int(row.candidate_cluster),
                              time_pairs=len(chosen))
                for arm in ['off', 'rigid']:
                    counts, med, q20 = metrics[key, arm]
                    for state, indices in [('high', hi), ('quiet', lo)]:
                        result[f'{arm}_{state}_events'] = int(counts[indices].sum())
                        result[f'{arm}_{state}_zero_bins'] = int(np.sum(counts[indices] == 0))
                    result[f'{arm}_high_quiet_rate_ratio'] = ratio(counts[hi].sum(), counts[lo].sum())
                    for label, value in [('amplitude_median', med), ('amplitude_q20', q20)]:
                        valid = [(h,q) for h,q in chosen if np.isfinite(value[h]) and np.isfinite(value[q])]
                        changes = [ratio(value[h], value[q]) for h,q in valid]
                        result[f'{arm}_{label}_paired_bins'] = len(valid)
                        result[f'{arm}_{label}_high_quiet_ratio'] = float(np.nanmedian(changes)) if changes else np.nan
                result['rate_ratio_of_ratios'] = ratio(result['rigid_high_quiet_rate_ratio'], result['off_high_quiet_rate_ratio'])
                results.append(result)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT / 'family_metrics.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT / 'selected_time_pairs.csv', index=False)
    summaries = []
    for (estimator, exposure), g in frame.groupby(['estimator','exposure']):
        valid = g.rate_ratio_of_ratios.dropna()
        summaries.append(dict(estimator=estimator, exposure=exposure,
            time_pairs=int(g.time_pairs.iloc[0]), families=len(g), valid_rate_families=len(valid),
            median_rate_ratio_of_ratios=float(valid.median()),
            fraction_rate_ratio_above_one=float((valid > 1).mean()),
            median_off_high_quiet_rate_ratio=float(g.off_high_quiet_rate_ratio.median()),
            median_rigid_high_quiet_rate_ratio=float(g.rigid_high_quiet_rate_ratio.median()),
            median_off_amplitude_high_quiet_ratio=float(g.off_amplitude_median_high_quiet_ratio.median()),
            median_rigid_amplitude_high_quiet_ratio=float(g.rigid_amplitude_median_high_quiet_ratio.median())))
    result = dict(schema_version='motion-conditioned-screen-v1', clock_origin_s=origin,
                  bin_s=120, maximum_pair_distance_s=600, full_bins=nbin,
                  source_sha256=hashes, summaries=summaries,
                  limitations=['Exploratory survivor cohort of 40 primary interior pairs; unmatched families are excluded.',
                               'Native amplitudes are not raw microvolts; amplitude ratios condition on nonempty bins.',
                               'Time-varying firing state remains a confound; selected bins are reused across estimators.',
                               'No inferential p-values or promotion decisions; no proof of biological identity.'])
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    run()
