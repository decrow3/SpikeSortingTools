"""Prepared, conditional cached-event diagnostic; never reads source voltage."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from spikeinterface.core import NumpyRecording
from threadpoolctl import threadpool_limits
import torch

from testing.luke_epoch_corroboration import ROOT, BASE
from testing.luke_dredge_bounded import estimate_bounded

SOURCE = ROOT / 'testing/outputs/luke_3sigma_lowpass_screen_v2'
OUT = ROOT / 'testing/outputs/luke_lowpass_common_events_v2'
ARMS = [('Bamp_Bloc', 'broad', 'broad'), ('Lamp_Bloc', 'lowpass', 'broad'),
        ('Bamp_Lloc', 'broad', 'lowpass'), ('Lamp_Lloc', 'lowpass', 'lowpass')]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def common_indices(broad, lowpass, nchannels):
    """Reject duplicate identities rather than silently choosing one localization."""
    keys = []
    for peaks in (broad, lowpass):
        assert np.all((peaks['channel_index'] >= 0) & (peaks['channel_index'] < nchannels))
        assert np.all(peaks['sample_index'] >= 0)
        assert np.all(peaks['segment_index'] == 0)
        key = peaks['sample_index'].astype(np.int64) * nchannels + peaks['channel_index']
        assert len(np.unique(key)) == len(key), 'Duplicate frame/channel source keys'
        keys.append(key)
    common, ib, il = np.intersect1d(*keys, assume_unique=True, return_indices=True)
    assert len(common), 'No exact common detections'
    return common, ib, il


def main():
    started = time.monotonic()
    source_files = [SOURCE / 'settings.json', SOURCE / 'summary.json',
                    BASE / 'recording/rescue_recording_manifest.json']
    for name in ('broad_3sigma', 'lowpass_adjusted'):
        source_files += [SOURCE / f'{name}_peaks.npy', SOURCE / f'{name}_locations.npy',
                         SOURCE / 'fields' / f'{name}.npz']
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Source run is not ready: ' + ', '.join(missing))
    assert json.loads((SOURCE / 'summary.json').read_text())['status'] == 'complete'
    source_settings = json.loads((SOURCE / 'settings.json').read_text())
    assert source_settings['interval_s'] == [4160, 4260]
    cfg = source_settings['estimator'].copy()
    rec = json.loads((BASE / 'recording/rescue_recording_manifest.json').read_text())
    fs = rec['sampling_frequency_hz']
    geo = np.asarray(rec['channel_locations_um'])
    frames = round(100 * fs)
    peaks, locations = {}, {}
    for label, name in [('broad', 'broad_3sigma'), ('lowpass', 'lowpass_adjusted')]:
        peaks[label] = np.load(SOURCE / f'{name}_peaks.npy')
        locations[label] = np.load(SOURCE / f'{name}_locations.npy')
        assert len(peaks[label]) == len(locations[label])
        assert np.all(peaks[label]['sample_index'] < frames)
        assert np.isfinite(peaks[label]['amplitude']).all()
        for field in locations[label].dtype.names:
            assert np.isfinite(locations[label][field]).all()
    assert peaks['broad'].dtype == peaks['lowpass'].dtype
    assert locations['broad'].dtype == locations['lowpass'].dtype
    common, ib, il = common_indices(peaks['broad'], peaks['lowpass'], len(geo))
    indices = {'broad': ib, 'lowpass': il}
    hashes = {str(path): digest(path) for path in source_files}
    OUT.mkdir(exist_ok=False)
    (OUT / 'fields').mkdir()
    settings = dict(
        interval_s=[4160, 4260], estimator=cfg, source_sha256=hashes,
        code_sha256={str(path): digest(path) for path in [Path(__file__).resolve(),
            ROOT / 'testing/luke_dredge_bounded.py', ROOT / 'testing/luke_screen_sweep.py']},
        factorial_arms=[dict(name=n, amplitude=a, localization=l) for n, a, l in ARMS],
        contextual_controls=['full_broad_3sigma', 'full_lowpass_adjusted'],
        matching='Exact sample_index/channel_index intersection, single segment; no timing tolerance',
        interpretation='Four arms hold exact frame/channel membership fixed while crossing measured amplitude and localization. Low-pass timing/channel shifts exclude otherwise related events, so the intersection is selection-biased and does not exhaust membership effects. Full-population controls are contextual, not factorial arms. Lighthouse disagreement is descriptive, not physical accuracy.',
        retained_fraction_broad=len(common) / len(peaks['broad']),
        retained_fraction_lowpass=len(common) / len(peaks['lowpass']),
        retained_fraction_denominator='Full broadband 3sigma population in manifest; source-specific fractions also saved',
        max_threads=1, thread_environment={key: os.environ.get(key) for key in
            ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']},
        resume='No automatic or within-field resume; existing output refused. Investigate failed evidence before recovery.',
        scope='Cached arrays only; no detection, localization, source voltage access, tuning, or sort')
    (OUT / 'settings.json').write_text(json.dumps(settings, indent=2))
    np.savez_compressed(OUT / 'common_event_ids.npz', key=common,
        sample_index=peaks['broad']['sample_index'][ib], channel_index=peaks['broad']['channel_index'][ib],
        absolute_frame=peaks['broad']['sample_index'][ib] + round(4160 * fs),
        broad_source_index=ib, lowpass_source_index=il)
    record = NumpyRecording(np.broadcast_to(np.zeros((1, len(geo)), dtype='float32'),
                                            (frames, len(geo))), fs)
    record.set_channel_locations(geo)
    manifest, timings = [], [dict(stage='load_intersect_hash', elapsed_s=time.monotonic() - started)]
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for name, amp_source, loc_source in ARMS:
            tick = time.monotonic()
            p = peaks['broad'][ib].copy()
            p['amplitude'] = peaks[amp_source]['amplitude'][indices[amp_source]]
            y = locations[loc_source][indices[loc_source]].copy()
            assert np.array_equal(p['sample_index'] * len(geo) + p['channel_index'], common)
            np.save(OUT / f'{name}_peaks.npy', p)
            np.save(OUT / f'{name}_locations.npy', y)
            motion, extra = estimate_bounded(record, p, y, cfg)
            np.savez_compressed(OUT / 'fields' / f'{name}.npz',
                time_s=motion.temporal_bins_s[0] + 4160, depth_um=motion.spatial_bins_um,
                displacement_um=motion.displacement[0], D=extra['D'], C=extra['C'], U=extra['U'])
            manifest.append(dict(name=name, role='factorial', peaks=len(p),
                retained_fraction=len(p) / len(peaks['broad']), amplitude_source=amp_source,
                localization_source=loc_source))
            timings.append(dict(stage=name, elapsed_s=time.monotonic() - tick))
            pd.DataFrame(timings).to_csv(OUT / 'timings.csv', index=False)
            print(name, 'complete', timings[-1], flush=True)
    for label, name in [('broad', 'broad_3sigma'), ('lowpass', 'lowpass_adjusted')]:
        control = 'full_' + name
        shutil.copyfile(SOURCE / 'fields' / f'{name}.npz', OUT / 'fields' / f'{control}.npz')
        manifest.append(dict(name=control, role='context_only', peaks=len(peaks[label]),
            retained_fraction=len(peaks[label]) / len(peaks['broad']),
            amplitude_source=label, localization_source=label))
    pd.DataFrame(manifest).to_csv(OUT / 'manifest.csv', index=False)
    from testing import luke_screen_sweep as scoring
    scoring.OUT = OUT
    scoring.analyze()
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.axes[2].set_title('Fraction of full broadband detections')
    fig.suptitle('Exact common events: crossed amplitude/localization inputs\nFull-population controls are contextual; provisional lighthouse differences are not ground-truth errors')
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'01_sweep_summary.{ext}', dpi=140)
    plt.close(fig)
    assert hashes == {str(path): digest(path) for path in source_files}, 'Source changed during diagnostic'
    timings.append(dict(stage='total', elapsed_s=time.monotonic() - started))
    pd.DataFrame(timings).to_csv(OUT / 'timings.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(dict(status='complete', common_events=len(common),
        broad_events=len(peaks['broad']), lowpass_events=len(peaks['lowpass']),
        source_hashes_unchanged=True, elapsed_s=time.monotonic() - started), indent=2))


if __name__ == '__main__':
    main()
