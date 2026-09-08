"""Validate the existing full-probe reference without sorting or changing inputs."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from pipeline.config import fingerprint
from pipeline.downstream import build_sort_identity, _atomic_json, _sha256
from testing.ladder_sorter import RESCUE, _json_safe

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0')
OUT = ROOT / 'testing/outputs/luke_full_reference_reuse_audit_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(path):
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    def progress(stage, **extra):
        state = dict(stage=stage, updated_unix=time.time(), **extra)
        _atomic_json(OUT/'progress.json', state)
        print(json.dumps(state), flush=True)

    progress('validate sort identity and settings')
    recording = read(BASE/'recording/rescue_recording_manifest.json')
    sort = read(BASE/'kilosort4/rescue_sort_manifest.json')
    assert sort['complete'] and recording['complete']
    assert (recording['num_samples'], recording['num_channels']) == (314204894,384)
    assert sort['sorter_params'] == _json_safe(RESCUE.params())
    assert sort['recording_request_digest'] == recording['request_digest']
    identity = build_sort_identity(BASE/'kilosort4')
    pinned = read(BASE/'rescue_sort_identity.json')
    assert identity['identity_digest'] == pinned['identity_digest']
    ops = np.load(BASE/'kilosort4/sorter_output/ops.npy', allow_pickle=True).item()
    assert ops['nblocks'] == 0 and ops['dshift'] is None
    assert ops['n_chan_bin'] == 384 and ops['Th_universal'] == 12 and ops['Th_learned'] == 9
    assert ops['fs'] == recording['sampling_frequency_hz']
    progress('validate downstream receipts and files')
    receipts = {}
    for folder, stem in [('cur','curation'),('qc','qc')]:
        request = read(BASE/folder/f'{stem}_request.json')
        receipt = read(BASE/folder/f'{stem}_receipt.json')
        payload = {k:request[k] for k in ['schema_version','stage','sort_identity_digest','settings']}
        assert fingerprint(payload) == request['request_digest'] == receipt['request_digest']
        assert receipt['complete']
        assert request['sort_identity_digest'] == receipt['sort_identity_digest'] == identity['identity_digest']
        for name in receipt['required_files']:
            path = Path(name)
            assert path.is_file() and path.stat().st_size > 0
            hashes[str(path)] = _sha256(path)
        receipts[stem] = {'request':request,'receipt':receipt}
    c = receipts['curation']['request']['settings']
    assert c['strategy']=='run_cur_final_cosine' and c['cosine_threshold']==.9 and c['ccg_threshold']==.5
    assert c['automatic_artifact_pair_merging'] is False
    q = receipts['qc']['request']['settings']
    assert q['waveform_seed']==0 and q['waveforms_per_unit']==512 and q['waveform_samples']==82
    progress('validate every retained event time and amplitude')
    cur = BASE/'cur/cur_output'
    arrays = {name:np.load(cur/name,mmap_mode='r') for name in ['spike_times.npy','spike_clusters.npy','full_st.npy','kept_spikes.npy']}
    ts, cl, full, kept = (arrays[n] for n in ['spike_times.npy','spike_clusters.npy','full_st.npy','kept_spikes.npy'])
    ts, cl = ts.reshape(-1), cl.reshape(-1)
    indices = np.flatnonzero(kept) if kept.dtype.kind=='b' else np.asarray(kept)
    assert len(ts)==len(cl)==len(indices)
    assert indices.dtype.kind in 'iu' and indices.min()>=0 and indices.max()<len(full)
    prior = -1
    for start in range(0,len(ts),1000000):
        end = min(start+1000000,len(ts))
        rows = full[indices[start:end]]
        t = ts[start:end]
        assert np.array_equal(t,rows[:,0]) and np.isfinite(rows[:,2]).all()
        assert t[0]>=prior and np.all(np.diff(t)>=0)
        prior = t[-1]
    assert ts.min()>=0 and ts.max()<recording['num_samples']
    assert len(ts)==receipts['curation']['receipt']['summary']['spike_count']
    assert len(np.unique(cl))==receipts['curation']['receipt']['summary']['unit_count']
    for name in arrays:
        hashes[str(cur/name)] = _sha256(cur/name)
    preliminary = dict(status='metadata_and_retained_lineage_verified_content_hash_pending',
        sort_identity=identity, curated_spikes=len(ts),curated_units=len(np.unique(cl)),
        effective_motion_off=True, downstream=receipts, source_sha256=hashes)
    _atomic_json(OUT/'preliminary.json',preliminary)
    del arrays,indices,ts,cl,full,kept
    progress('full recording content hash',bytes_read=0,total_bytes=recording['expected_binary_bytes'])
    aggregate=hashlib.sha256(); files=[]; total=0; last_report=0
    paths=sorted(list((BASE/'recording').glob('*.raw'))+list((BASE/'recording').glob('*.bin')),key=lambda p:p.name)
    assert paths
    for path in paths:
        before=path.stat(); digest=hashlib.sha256()
        with path.open('rb') as handle:
            while block:=handle.read(8*1024*1024):
                digest.update(block);total+=len(block)
                if total-last_report>=4*1024**3:
                    progress('full recording content hash',bytes_read=total,total_bytes=recording['expected_binary_bytes'])
                    last_report=total
        after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        h=digest.hexdigest();files.append(dict(name=path.name,size_bytes=after.st_size,sha256=h))
        aggregate.update(f'{path.name}\0{after.st_size}\0{h}\n'.encode())
    assert total==recording['expected_binary_bytes']
    assert files==recording['recording_binary_files']
    assert aggregate.hexdigest()==recording['recording_content_sha256']
    result=dict(preliminary,status='reference_reusable',recording_request_digest=recording['request_digest'],
        recording_content_sha256=aggregate.hexdigest(),verified_recording_bytes=total,
        limitations=['Cached QC lineage and integrity checked; scientific completeness is not established.',
                     'No raw-waveform diagnostic panel or cross-arm biological family validation completed by this audit.'])
    _atomic_json(OUT/'summary.json',result)
    progress('complete',status='reference_reusable')


if __name__=='__main__':
    main()
