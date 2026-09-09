"""Pinned full-probe registration ONLY; never calls the full sorting pipeline.

The drift estimator's own universal-template extraction and event detection are
required. Downstream learned-template extraction, clustering and export are
blocked at runtime. No retry or checkpoint/resume is implemented.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import copy
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import numpy as np
from testing.managed_job import _atomic_json

ROOT=Path(__file__).resolve().parents[1]
REF=Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0')
COORD=Path('/mnt/NPX/Luke/20250804/shared_analysis/luke_next_stage_coordination_20260907_v1')
HOLD=ROOT/'configs/luke_full_session_rigid.HOLD.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return safe(v.item())
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not np.isfinite(v):return str(v)
    return v


def forbidden(*args,**kwargs):
    raise RuntimeError('Registration-only boundary: downstream sorting is prohibited')


def registration_only(api, settings, probe, device, progress, seed):
    """Only allowed stage sequence; close drift's corrected reader immediately."""
    with ExitStack() as guards:
        for name in ['run_kilosort','detect_spikes','cluster_spikes','save_sorting']:
            guards.enter_context(patch.object(api,name,forbidden))
        guards.enter_context(patch.object(api.template_matching,'extract',forbidden))
        guards.enter_context(patch.object(api.clustering_qr,'run',forbidden))
        tic=time.time()
        progress('initialization')
        ops=api.initialize_ops(settings,probe,'int16',True,False,device,False)
        progress('preprocessing')
        ops=api.compute_preprocessing(ops,device,tic0=tic)
        seed()
        progress('drift_estimation_including_own_detections')
        ops,bfile,st=api.compute_drift_correction(ops,device,tic0=tic,clear_cache=True)
        try:
            progress('registration_returned_closing_binary')
        finally:
            bfile.close()
        return ops,st


def prepare(out):
    from kilosort.parameters import DEFAULT_SETTINGS
    from pipeline.sorting import build_kilosort4_params,_json_safe_params
    from pipeline.runtime import validate_production_environment
    environment=validate_production_environment(require_cuda=False)
    sources={name:REF/rel for name,rel in dict(recording='recording/rescue_recording_manifest.json',
        binary_metadata='recording/binary.json',reference_ops='kilosort4/sorter_output/ops.npy',
        sort_manifest='kilosort4/rescue_sort_manifest.json').items()}
    recording=json.loads(sources['recording'].read_text())
    manifest=json.loads(sources['sort_manifest'].read_text())
    binary_meta=json.loads(sources['binary_metadata'].read_text())
    refops=np.load(sources['reference_ops'],allow_pickle=True).item()
    expected=(314204894,384,29999.835983263598)
    assert tuple(recording[k] for k in ['num_samples','num_channels','sampling_frequency_hz'])==expected
    assert recording['request_digest']==manifest['recording_request_digest']
    assert manifest['complete'] and refops['nblocks']==0 and refops['dshift'] is None
    assert manifest['sorter_params']==_json_safe_params(build_kilosort4_params())
    assert refops['do_CAR'] and not refops['invert_sign']
    probe=refops['probe']
    assert np.array_equal(probe['chanMap'],np.arange(384))
    assert len(probe['xc'])==len(probe['yc'])==384
    assert (probe['yc'].min(),probe['yc'].max())==(0,3820)
    assert len(set(zip(probe['xc'],probe['yc'])))==384
    binary=REF/'recording/traces_cached_seg0.raw'
    stat=binary.stat()
    assert stat.st_size==expected[0]*expected[1]*2
    settings={k:copy.deepcopy(refops['settings'][k]) for k in DEFAULT_SETTINGS}
    settings.update(nblocks=1,filename=str(binary),data_dir=str(binary.parent))
    assert settings['Th_universal']==12 and settings['Th_learned']==9
    assert settings['tmin']==0 and np.isinf(settings['tmax'])
    package=Path(importlib.import_module('kilosort').__file__).parent
    implementations=[package/n for n in ['run_kilosort.py','spikedetect.py','datashift.py','preprocessing.py','io.py']]
    implementations += [Path(__file__),ROOT/'pipeline/sorting.py',ROOT/'testing/managed_job.py']
    snapshot=out/'inputs';snapshot.mkdir(exist_ok=False)
    for name,p in sources.items():
        (snapshot/(name+p.suffix)).write_bytes(p.read_bytes())
    np.savez(snapshot/'probe.npz',**probe)
    np.save(snapshot/'settings.npy',settings,allow_pickle=True)
    plan=dict(scope='Full-probe rigid registration only; no downstream sort',
        recording_request_digest=recording['request_digest'],
        recording_content_sha256=recording['recording_content_sha256'],
        source_binary=str(binary),source_stat=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),
        num_samples=expected[0],num_channels=expected[1],fs_hz=expected[2],
        clock_origin_s=binary_meta['kwargs']['t_starts'][0],
        settings=safe(settings),reference_effective_nblocks=0,diagnostic_effective_nblocks=1,
        environment=environment,source_sha256={str(p):sha(p) for p in sources.values()},
        implementation_sha256={str(p):sha(p) for p in implementations},
        hold_sha256=sha(HOLD),assignment_sha256=sha(COORD/'README.md'),
        checkpoint_policy='No within-estimation checkpoint. Interruption requires full preprocessing and drift-estimation restart after investigation; no automatic retries.',
        universal_template_note='templates_from_data stays true: drift detection computes its own universal templates. Post-drift learned-template extraction is prohibited.')
    _atomic_json(out/'plan.json',plan)
    return plan


def execute(out,qualification):
    import torch
    from pipeline.runtime import validate_production_environment
    plan=json.loads((out/'plan.json').read_text())
    if (out/'registration_started.json').exists():raise RuntimeError('Refusing retry: preserve and investigate prior attempt')
    assert sha(HOLD)==plan['hold_sha256'] and json.loads(HOLD.read_text())['state']=='hold'
    for p,h in plan['implementation_sha256'].items():assert sha(p)==h,p
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    binary=Path(plan['source_binary']);stat=binary.stat()
    assert dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)==plan['source_stat']
    q=json.loads(qualification.read_text())
    # A small, explicit attestation is prepared only after parent summary review.
    assert q['reference_qualified'] is True and q['bulk_reads_released'] is True
    assert q['recording_content_sha256']==plan['recording_content_sha256']
    assert q['recording_request_digest']==plan['recording_request_digest']
    assert sha(Path(q['parent_summary_path']))==q['parent_summary_sha256']
    environment=validate_production_environment(require_cuda=True)
    assert torch.cuda.device_count()==1
    manager=json.loads((out/'manager-proof.json').read_text())
    assert manager['state']=='complete' and manager['returncode']==0
    assert (out/'manager-proof-disconnected.json').is_file()
    settings=np.load(out/'inputs/settings.npy',allow_pickle=True).item()
    probe=dict(np.load(out/'inputs/probe.npz'))
    probe['n_chan']=int(probe['n_chan'])
    api=importlib.import_module('kilosort.run_kilosort')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    t0=time.time();torch.cuda.reset_peak_memory_stats()
    _atomic_json(out/'registration_started.json',dict(pid=os.getpid(),unix=t0,environment=environment,qualification=q))
    def progress(stage):
        _atomic_json(out/'progress.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time()))
        print(stage,flush=True)
    def seed():
        np.random.seed(1);torch.cuda.manual_seed_all(1);torch.random.manual_seed(1)
    ops,st=registration_only(api,settings,probe,torch.device('cuda'),progress,seed)
    progress('saving_registration_only_outputs')
    assert st.ndim==2 and st.shape[1]==6 and np.isfinite(st).all()
    d=np.asarray(ops['dshift']);y=np.asarray(ops['yblk']);n=int(ops['Nbatches'])
    assert d.shape==(n,1) and n==5237 and np.isfinite(d).all() and y.shape==(1,)
    assert ops['nblocks']==1 and len(ops['yc'])==384
    np.save(out/'native_pre_registration_detections.npy',st)
    api.io.save_ops(ops,results_dir=out)
    centers=(np.arange(n)+.5)*plan['settings']['batch_size']/plan['fs_hz']
    np.savez_compressed(out/'native_rigid_field.npz',time_s=centers,depth_um=y,
        native_dshift_um=d,physical_displacement_um=-d,
        channel_positions_um=np.c_[probe['xc'],probe['yc']])
    # Save compact support for all detections, not a selected scatter subset.
    edges=np.arange(-5,3835,5);hist=np.zeros((n,len(edges)-1),dtype=np.int64)
    batches=st[:,4].astype(int);depthbin=np.searchsorted(edges,st[:,1],side='right')-1
    keep=(depthbin>=0)&(depthbin<hist.shape[1])&(batches>=0)&(batches<n)
    np.add.at(hist,(batches[keep],depthbin[keep]),1)
    np.savez_compressed(out/'native_detection_raster.npz',counts=hist,depth_um=(edges[:-1]+edges[1:])/2,time_s=centers)
    assert sha(HOLD)==plan['hold_sha256']
    assert dict(size=binary.stat().st_size,mtime_ns=binary.stat().st_mtime_ns)==plan['source_stat']
    files=['native_pre_registration_detections.npy','ops.npy','native_rigid_field.npz','native_detection_raster.npz']
    summary=dict(status='registration_complete_audit_pending',elapsed_s=time.time()-t0,
        detections=len(st),detection_columns=['time_seconds','depth_um','amplitude','template_index','batch_index','universal_spatial_template_index'],
        geometry_channels=384,batches=n,physical_sign='physical_displacement = -dshift',
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        source_plan_sha256=sha(out/'plan.json'),hold_sha256=sha(HOLD),full_sort_launched=False,
        file_sha256={f:sha(out/f) for f in files})
    _atomic_json(out/'registration_summary.json',summary)
    progress('registration_complete_audit_pending')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','execute'])
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--qualification',type=Path)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
    if a.action=='prepare':prepare(a.output_root)
    else:execute(a.output_root,a.qualification)

if __name__=='__main__':main()
