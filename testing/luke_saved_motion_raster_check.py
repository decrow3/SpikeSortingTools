"""Exploratory check against saved, uncorrected, localized detection profiles.

Select native-rigid excursions >100 um from its median and nearby quiet batches
within 10 s. This is a diagnostic of suspicious estimates, not an efficacy gate.
No sorting or raw-voltage inference is performed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from testing.luke_saved_motion_field_audit import sha256


def run(group,motion,out):
    out.mkdir(parents=True,exist_ok=False)
    ops_path=group/'arms/rescue_12_9_native_rigid/sort/sorter_output/ops.npy'
    ops=np.load(ops_path,allow_pickle=True).item()
    d=np.asarray(ops['dshift']).reshape(-1)
    centered=d-np.median(d)
    fs=ops['fs'];batch=ops['batch_size'];n=len(d)
    # Detection samples are recording-relative, as in SI's peak dtype.
    paths=[ops_path,motion/'peaks.npy',motion/'peak_locations.npy']
    peaks=np.load(paths[1],mmap_mode='r');loc=np.load(paths[2],mmap_mode='r')
    assert len(peaks)==len(loc)
    edges=np.arange(1400,2385,5);centers=(edges[:-1]+edges[1:])/2
    histogram=np.zeros((n,len(centers)),dtype=np.int64)
    for start in range(0,len(peaks),1000000):
        p=peaks[start:start+1000000];y=loc['y'][start:start+1000000]
        assert np.all(p['segment_index']==0)
        b=p['sample_index']//batch
        k=np.searchsorted(edges,y,side='right')-1
        keep=(b>=0)&(b<n)&(k>=0)&(k<len(centers))
        np.add.at(histogram,(b[keep],k[keep]),1)
    interior=(centers>=1600)&(centers<=2180)
    quiet=np.flatnonzero(abs(centered)<=25)
    rows=[]
    for h in np.flatnonzero(abs(centered)>100):
        choices=quiet[abs(quiet-h)<=5]
        if not len(choices):continue
        q=min(choices,key=lambda q:(abs(q-h),q))
        a=histogram[h].astype(float);v=histogram[q,interior].astype(float)
        if a.sum()<100 or v.sum()<100:continue
        # corrected high(y) = observed high(y - (d_high-d_quiet)).
        native=np.interp(centers[interior]-(d[h]-d[q]),centers,a,left=np.nan,right=np.nan)
        raw=a[interior];valid=np.isfinite(native)
        if valid.sum()<80:continue
        def corr(a,b):
            return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else np.nan
        rows.append(dict(high_batch=int(h),quiet_batch=int(q),high_start_s=h*batch/fs,
                         quiet_start_s=q*batch/fs,native_relative_dshift_um=float(d[h]-d[q]),
                         raw_profile_correlation=corr(raw[valid],v[valid]),
                         native_profile_correlation=corr(native[valid],v[valid]),
                         valid_depth_bins=int(valid.sum()),high_strip_events=int(a.sum()),
                         quiet_interior_events=int(v.sum())))
    df=pd.DataFrame(rows).dropna()
    df.to_csv(out/'suspicious_batch_pairs.csv',index=False)
    np.savez_compressed(out/'detection_histogram.npz',counts=histogram,depth_um=centers,
                        time_s=(np.arange(n)+.5)*batch/fs)
    result=dict(pairs=len(df),unique_quiet_batches=int(df.quiet_batch.nunique()),
                median_raw_profile_correlation=float(df.raw_profile_correlation.median()),
                median_native_profile_correlation=float(df.native_profile_correlation.median()),
                fraction_native_lower=float((df.native_profile_correlation<df.raw_profile_correlation).mean()),
                source_sha256={str(p):sha256(p) for p in paths},
                limitations=['Selected on extreme native estimates; not a population efficacy endpoint.',
                             'Independent saved detections share preprocessing and localization with the independent fields.',
                             'Quiet batches can be reused; no p-values or independent-trial interpretation.',
                             'Aggregate spatial profiles mix neurons and firing changes; no validated identity or raw waveform test.',
                             'Depth range 1600–2180 um, only common interpolation support, 5 um bins.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['group-root','motion-root','output-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.group_root,a.motion_root,a.output_root)
