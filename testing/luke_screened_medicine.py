"""Chunked screened MEDiCINe benchmark; launch through the independent job manager."""
import argparse,json,os,subprocess,time,resource
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from testing.luke_screen_sweep import features,mask
from testing.luke_ap_methods_sweep_v1 import ROOT,PYTHON,MEDPY,sha,valid,seal
from testing.luke_epoch_corroboration import BASE
SRC=ROOT/'testing/outputs'
VARIANTS={'5sigma_relaxed':(5,dict(snr=6,center=.5,neighbor=.6)), '6sigma_full':(6,{})}

def save(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));os.replace(temp,path)

def screened_features(clean,post,peaks,geo,noise,fs,batch=2048):
    """Bound waveform temporary memory without altering per-event feature definitions."""
    parts=[features(clean,post,peaks[i:i+batch],geo,noise,fs) for i in range(0,len(peaks),batch)]
    if not parts:return features(clean,post,peaks,geo,noise,fs)
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}

def run_fit(source,dest,steps):
    if valid(dest):return
    dest.mkdir();cmd=[str(PYTHON),'-m','testing.managed_job','--receipt',str(dest/'receipt.json'),'--cwd',str(ROOT),'--',str(MEDPY),'-m','testing.luke_screened_medicine_fit','--input',str(source),'--output',str(dest),'--steps',str(steps)]
    save(dest/'command.json',cmd);env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (dest/'stdout.log').open('w') as out,(dest/'stderr.log').open('w') as err:subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,check=True)
    seal(dest)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--start',type=int,default=930);ap.add_argument('--stop',type=int,default=1230);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert args.start==930, "This Luke benchmark anchors the frozen930–940s seed interval"
    assert args.stop>args.start and (args.stop-args.start)%20==0 and args.start>=1
    out=args.output.resolve();out.mkdir(exist_ok=True);begin=time.monotonic()
    mp=BASE/'recording/rescue_recording_manifest.json';m=json.loads(mp.read_text());fs=m['sampling_frequency_hz'];geo=np.array(m['channel_locations_um']);raw=BASE/'recording/traces_cached_seg0.raw';stat=raw.stat()
    assert round(args.stop*fs)+round(.05*fs)<=stat.st_size//768
    modelpath=SRC/'luke_common_event_screen_v1/shared_response_model.npz';noisepath=SRC/'luke_peak_threshold_screen_v1/noise_uv.npy';model=np.load(modelpath);noise=np.load(noisepath)
    sources=[Path(__file__),ROOT/'testing/luke_screened_medicine_fit.py',ROOT/'testing/luke_screen_sweep.py',ROOT/'testing/luke_ap_methods_sweep_fit_v1.py',ROOT/'testing/luke_screened_medicine_report.py',mp,modelpath,noisepath]
    cfg=dict(start_s=args.start,stop_s=args.stop,sampling_frequency_hz=fs,chunk_s=20,variants=VARIANTS,feature_batch=2048,localization='Screen first, localize union of surviving peaks once; monopolar radius75um,4workers',detection='Separate5/6sigma negative locally exclusive50um; frozen noise; exclude50samples at chunk edges',training_steps=[10000,30000],raw_stat=[stat.st_size,stat.st_mtime_ns],source_sha256={str(p):sha(p) for p in sources},restart='Only sealed completed stages reusable; unsealed stage requires investigation; no optimizer resume')
    if (out/'settings.json').exists():assert json.loads((out/'settings.json').read_text())==json.loads(json.dumps(cfg))
    else:save(out/'settings.json',cfg)
    # Cheap model reproduction before reading additional voltage.
    control=out/'control_input'
    if not valid(control):
        control.mkdir();src=SRC/'luke_early_sigma_screen250_v1/fields/5sigma_relaxed_bound250';assert valid(src)
        for fn in ['peaks.npy','locations.npy']:
            import shutil
            shutil.copy2(src/fn,control/fn)
        save(control/'input.json',dict(start_s=930,stop_s=1030,sampling_frequency_hz=fs));seal(control)
    run_fit(control,out/'control_fit',10000)
    ref=np.load(SRC/'luke_ap_methods_sweep_v1/arms/5sigma_relaxed_bound250__medicine/field.npz');got=np.load(out/'control_fit/field.npz')
    assert np.array_equal(ref['time_s'],got['time_s']) and np.array_equal(ref['depth_um'],got['depth_um'])
    delta=abs(ref['displacement_um']-got['displacement_um']);save(out/'reproduction.json',dict(max_abs_difference_um=float(delta.max()),rms_difference_um=float(np.sqrt(np.mean(delta**2))),tolerance_um=.1))
    assert delta.max()<.1,'Model reproduction differs; investigate before longer fit'
    print('100s model reproduction passed',flush=True)
    chunks=[]
    for start in range(args.start,args.stop,20):
        stage=out/f'chunk_{start}';chunks.append(stage)
        if valid(stage):continue
        stage.mkdir();tick=time.monotonic();cached=SRC/f'luke_early_sigma_screen250_v1/chunk_{start}'
        if cached.exists() and start in [930,950,970,990,1010]:
            assert valid(cached);p=np.load(cached/'union_peaks.npy');loc=np.load(cached/'union_locations.npy');f=np.load(cached/'union_features.npz')
            for name,(sigma,kw) in VARIANTS.items():
                ix=np.load(cached/f'{sigma}sigma_union_indices.npy');ix=ix[mask({k:f[k][ix] for k in f.files},**kw)];np.save(stage/f'{name}_peaks.npy',p[ix]);np.save(stage/f'{name}_locations.npy',loc[ix])
            save(stage/'audit.json',dict(reused_from=str(cached),source_seal_sha256=sha(cached/'complete.json'),seconds=time.monotonic()-tick))
        else:
            n=round(20*fs);pad=round(.05*fs)
            with raw.open('rb') as handle:handle.seek((round(start*fs)-pad)*768);buf=handle.read((n+2*pad)*768)
            assert len(buf)==(n+2*pad)*768
            x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf
            x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');refv=np.median(x,axis=1);clean=np.empty((n,384),np.float32)
            for lo in range(0,n,10000):
                ev=np.arange(lo,min(n,lo+10000));clean[ev]=x[pad+ev]-refv[pad+ev[:,None]+model['lag_samples']]@model['coefficients']
            post=x[pad:pad+n]-refv[pad:pad+n,None];del x,refv
            rec=NumpyRecording(clean,fs);rec.set_channel_locations(geo);det={}
            for name,(sigma,kw) in VARIANTS.items():
                p=detect_peaks(rec,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=float(sigma),noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False);det[name]=p[(p['sample_index']>=50)&(p['sample_index']<n-50)]
            joined=np.concatenate(list(det.values()));keys=lambda p:p['sample_index']*384+p['channel_index'];uk,ix=np.unique(keys(joined),return_index=True);p=joined[ix]
            f=screened_features(clean,post,p,geo,model['residual_noise_uv'],fs);np.save(stage/'detected_union.npy',p);np.savez_compressed(stage/'features.npz',**f)
            selected={};union_keep=np.zeros(len(p),bool)
            for name,(_,kw) in VARIANTS.items():
                ix=np.searchsorted(uk,keys(det[name]));assert np.array_equal(uk[ix],keys(det[name]));keep=mask({k:f[k][ix] for k in f},**kw);selected[name]=ix[keep];union_keep[ix[keep]]=True;np.save(stage/f'{name}_indices.npy',ix[keep])
            survivors=np.flatnonzero(union_keep)
            assert len(survivors)>0,'No screen survivors: preserve chunk and inspect'
            loc=localize_peaks(rec,p[survivors],method='monopolar_triangulation',radius_um=75.,n_jobs=4,mp_context='fork',chunk_duration='1s',progress_bar=False);assert np.isfinite(loc['y']).all()
            for name,ix in selected.items():np.save(stage/f'{name}_peaks.npy',p[ix]);np.save(stage/f'{name}_locations.npy',loc[np.searchsorted(survivors,ix)])
            save(stage/'audit.json',dict(detected_union=len(p),localized=len(survivors),seconds=time.monotonic()-tick,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));del rec,clean,post
        seal(stage);print('SEALED',stage.name,round(time.monotonic()-tick,1),flush=True)
    for name in VARIANTS:
        stage=out/f'input_{name}'
        if not valid(stage):
            stage.mkdir();parts=[];locations=[]
            for start,chunk in zip(range(args.start,args.stop,20),chunks):
                p=np.load(chunk/f'{name}_peaks.npy');p['sample_index']+=round((start-args.start)*fs);parts.append(p);locations.append(np.load(chunk/f'{name}_locations.npy'))
            p=np.concatenate(parts);loc=np.concatenate(locations);assert len(p)==len(loc) and np.all(np.diff(p['sample_index'])>=0)
            np.save(stage/'peaks.npy',p);np.save(stage/'locations.npy',loc);save(stage/'input.json',dict(start_s=args.start,stop_s=args.stop,sampling_frequency_hz=fs,variant=name,peaks=len(p)));seal(stage)
        for steps in ([10000,30000] if name=='5sigma_relaxed' else [10000]):
            print('FIT',name,steps,flush=True);run_fit(stage,out/f'fit_{name}_{steps}',steps)
    assert [raw.stat().st_size,raw.stat().st_mtime_ns]==cfg['raw_stat']
    assert cfg['source_sha256']=={str(p):sha(p) for p in sources}
    from testing.luke_screened_medicine_report import report
    report(out)
    save(out/'summary.json',dict(status='complete',seconds=time.monotonic()-begin,interval_s=[args.start,args.stop]));print('BENCHMARK COMPLETE',flush=True)
if __name__=='__main__':main()
