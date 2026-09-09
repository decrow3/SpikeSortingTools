"""Durable40-combination comparison on exact saved sigma/screening inputs."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';OUT=SRC/'luke_ap_methods_sweep_v1';INPUT=SRC/'luke_early_sigma_screen250_v1';PYTHON=ROOT/'environments/rescue-production/.venv/bin/python';MEDPY=Path('/home/huklab/anaconda3/envs/spikeinterface/bin/python')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,p)
def valid(p):
 seal=p/'complete.json'
 if not seal.exists():
  if p.exists():raise RuntimeError(f'Unsealed arm at{p}; preserve and investigate before restart')
  return False
 for fn,h in json.loads(seal.read_text()).items():assert sha(p/fn)==h,f'Corrupt checkpoint{p/fn}'
 return True
def seal(p):atomic(p/'complete.json',{str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file()})
def main():
 OUT.mkdir(exist_ok=True);(OUT/'arms').mkdir(exist_ok=True);start=time.monotonic();assert json.loads((INPUT/'summary.json').read_text())['status']=='complete';manifest=pd.read_csv(INPUT/'manifest.csv');manifest=manifest[manifest.bound_um==250];assert len(manifest)==20
 baseline='5sigma_none_bound250';order=[baseline]+[s for s in manifest.name if s!=baseline];sources=[Path(__file__),ROOT/'testing/luke_ap_methods_sweep_fit_v1.py',ROOT/'testing/luke_ap_methods_sweep_report_v1.py',ROOT/'testing/luke_dredge_bounded_250_v1.py',ROOT/'docs/luke_ap_motion_methods_20260908.md',INPUT/'settings.json',INPUT/'manifest.csv',SRC/'luke_waveform_only_expansion_v1/candidate_audit.csv',SRC/'luke_waveform_only_expansion_v1/overlay_events.csv',SRC/'luke_population_depth_v2/templates.npz',SRC/'luke_early_medicine_peak_comparison_v8/dredge_settings.json',SRC/'luke_early_medicine_peak_comparison_v8/compensated_dredge_motion.npz',ROOT/'testing/luke_early_medicine_peak_comparison_v6.py']
 for name in order:
  stage=INPUT/'fields'/name;checks=json.loads((stage/'complete.json').read_text())
  for fn in ['peaks.npy','locations.npy']:assert sha(stage/fn)==checks[fn];sources.append(stage/fn)
 for fn in ['motion.npy','time_bins.npy','depth_bins.npy','medicine_parameters.json','complete.json']:sources.append(SRC/'luke_early_medicine_peak_comparison_v6/compensated'/fn)
 settings=dict(interval_s=[930,1030],methods=['DREDGEv8','MEDiCINev6'],combinations=40,inputs=order,reference='Frozen17 waveform candidates and strict/lower-score/ambiguous events; seed930–940s,compare940–1030s. No reselection or fitting to references.',dredge='Documentedv8:0.25s bins,0.5s histogram sigma,20 windows,strict±250um pairwise bound',medicine='Documentedv6:0.25s bins,1s kernel,4 depth bins,motion_bound500,10000steps,seed0,CUDA; all supplied absolute amplitudes retained',reuse='Reuse cached5sigma/no-screen fits only after exact input array equality; normalize outputs without refitting. Remaining38 fits are fresh.',restart='Sequential methods inside independent systemd service; each child wrapped with persistent exit receipt. Completed arms hash checked; interrupted arm cannot resume optimizer and must be investigated/restarted.',source_sha256={str(p):sha(p) for p in sources})
 sp=OUT/'settings.json'
 if sp.exists():assert json.loads(sp.read_text())==settings,'Changed configuration needs new version'
 else:atomic(sp,settings)
 for fn,old in [('peaks.npy','compensated_peaks.npy'),('locations.npy','compensated_locations.npy')]:assert np.array_equal(np.load(INPUT/'fields'/baseline/fn),np.load(SRC/'luke_long_context_validation_v1'/old)), 'Baseline cache mismatch; do not reuse'
 rows=[]
 for name in order:
  for method in ['dredge','medicine']:
   stage=OUT/'arms'/f'{name}__{method}'
   if not valid(stage):
    stage.mkdir();tick=time.monotonic()
    if name==baseline:
     if method=='dredge':shutil.copy2(SRC/'luke_early_medicine_peak_comparison_v8/compensated_dredge_motion.npz',stage/'field.npz')
     else:
      p=SRC/'luke_early_medicine_peak_comparison_v6/compensated';np.savez_compressed(stage/'field.npz',time_s=np.load(p/'time_bins.npy'),depth_um=np.load(p/'depth_bins.npy'),displacement_um=np.load(p/'motion.npy'))
     atomic(stage/'fit_audit.json',dict(method=method,input=name,reused=True,inputs_exact=True,source='Documentedv8 DREDGE or v6 MEDiCINe compensated5sigma cache'))
    else:
     command=[str(PYTHON),'-m','testing.managed_job','--receipt',str(stage/'process_receipt.json'),'--cwd',str(ROOT),'--',str(PYTHON if method=='dredge' else MEDPY),'-m','testing.luke_ap_methods_sweep_fit_v1','--method',method,'--input',name,'--stage',str(stage)];atomic(stage/'launch_command.json',dict(command=command));env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4');print('START',method,name,flush=True)
     with (stage/'stdout.log').open('w') as out,(stage/'stderr.log').open('w') as err:subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err,check=True)
    z=np.load(stage/'field.npz');assert z['displacement_um'].shape==(len(z['time_s']),len(z['depth_um'])) and len(z['time_s'])==400 and np.isfinite(z['displacement_um']).all()
    if method=='dredge':assert abs(z['D']).max()<=250
    atomic(stage/'arm.json',dict(input=name,method=method,reused=name==baseline,seconds_this_launch=time.monotonic()-tick,field=str(stage/'field.npz')));seal(stage);print('SEALED',method,name,flush=True)
   rows.append(json.loads((stage/'arm.json').read_text()));pd.DataFrame(rows).to_csv(OUT/'manifest.csv',index=False)
  if name==baseline:
   from testing.luke_ap_methods_sweep_report_v1 import report
   report(baseline_only=True)
 assert settings['source_sha256']=={str(p):sha(p) for p in sources},'Source changed during fitting'
 from testing.luke_ap_methods_sweep_report_v1 import report
 report();atomic(OUT/'summary.json',dict(status='complete',combinations=len(rows),cached_reused=2,fresh_fits=38,seconds=time.monotonic()-start));print('COMPLETE40 combinations and reports',flush=True)
if __name__=='__main__':main()
