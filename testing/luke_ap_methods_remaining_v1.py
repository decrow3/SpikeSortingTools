"""Independent remaining-arm runner; preserves all original sweep evidence."""
import os,json,subprocess,time
from pathlib import Path
from testing.luke_ap_methods_sweep_v1 import ROOT,PYTHON,MEDPY,sha,seal,valid
OUT=ROOT/'testing/outputs/luke_ap_methods_remaining_v1'
def main():
 OUT.mkdir(exist_ok=True);rows=[]
 for method in ['medicine','dredge']:
  for screen in ['none','center_only','relaxed','full']:
   name=f'8sigma_{screen}_bound250';stage=OUT/f'{name}__{method}'
   if stage.exists():raise RuntimeError(f'Preserve existing stage: {stage}')
   stage.mkdir();src=ROOT/'testing/outputs/luke_early_sigma_screen250_v1/fields'/name
   checks=json.loads((src/'complete.json').read_text())
   for fn in ['peaks.npy','locations.npy']:assert sha(src/fn)==checks[fn]
   cmd=[str(PYTHON),'-m','testing.managed_job','--receipt',str(stage/'process_receipt.json'),'--cwd',str(ROOT),'--',str(MEDPY if method=='medicine' else PYTHON),'-m','testing.luke_ap_methods_remaining_fit_v1','--method',method,'--input',name,'--stage',str(stage)]
   (stage/'launch_command.json').write_text(json.dumps(cmd,indent=2));env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',DREDGE_AUDIT_DIR=str(stage))
   print('START',method,name,flush=True);start=time.monotonic()
   with (stage/'stdout.log').open('w') as out,(stage/'stderr.log').open('w') as err:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=out,stderr=err)
   rows.append(dict(input=name,method=method,returncode=r.returncode,seconds=time.monotonic()-start,field=str(stage/'field.npz')))
   if r.returncode==0:seal(stage)
   (OUT/'progress.json').write_text(json.dumps(rows,indent=2));print('FINISH',method,name,r.returncode,flush=True)
   if r.returncode and method=='dredge':break
if __name__=='__main__':main()
