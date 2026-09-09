"""Fit a saved screened peak population using the documented MEDiCINe model."""
import argparse, json, time, resource, hashlib, inspect
from pathlib import Path
import numpy as np
import torch
from testing.luke_ap_methods_sweep_fit_v1 import MED_ARGS

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=10000)
    args=ap.parse_args()
    import medicine
    assert torch.cuda.is_available(), 'Documented configuration requires CUDA'
    cfg=json.loads((args.input/'input.json').read_text())
    p=np.load(args.input/'peaks.npy',mmap_mode='r');loc=np.load(args.input/'locations.npy',mmap_mode='r')
    times=p['sample_index']/cfg['sampling_frequency_hz']+cfg['start_s']
    assert len(p)==len(loc) and len(p)>0 and np.isfinite(loc['y']).all()
    assert times.min()>=cfg['start_s'] and times.max()<cfg['stop_s']
    torch.set_num_threads(4);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats();kw=dict(MED_ARGS,training_steps=args.steps)
    started=time.monotonic()
    trainer=medicine.run_medicine(peak_times=times,peak_depths=loc['y'],peak_amplitudes=abs(p['amplitude']),output_dir=args.output/'fit',optimizer=torch.optim.Adam,**kw)
    t=np.load(args.output/'fit/time_bins.npy');y=np.load(args.output/'fit/depth_bins.npy');v=np.load(args.output/'fit/motion.npy')
    assert v.shape==(len(t),4) and np.isfinite(v).all() and np.all(np.diff(t)>0) and np.all(np.diff(y)>0)
    assert abs(len(t)-(cfg['stop_s']-cfg['start_s'])/.25)<=1
    np.savez_compressed(args.output/'field.npz',time_s=t,depth_um=y,displacement_um=v)
    np.save(args.output/'loss.npy',np.asarray(trainer.losses))
    impl=Path(inspect.getsourcefile(medicine.run_medicine));model=impl.with_name('model.py')
    audit=dict(settings=kw,seed=0,optimizer='torch.optim.Adam',peaks=len(p),seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),torch_version=torch.__version__,time_support_s=[float(t[0]),float(t[-1])],source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),impl,model]})
    (args.output/'fit_audit.json').write_text(json.dumps(audit,indent=2))
    print('FIT COMPLETE',audit['seconds'],flush=True)
if __name__=='__main__':main()
