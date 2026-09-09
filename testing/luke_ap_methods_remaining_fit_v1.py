"""One independently managed arm of documented DREDGEv8/MEDiCINev6."""
import os
import argparse,hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs'
MED_ARGS=dict(motion_bound=500.,time_bin_size=.25,time_kernel_width=1.,activity_network_hidden_features=(256,256),num_depth_bins=4,amplitude_threshold_quantile=0.,batch_size=4096,training_steps=10000,initial_motion_noise=.1,motion_noise_steps=2000,learning_rate=.0005,epsilon=.001,plot_figures=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['dredge','medicine'],required=True);ap.add_argument('--input',required=True);ap.add_argument('--stage',type=Path,required=True);args=ap.parse_args();stage=args.stage;os.environ.setdefault('DREDGE_AUDIT_DIR',str(stage));source=SRC/'luke_early_sigma_screen250_v1/fields'/args.input;mp=Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording/rescue_recording_manifest.json');m=json.loads(mp.read_text());fs=m['sampling_frequency_hz'];p=np.load(source/'peaks.npy');loc=np.load(source/'locations.npy');torch.set_num_threads(4);start=time.monotonic();np.random.seed(0);torch.manual_seed(0)
 if args.method=='dredge':
  from spikeinterface.core import NumpyRecording
  from testing.luke_dredge_bounded_250_audit_v1 import estimate_bounded
  cfg=json.loads((SRC/'luke_early_medicine_peak_comparison_v8/dredge_settings.json').read_text())['estimator'];assert cfg['bin_s']==.25 and cfg['histogram_time_smooth_s']==.5 and cfg['max_disp_um']==250
  rec=NumpyRecording(np.broadcast_to(np.zeros((1,384),np.float32),(round(100*fs),384)),fs);rec.set_channel_locations(np.array(m['channel_locations_um']));motion,extra=estimate_bounded(rec,p,loc,cfg);t=motion.temporal_bins_s[0]+930;y=motion.spatial_bins_um;v=motion.displacement[0];assert abs(extra['D']).max()<=250;np.savez_compressed(stage/'field.npz',time_s=t,depth_um=y,displacement_um=v,**extra);settings=cfg;implementation=Path(inspect.getsourcefile(estimate_bounded));device='cpu'
 else:
  import medicine
  assert torch.cuda.is_available(),'CUDA required to reproduce documented MEDiCINe configuration'
  torch.cuda.manual_seed_all(0);settings=dict(MED_ARGS,optimizer='torch.optim.Adam',seed=0);device=torch.cuda.get_device_name(0);implementation=Path(inspect.getsourcefile(medicine.run_medicine));medicine.run_medicine(peak_times=p['sample_index']/fs+930,peak_depths=loc['y'],peak_amplitudes=abs(p['amplitude']),output_dir=stage/'fit',optimizer=torch.optim.Adam,**MED_ARGS);t=np.load(stage/'fit/time_bins.npy');y=np.load(stage/'fit/depth_bins.npy');v=np.load(stage/'fit/motion.npy');np.savez_compressed(stage/'field.npz',time_s=t,depth_um=y,displacement_um=v)
 assert v.shape==(len(t),len(y)) and len(t)==400 and np.isfinite(v).all() and np.all(np.diff(t)>0) and np.all(np.diff(y)>0)
 (stage/'fit_audit.json').write_text(json.dumps(dict(method=args.method,input=args.input,peaks=len(p),settings=settings,device=device,torch_version=torch.__version__,implementation=str(implementation),implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest(),runtime_s=time.monotonic()-start,shape=list(v.shape),time_support_s=[float(t[0]),float(t[-1])]),indent=2));print(args.method,args.input,'complete',round(time.monotonic()-start,1),'s',flush=True)
if __name__=='__main__':main()
