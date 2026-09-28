"""Materialize matched q=0 and lattice-remapped W2 rescue-KS inputs."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,time,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from testing.dd_lattice_remap import remap_chunk,sample_q

FS=29999.759166666667;START=26999783;END=37199701;NC=384
SOURCE=Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec1/recording/traces_cached_seg0.raw')
RECORDING=Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec1/recording')
KNOTS=Path('/mnt/NPX/Luke/DARTsort_motion_experiments/dd_lattice_w2_20260928/host_h1/preflight/W2_KNOT_Q.csv')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def wj(p,x):
 q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def main():
 raise RuntimeError("SUPERSEDED: full-384 remap and remap-before-time-filter violate the frozen DD AP202:383 preprocessing boundary")
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--chunk-seconds',type=float,default=1.0);a=ap.parse_args()
 out=a.output.resolve();
 if out.exists():raise FileExistsError(out)
 if shutil.disk_usage(out.parent).free<60_000_000_000:raise RuntimeError('60 GB free-space gate')
 out.mkdir(parents=True);started=time.perf_counter();knots=pd.read_csv(KNOTS)
 from spikeinterface.core import load
 rec=load(RECORDING);geom=np.asarray(rec.get_channel_locations(),float);source=np.memmap(SOURCE,dtype=np.int16,mode='r',shape=(rec.get_num_samples(),NC))
 n=END-START;step=max(1,int(round(a.chunk_seconds*FS)));arms={}
 for arm,use_q in [('KS_0',False),('KS_L',True)]:
  partial=out/f'{arm}.raw.partial';dest=np.memmap(partial,dtype=np.int16,mode='w+',shape=(n,NC));zeros=0
  for lo in range(0,n,step):
   hi=min(n,lo+step);src=np.asarray(source[START+lo:START+hi]);q=sample_q(hi-lo,START+lo,FS,knots) if use_q else np.zeros(hi-lo,np.int64);mapped,info=remap_chunk(src,q,geom);dest[lo:hi]=mapped;zeros+=info['zero_filled_values']
  dest.flush();del dest;final=out/f'{arm}.raw';os.replace(partial,final)
  arms[arm]={'path':str(final),'bytes':final.stat().st_size,'sha256':sha(final),'zero_filled_values':zeros,'q0_expected_crop_byte_equal':not use_q}
 if arms['KS_0']['sha256']!=hashlib.sha256(memoryview(np.asarray(source[START:END])).cast('B')).hexdigest():raise ValueError('q0 crop is not byte-identical')
 wj(out/'MATERIALIZATION_RECEIPT.json',{'status':'complete','source_frames_half_open':[START,END],'sampling_frequency_hz':FS,'channels':NC,'dtype':'int16','arms':arms,'elapsed_s':time.perf_counter()-started,'source_binary':str(SOURCE),'source_preprocessing':'accepted phase/blank/AP191 interpolate only','sorter_preprocessing':'Kilosort highpass/CAR/whitening applied once later'})
 wj(out/'COMPLETE.json',{'status':'complete','receipt_sha256':sha(out/'MATERIALIZATION_RECEIPT.json'),'written_last':True})
if __name__=='__main__':main()
