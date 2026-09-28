"""Run one frozen rescue Kilosort arm on a materialized DD W2 binary."""
from __future__ import annotations
import argparse,hashlib,json,math,os,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.kilosort_compat import ensure_kilosort_compatibility
from pipeline.sorting import build_kilosort4_params,validate_applied_settings

FS=29999.759166666667;N=10199918;NC=384
ORIGINAL=Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec1/recording')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def wj(p,x):
 q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def main():
 raise RuntimeError("SUPERSEDED: NC=384 worker is not authorized; use reviewed AP202:383 filtered-parent input and explicit whitening policy")
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['KS_0','KS_L'],required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 out=a.output.resolve();partial=out.with_name(out.name+'.partial')
 if out.exists() or partial.exists():raise FileExistsError(out)
 if shutil.disk_usage(out.parent).free<60_000_000_000:raise RuntimeError('60 GB free-space gate')
 partial.mkdir(parents=True);started=time.perf_counter();wj(partial/'START_RECEIPT.json',{'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'arm':a.arm,'input':str(a.input),'input_sha256':sha(a.input),'sampling_frequency_hz':FS,'samples':N,'channels':NC,'rng_policy':'frozen rescue Kilosort defaults/seed path','training_policy':'arm-local adaptive fits; identical settings apart from input q'})
 ensure_kilosort_compatibility()
 from spikeinterface.core import BinaryRecordingExtractor,load
 from spikeinterface.sorters import run_sorter
 original=load(ORIGINAL);rec=BinaryRecordingExtractor(a.input,FS,'int16',num_channels=NC,channel_ids=original.get_channel_ids(),gain_to_uV=2.34375,offset_to_uV=0.,is_filtered=False);rec=rec.set_probe(original.get_probe(),in_place=False)
 params=build_kilosort4_params();sortdir=partial/'kilosort4'
 run_sorter('kilosort4',rec,folder=str(sortdir),verbose=True,remove_existing_folder=False,**params)
 so=sortdir/'sorter_output';ops=np.load(so/'ops.npy',allow_pickle=True).item();applied=dict(ops.get('settings',{}));applied.update(ops);validated=validate_applied_settings(applied,params)
 times=np.load(so/'spike_times.npy',mmap_mode='r').reshape(-1);clusters=np.load(so/'spike_clusters.npy',mmap_mode='r').reshape(-1)
 if np.any((times<0)|(times>=N)):raise ValueError('sort time outside local W2 clock')
 wj(partial/'FINAL_RECEIPT.json',{'status':'complete','arm':a.arm,'elapsed_s':time.perf_counter()-started,'events':int(times.size),'units':int(np.unique(clusters).size),'local_clock_half_open':[0,N],'source_clock_half_open':[26999783,37199701],'critical_effective_settings':validated,'do_CAR':bool(applied['do_CAR']),'highpass_cutoff':float(applied['highpass_cutoff']),'whitening_range':int(applied['whitening_range']),'all_final_rows_saved':True})
 wj(partial/'COMPLETE.json',{'status':'complete','receipt_sha256':sha(partial/'FINAL_RECEIPT.json'),'written_last':True});os.replace(partial,out)
if __name__=='__main__':main()
