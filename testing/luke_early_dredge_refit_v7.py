"""Fresh cached-peak DREDGE fits with unchanged settings."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from spikeinterface.core import NumpyRecording
from testing.luke_dredge_bounded_250_v1 import estimate_bounded
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'testing/outputs'; PRE=SRC/'luke_long_context_validation_v1'
OUT=SRC/'luke_early_medicine_peak_comparison_v7'
def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    m=json.loads(Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording/rescue_recording_manifest.json').read_text())
    fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um'])
    cfg=json.loads((PRE/'settings.json').read_text())['estimator'];cfg.update(bin_s=.25,max_disp_um=250.,histogram_time_smooth_s=.25)
    (OUT/'dredge_settings.json').write_text(json.dumps(dict(estimator=cfg,strict_pairwise_bound_um=250,scope='Fresh fit on exact 930–1030 cached peaks, 0.25 s bins, 0.25 s Gaussian histogram smoothing, ±250 µm strict pairwise bound',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    record=NumpyRecording(np.broadcast_to(np.zeros((1,len(geo)),np.float32),(round(100*fs),len(geo))),fs);record.set_channel_locations(geo)
    for arm in ['original','compensated']:
        dest=OUT/f'{arm}_dredge_motion.npz'
        if dest.exists(): continue
        t=time.monotonic();p=np.load(PRE/f'{arm}_peaks.npy');loc=np.load(PRE/f'{arm}_locations.npy')
        motion,extra=estimate_bounded(record,p,loc,cfg)
        delta=float(np.max(np.abs(motion.displacement[0]-np.load(SRC/'luke_early_medicine_peak_comparison_v6'/f'{arm}_dredge_motion.npz')['displacement_um'])))
        np.savez_compressed(dest,time_s=motion.temporal_bins_s[0]+930,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],**extra)
        (OUT/f'{arm}_dredge_receipt.json').write_text(json.dumps(dict(runtime_s=time.monotonic()-t,max_difference_from_cached_um=delta,peaks=len(p)),indent=2))
        print(arm,'DREDGE completed; max change',delta,flush=True)
if __name__=='__main__':main()
