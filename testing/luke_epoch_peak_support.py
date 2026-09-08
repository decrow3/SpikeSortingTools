"""Compare existing imec0 detection support, without equating detector amplitudes."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_epoch_corroboration import DATA,BASE,SHARED,OUT,savefig

def main():
    p=DATA/'dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion'; peaks=np.load(p/'peaks.npy',mmap_mode='r');loc=np.load(p/'peak_locations.npy',mmap_mode='r');fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz'];field=np.load(SHARED/'native_rigid_field.npz');t=field['time_s'];native=pd.read_csv(SHARED/'audit/detection_counts.csv');assert np.allclose(native.time_s,t)
    selected=np.flatnonzero((t>=8160)&(t<8280));half=(t[1]-t[0])/2;edges=np.r_[t[selected]-half,t[selected[-1]]+half];frames=np.ceil(edges*fs).astype('int64');left,right=np.searchsorted(peaks['sample_index'],frames[[0,-1]]); pp=peaks[left:right];yy=loc['y'][left:right];counts=np.histogram(pp['sample_index'],frames)[0];rows=[]
    for k,i in enumerate(selected):
        take=(pp['sample_index']>=frames[k])&(pp['sample_index']<frames[k+1]);amp=np.abs(pp['amplitude'][take]);depth=yy[take];rows.append(dict(time_s=t[i],historical_AP_count=int(take.sum()),native_count=int(native.detections.iloc[i]),historical_amplitude_median=float(np.median(amp)),historical_fraction_depth_lt1400=float(np.mean(depth<1400)),native_physical_um=float(field['physical_displacement_um'][i,0])))
    df=pd.DataFrame(rows);df.to_csv(OUT/'imec0_peak_support.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True);axes[0].plot(df.time_s,df.historical_AP_count,label='Historical AP detections');axes[0].plot(df.time_s,df.native_count,label='Native KS drift detections');axes[0].set_ylabel('Events per ≈2 s');axes[0].legend();axes[1].plot(df.time_s,df.historical_fraction_depth_lt1400);axes[1].set_ylabel('Historical fraction\nshallower than 1400 µm');axes[1].set_ylim(0,1);axes[2].plot(df.time_s,df.native_physical_um);axes[2].set_ylabel('Native displacement (µm)');axes[2].set_xlabel('Seconds from recording frame zero');fig.suptitle('Existing imec0 peak support on common time bins\nDifferent conditioning and detectors: count differences are descriptive, not a controlled filter effect');fig.tight_layout(rect=[0,0,1,.92]);savefig(fig,'05_imec0_peak_support')
    result=dict(status='existing_support_aligned',bins=len(df),historical_events=int(counts.sum()),native_events=int(df.native_count.sum()),native_amplitude_comparison='Unavailable: native event table remains on huklaban5; shared counts are available. Amplitude units would need calibration before comparison.',historical_source=str(p),native_source=str(SHARED/'audit/detection_counts.csv'),time_method='Native batch centers ± half batch; historical sample indices converted at actual imec0 sampling frequency. No clock shift fitting.')
    (OUT/'peak_support_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
