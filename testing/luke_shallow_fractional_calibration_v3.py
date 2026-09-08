"""Approximate fractional injection sensitivity; frozen unit154 matcher, no motion input."""
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from testing.luke_shallow_identity_audit_v3 import score, adjudicate
from testing.luke_lowpass_waveform_preservation_v2 import reconstruct, digest, SRC, BASE

OUT=SRC/'luke_shallow_fractional_calibration_v3'
BANK=SRC/'luke_shallow_identity_audit_v3'
FRACTIONS=[-20,-10,-5,0,5,10,20]
GAINS=[.75,1.,1.25]
N_BACKGROUND=40


def translate(wave, geometry, shift):
    """Linear interpolation within each actual x column; zeros beyond support."""
    result=np.zeros_like(wave)
    for x in np.unique(geometry[:,0]):
        ix=np.flatnonzero(geometry[:,0]==x)
        ix=ix[np.argsort(geometry[ix,1])]
        depth=geometry[ix,1]
        for t in range(len(wave)):
            result[t,ix]=np.interp(depth-shift,depth,wave[t,ix],left=0,right=0)
    return result


def centroid(wave, depths):
    energy=np.sum(wave*wave,axis=-2)
    return np.sum(energy*depths,axis=-1)/np.maximum(np.sum(energy,axis=-1),1e-20)


def main():
    begun=time.monotonic();OUT.mkdir(exist_ok=False)
    manifestfile=BASE/'recording/rescue_recording_manifest.json'
    modelfile=SRC/'luke_common_event_screen_v1/shared_response_model.npz'
    bankfile=BANK/'u154_bank.npz';infofile=BANK/'u154_hypotheses.csv'
    m=json.loads(manifestfile.read_text());fs=m['sampling_frequency_hz'];allgeo=np.asarray(m['channel_locations_um'])
    bank=np.load(bankfile);channels=bank['channels'];geometry=allgeo[channels];info=pd.read_csv(infofile).to_dict('records')
    hypotheses=bank['training'];weights=bank['common_weights']
    idx=next(i for i,r in enumerate(info) if r['identity']=='label_154' and r['shift_um']==0)
    independent=bank['independent_half'][idx]
    settings=dict(target=154,background_s=[4100,4110],known_center_injections=True,requested_shifts_um=FRACTIONS,gains=GAINS,paired_backgrounds_per_gain=N_BACKGROUND,
        model='Linear spatial interpolation within actual same-x columns; zero outside saved waveform support. Approximate injection model, not biological ground truth.',
        matching='Frozen qualified bank/weights/cosine0.8/gain0.4–2.5/identitymargin0.03; shiftunique margin0.03. Known centers, ±3-sample timing; no independent detection or detection-recall claim.',
        centroid='Same transition measurement: raw squared-energy centroid within±120um of620um+winning gridshift, at accepted matched timing.',
        comparison='Paired same-background/same-gain displacement relativezero; report only pairs accepted with unique shifts at both requestedshift andzero; persist rejection counts and allscores.',
        limitations=['Spatial interpolation may alter shape and is not a validated physical tissue-displacement model.',
                    'Conditional selection can bias recovered sensitivity; acceptance by shift/gain is mandatory context.',
                    'Quiet backgrounds and one template do not establish transition-time identity or amplitude stationarity.',
                    'Coarse zero-shift winner does not mean zero physical motion. No DREDGE field is read.'],
        sha256={str(p):digest(p) for p in [Path(__file__).resolve(),bankfile,infofile,manifestfile,modelfile,Path(__file__).with_name('luke_shallow_identity_audit_v3.py'),Path(__file__).with_name('luke_lowpass_waveform_preservation_v2.py')]})
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    regressions=[]
    for shift in [-80,-40,0,40,80]:
        ref=next(i for i,r in enumerate(info) if r['identity']=='label_154' and r['shift_um']==shift)
        predicted=translate(independent,geometry,shift);expected=bank['independent_half'][ref]
        error=float(np.max(abs(predicted-expected)))
        assert np.allclose(predicted,expected,atol=1e-5,rtol=1e-6),f'Exact geometry regression failed at{shift}'
        regressions.append(dict(shift_um=shift,max_absolute_waveform_error_uv=error))
    pd.DataFrame(regressions).to_csv(OUT/'exact_geometry_regression.csv',index=False)
    quiet,unused,first=reconstruct(4100,m,np.load(modelfile));del unused
    quiet=quiet[:round(10*fs)]
    rng=np.random.default_rng(20260908154)
    frames=rng.integers(40,len(quiet)-40,size=N_BACKGROUND)
    backgrounds=quiet[frames[:,None,None]+np.arange(-33,34)[None,:,None],channels[None,None,:]].copy();del quiet
    rows=[];allscore=[];allgain=[];alllag=[];shifted={}
    centerfoot=abs(geometry[:,1]-620)<=120
    cleanzero=float(centroid(independent[:,centerfoot],geometry[centerfoot,1]))
    for shift in FRACTIONS:
        injection=translate(independent,geometry,shift);shifted[f'shift_{shift}']=injection
        clean_displacement=float(centroid(injection[:,centerfoot],geometry[centerfoot,1])-cleanzero)
        for amplitude in GAINS:
            waves=backgrounds.copy();waves[:,3:64]+=amplitude*injection
            sc,ga,la=score(waves,hypotheses,weights);accepted,col,top,margin,sm=adjudicate(sc,info,154)
            unique=sm>=.03
            for j in range(N_BACKGROUND):
                bestshift=info[col[j]]['shift_um'];foot=abs(geometry[:,1]-(620+bestshift))<=120
                lag=int(la[j,col[j]]);w=waves[j,3+lag:64+lag,foot].T
                # NumPy mixed indexing moves channel axis first; transpose restores time×channel.
                cent=float(centroid(w,geometry[foot,1]))
                rows.append(dict(requested_shift_um=shift,gain=amplitude,pair_id=j,background_frame=int(first+frames[j]),
                    accepted_identity=bool(accepted[j]),unique_shift=bool(unique[j]),best_shift_um=bestshift,score=float(top[j]),
                    identity_margin=float(margin[j]),shift_margin=float(sm[j]),timing_lag=lag,centroid_um=cent,
                    clean_model_centroid_displacement_um=clean_displacement))
            allscore.append(sc);allgain.append(ga);alllag.append(la)
    data=pd.DataFrame(rows)
    baseline=data[data.requested_shift_um==0][['gain','pair_id','accepted_identity','unique_shift','centroid_um']].rename(columns={k:'zero_'+k for k in ['accepted_identity','unique_shift','centroid_um']})
    data=data.merge(baseline,on=['gain','pair_id'],validate='many_to_one')
    data['pair_supported']=data.accepted_identity&data.unique_shift&data.zero_accepted_identity&data.zero_unique_shift
    data['paired_centroid_displacement_um']=np.where(data.pair_supported,data.centroid_um-data.zero_centroid_um,np.nan)
    data.to_csv(OUT/'injected_events.csv',index=False)
    np.savez_compressed(OUT/'all_scores.npz',scores=np.concatenate(allscore),gains=np.concatenate(allgain),timing_lags=np.concatenate(alllag))
    np.savez_compressed(OUT/'injection_templates.npz',channels=channels,**shifted)
    summaries=[]
    for (shift,amplitude),g in data.groupby(['requested_shift_um','gain'],sort=True):
        supported=g[g.pair_supported];delta=supported.paired_centroid_displacement_um
        summaries.append(dict(requested_shift_um=shift,gain=amplitude,injections=len(g),accepted_fraction=float(g.accepted_identity.mean()),
            accepted_unique_fraction=float((g.accepted_identity&g.unique_shift).mean()),paired_supported=len(supported),
            zero_grid_winner_fraction=float((g.best_shift_um==0).mean()),
            median_recovered_um=float(delta.median()) if len(delta) else np.nan,
            q10_recovered_um=float(delta.quantile(.1)) if len(delta) else np.nan,q90_recovered_um=float(delta.quantile(.9)) if len(delta) else np.nan,
            clean_model_centroid_displacement_um=float(g.clean_model_centroid_displacement_um.iloc[0])))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'sensitivity_by_shift_gain.csv',index=False)
    slopes=[]
    for amplitude,g in summary.groupby('gain'):
        ok=g.median_recovered_um.notna()&(g.paired_supported>=10)
        slope,intercept=np.polyfit(g.loc[ok,'requested_shift_um'],g.loc[ok,'median_recovered_um'],1) if ok.sum()>=3 else [np.nan,np.nan]
        slopes.append(dict(gain=amplitude,descriptive_median_response_slope=float(slope),intercept_um=float(intercept),supported_shift_levels=int(ok.sum())))
    pd.DataFrame(slopes).to_csv(OUT/'descriptive_slopes.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for amplitude,color in zip(GAINS,['#2468a2','#cc7722','#555555']):
        g=summary[summary.gain==amplitude]
        axes[0].plot(g.requested_shift_um,g.accepted_unique_fraction,'o-',color=color,label=f'gain{amplitude}')
        axes[1].plot(g.requested_shift_um,g.median_recovered_um,'o-',color=color,label=f'gain{amplitude}')
        axes[1].fill_between(g.requested_shift_um,g.q10_recovered_um,g.q90_recovered_um,color=color,alpha=.12)
        axes[2].plot(g.requested_shift_um,g.paired_supported,'o-',color=color,label=f'gain{amplitude}')
    axes[0].set_ylabel('Identity accepted + unique shift fraction');axes[0].set_ylim(0,1.05)
    axes[1].plot(FRACTIONS,FRACTIONS,':',color='black',label='Requested displacement');axes[1].set_ylabel('Paired energy-centroid displacement (µm)')
    axes[2].set_ylabel('Supported zero/shift pairs (of40)');axes[2].set_ylim(0,42)
    for ax in axes:ax.set_xlabel('Injected interpolation shift (µm)');ax.legend(fontsize=8)
    fig.suptitle('Unit154 fractional-injection sensitivity with frozen40µm-grid matcher\nLinear same-column interpolation; paired real quiet backgrounds; shaded event10–90% range, not confidence intervals')
    for ext in ['png','pdf']:fig.savefig(OUT/f'01_fractional_sensitivity.{ext}',dpi=150)
    result=dict(status='complete',seconds=time.monotonic()-begun,injections=len(data),exact_geometry_regression='all5shifts pass',slopes=slopes,
                interpretation='Approximate injection-model sensitivity only; no DREDGE comparison, detection recall, or biological motion ground truth.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
