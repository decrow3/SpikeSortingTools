import numpy as np
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from testing.luke_screened_medicine import screened_features
from testing.luke_screen_sweep import features,mask

def test_bounded_feature_batches_preserve_features_and_masks():
    rng=np.random.default_rng(2);geo=np.c_[np.tile([0,20],192),np.repeat(np.arange(192)*20,2)];clean=rng.normal(0,2,(500,384)).astype('float32');post=clean+rng.normal(0,.1,clean.shape).astype('float32');p=np.zeros(19,dtype=[('sample_index','i8'),('channel_index','i8'),('amplitude','f8'),('segment_index','i8')]);p['sample_index']=np.arange(19)*15+70;p['channel_index']=np.arange(19)*19
    noise=np.ones(384);old=features(clean,post,p,geo,noise,30000);new=screened_features(clean,post,p,geo,noise,30000,batch=3)
    for k in old:np.testing.assert_allclose(new[k],old[k],rtol=1e-6,atol=1e-8)
    for kw in [{},dict(snr=6,center=.5,neighbor=.6)]:np.testing.assert_array_equal(mask(old,**kw),mask(new,**kw))

def test_localization_is_independent_of_discarded_events():
    fs=30000;geo=np.c_[np.tile([0,20],8),np.repeat(np.arange(8)*20,2)];x=np.zeros((3000,16),np.float32);p=np.zeros(4,dtype=[('sample_index','i8'),('channel_index','i8'),('amplitude','f8'),('segment_index','i8')]);p['sample_index']=[400,900,1500,2100];p['channel_index']=[3,6,9,12]
    for event in p:
        amps=1000/np.sqrt(np.sum((geo-geo[event['channel_index']])**2,axis=1)+400)
        x[event['sample_index']-20:event['sample_index']+21]-=np.exp(-np.arange(-20,21)**2/25)[:,None]*amps
    rec=NumpyRecording(x,fs);rec.set_channel_locations(geo);kw=dict(method='monopolar_triangulation',radius_um=75.,n_jobs=1,progress_bar=False)
    all_loc=localize_peaks(rec,p,**kw);subset=localize_peaks(rec,p[[0,2]],**kw)
    for key in all_loc.dtype.names:np.testing.assert_allclose(subset[key],all_loc[key][[0,2]],rtol=1e-6,atol=1e-6)
