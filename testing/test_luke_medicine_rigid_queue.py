import numpy as np
import pytest
from spikeinterface.core import NumpyRecording, load

from testing.luke_medicine_rigid_queue import (
    apply_rigid, checked_int16, common_channel_indices, dependency_decision, validate_field,
)

GOOD = {'state':'complete','returncode':0}
EXIT = {'SERVICE_RESULT':'success','EXIT_STATUS':'0'}


def test_queue_waits_for_actual_exit_even_when_success_artifacts_exist():
    assert dependency_decision({'ActiveState':'active','process_exists':True},GOOD,EXIT) == 'wait'
    assert dependency_decision({'ActiveState':'inactive','process_exists':False},GOOD,EXIT) == 'ready'
    with pytest.raises(RuntimeError,match='without successful'):
        dependency_decision({'ActiveState':'inactive'},None,None)


@pytest.mark.parametrize('receipt,result', [({'state':'failed','returncode':1},None),
                                           (GOOD,{'SERVICE_RESULT':'signal','EXIT_STATUS':'TERM'})])
def test_failed_or_cancelled_estimator_never_releases_sort(receipt,result):
    with pytest.raises(RuntimeError):
        dependency_decision({'ActiveState':'inactive'},receipt,result)


def test_field_rejects_internal_gaps_and_excessive_terminal_extrapolation():
    t=np.arange(.1,3.,.25)
    z=dict(time_s=t,depth_um=np.array([0.,1000.]),rigid_displacement_um=np.zeros(len(t)),nonrigid_displacement_um=np.zeros((len(t),2)))
    validate_field(z,1000,3000,1.)
    with pytest.raises(ValueError,match='cover recording'):
        validate_field(z,1000,5000,1.)
    z['time_s']=t.copy();z['time_s'][4]+=.1
    with pytest.raises(ValueError,match='sampling'):
        validate_field(z,1000,3000,1.)


def test_common_domain_accounts_for_both_arms_and_signed_extremes():
    geo=np.c_[np.zeros(192),np.arange(192)*20.]
    rigid=np.array([-80.,40.]);nonrigid=np.array([[-120.,50.],[-100.,140.]])
    ix=common_channel_indices(geo,np.array([0.,3820.]),rigid,nonrigid,60.)
    assert geo[ix,1].min()==180 and geo[ix,1].max()==3620
    for d in [-120.,140.]:
        assert (geo[ix,1]+d>=60).all() and (geo[ix,1]+d<=3760).all()


def test_nonzero_acquisition_clock_and_signed_translation_are_correct():
    fs=1000.;geo=np.c_[np.zeros(96),np.arange(96)*20.]
    wave=-np.exp(-((np.arange(40)-20)/4)**2)
    target=900.
    for shift in [-400.,-40.,7.,40.,400.]:
        profile=np.exp(-.5*((geo[:,1]-(target+shift))/40)**2)
        rec=NumpyRecording((wave[:,None]*profile).astype('float32'),fs,t_starts=[3057.677050340359])
        rec.set_channel_locations(geo)
        corrected=apply_rigid(rec,np.array([0.,.25]),np.array([shift,shift]),dict(sigma_um=20.,p=1))
        voltage=corrected.get_traces();weights=np.max(abs(voltage),axis=0)
        centroid=np.sum(weights*corrected.get_channel_locations()[:,1])/sum(weights)
        # Exact site translations preserve the centroid; fractional interpolation
        # is approximate (allow 1 µm on this 20 µm grid, not biological precision).
        assert abs(centroid-target)<(1. if shift==7. else .1)
    # A changing trajectory detects failure to map field time to acquisition time.
    rec=NumpyRecording(np.tile(profile,(1000,1)).astype('float32'),fs,t_starts=[3057.677050340359])
    rec.set_channel_locations(geo)
    corrected=apply_rigid(rec,np.array([0.,.25,.5,.75]),np.array([0.,40.,-40.,0.]),dict(sigma_um=20.,p=1))
    a=corrected.get_traces(start_frame=0,end_frame=10)
    b=corrected.get_traces(start_frame=250,end_frame=260)
    assert not np.allclose(a,b)


def test_checked_conversion_materializes_with_full_time_and_refuses_overflow(tmp_path):
    rec=NumpyRecording(np.tile(np.arange(8),(120,1)).astype('float32')+.6,1000.,t_starts=[3057.])
    rec.set_channel_locations(np.c_[np.zeros(8),np.arange(8)*20.])
    guarded=checked_int16(rec)
    guarded.save(folder=tmp_path/'recording',n_jobs=1,progress_bar=False)
    saved=load(tmp_path/'recording')
    assert saved.get_num_samples()==120
    assert saved.sample_index_to_time(0)==3057.
    np.testing.assert_array_equal(saved.get_traces()[0],np.arange(8)+1)
    for invalid in [40000.,np.nan]:
        bad=checked_int16(NumpyRecording(np.full((20,8),invalid,dtype='float32'),1000.))
        with pytest.raises(ValueError):bad.get_traces()
