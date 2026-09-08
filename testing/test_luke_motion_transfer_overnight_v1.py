"""Cheap checkpoint integrity controls; never access recording voltage."""
import json
import subprocess
import sys

import numpy as np
import pytest

from testing.luke_motion_transfer_overnight_v1 import atomic_json, committed_chunk, digest


def fixture_checkpoint(tmp_path):
    attempt = tmp_path / 'attempt_fixture'
    attempt.mkdir()
    p = np.array([(60, 1, 0, -12.)], dtype=[('sample_index','i8'),('channel_index','i8'),('segment_index','i8'),('amplitude','f4')])
    y = np.array([(20.,)], dtype=[('y','f8')])
    np.save(attempt / 'peaks.npy', p)
    np.save(attempt / 'locations.npy', y)
    expected = dict(settings_sha256='frozen',raw_bytes_sha256='input',num_samples=1000,num_channels=4)
    receipt = dict(input=expected,attempt=attempt.name,sha256={n:digest(attempt/n) for n in ['peaks.npy','locations.npy']})
    atomic_json(tmp_path / 'complete.json',receipt)
    return attempt,expected,p,y


def test_kill_before_commit_then_restart(tmp_path):
    code = "import pathlib,os,signal,sys; p=pathlib.Path(sys.argv[1])/'attempt_killed'; p.mkdir(); (p/'peaks.npy').write_bytes(b'partial'); os.kill(os.getpid(),signal.SIGKILL)"
    killed=subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=False)
    assert killed.returncode == -9
    assert committed_chunk(tmp_path,{}) is None
    _,expected,p,y=fixture_checkpoint(tmp_path)
    restored=committed_chunk(tmp_path,expected)
    assert np.array_equal(restored[0],p) and np.array_equal(restored[1],y)
    assert (tmp_path/'attempt_killed/peaks.npy').read_bytes()==b'partial'


def test_changed_settings_and_input_refused(tmp_path):
    _,expected,_,_=fixture_checkpoint(tmp_path)
    for key in ['settings_sha256','raw_bytes_sha256']:
        changed=dict(expected,**{key:'different'})
        with pytest.raises(AssertionError,match='input/settings mismatch'):
            committed_chunk(tmp_path,changed)


def test_corrupted_output_refused(tmp_path):
    attempt,expected,_,_=fixture_checkpoint(tmp_path)
    (attempt/'locations.npy').write_bytes(b'corrupted')
    with pytest.raises(AssertionError,match='artifact hash mismatch'):
        committed_chunk(tmp_path,expected)


def test_missing_output_refused(tmp_path):
    attempt,expected,_,_=fixture_checkpoint(tmp_path)
    (attempt/'peaks.npy').unlink()
    with pytest.raises(FileNotFoundError):
        committed_chunk(tmp_path,expected)


def test_empty_reference_figures(tmp_path):
    import pandas as pd
    from testing.luke_motion_transfer_overnight_v1 import figures
    p=np.array([(100,1,0,-12.)],dtype=[('sample_index','i8'),('channel_index','i8'),('segment_index','i8'),('amplitude','f4')])
    y=np.array([(200.,)],dtype=[('y','f8')])
    t=np.arange(6000.5,6100,1); depths=np.array([200.,400.])
    field=dict(time_s=t,depth_um=depths,displacement_um=np.zeros((100,2)),D=np.zeros((2,100,100)),C=np.ones((2,100,100)),U=np.ones((2,100,100)))
    tracks=pd.DataFrame(columns=['candidate','time_s','events','centroid_um','depth_um'])
    events=pd.DataFrame(columns=['candidate','frame'])
    figures(tmp_path,'synthetic',6000,6100,p,y,field,30000.,tracks,events,None)
    assert pd.read_csv(tmp_path/'cached_observations.csv').empty
    assert set(pd.read_csv(tmp_path/'sampled_cycle_consistency.csv').cycle_p95_um)=={0.}
    for name in ['01_motion_raster','02_cached_waveform_corroboration','03_pairwise_support']:
        for ext in ['png','pdf']:
            assert (tmp_path/f'{name}.{ext}').stat().st_size>0
