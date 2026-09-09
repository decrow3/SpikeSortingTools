from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from testing.luke_full_probe_registration import registration_only


def fixture_api():
    reader=Mock()
    ops={'stage':'fixture'}
    api=SimpleNamespace(initialize_ops=Mock(return_value=ops),
        compute_preprocessing=Mock(return_value=ops),
        compute_drift_correction=Mock(return_value=(ops,reader,'detections')),
        run_kilosort=Mock(),detect_spikes=Mock(),cluster_spikes=Mock(),save_sorting=Mock(),
        template_matching=SimpleNamespace(extract=Mock()),clustering_qr=SimpleNamespace(run=Mock()))
    return api,reader


def test_only_registration_stages_and_reader_closed():
    api,reader=fixture_api();stages=[];seed=Mock()
    ops,st=registration_only(api,{}, {},'cpu',stages.append,seed)
    assert st=='detections'
    assert stages==['initialization','preprocessing','drift_estimation_including_own_detections','registration_returned_closing_binary']
    reader.close.assert_called_once();seed.assert_called_once()
    for name in ['run_kilosort','detect_spikes','cluster_spikes','save_sorting']:
        getattr(api,name).assert_not_called()
    api.template_matching.extract.assert_not_called();api.clustering_qr.run.assert_not_called()


@pytest.mark.parametrize('name',['run_kilosort','detect_spikes','cluster_spikes','save_sorting'])
def test_downstream_api_entry_is_blocked_even_if_called_inside_drift(name):
    api,_=fixture_api()
    api.compute_drift_correction.side_effect=lambda *a,**kw:getattr(api,name)()
    with pytest.raises(RuntimeError,match='boundary'):
        registration_only(api,{}, {},'cpu',lambda stage:None,lambda:None)


def test_close_on_post_return_callback_failure():
    api,reader=fixture_api()
    def progress(stage):
        if stage=='registration_returned_closing_binary':raise ValueError('disk unavailable')
    with pytest.raises(ValueError):registration_only(api,{}, {},'cpu',progress,lambda:None)
    reader.close.assert_called_once()
