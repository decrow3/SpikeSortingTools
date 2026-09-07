"""One full Luke0804 imec0 development run using existing sorting/curation/QC."""
from pathlib import Path
import json
import os
import shutil
import time

from pipeline.downstream import pin_sort_identity, run_curation_stage, run_qc_stage
from pipeline.preprocess import validate_accepted_recording
from pipeline.runtime import validate_production_environment
from testing.ladder_sorter import RESCUE_RIGID, RESCUE, run_sorter_config, check_effective_settings, _json_safe

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0')
OUT = ROOT / 'testing/outputs/luke_full_session_rigid_v1'


def main():
    if (OUT / 'HOLD.json').exists():
        raise RuntimeError('Run is on user-requested hold; see HOLD.json. Do not restart without subsequent authorization.')
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/luke-full-session-matplotlib')
    environment = validate_production_environment(require_cuda=True)
    baseline = json.loads((BASE / 'kilosort4/rescue_sort_manifest.json').read_text())
    if baseline.get('complete') is not True:
        raise RuntimeError('baseline incomplete')
    expected = _json_safe(RESCUE.params())
    if baseline['sorter_params'] != expected:
        raise RuntimeError('baseline settings differ from current rescue defaults')
    manifest = json.loads((BASE / 'recording/rescue_recording_manifest.json').read_text())
    if baseline['recording_request_digest'] != manifest['request_digest']:
        raise RuntimeError('baseline and accepted recording differ')
    if manifest['num_samples'] != 314204894 or manifest['num_channels'] != 384:
        raise RuntimeError('unexpected full-session recording extent')
    if shutil.disk_usage(OUT).free < 40 * 1024**3:
        raise RuntimeError('less than 40 GiB output space available')
    request = dict(schema='luke-full-session-rigid-v1', development_only=True,
        recording=str(BASE / 'recording'), baseline=str(BASE),
        duration_s=manifest['num_samples']/manifest['sampling_frequency_hz'],
        recording_request_digest=manifest['request_digest'],
        recording_content_sha256=manifest['recording_content_sha256'],
        baseline_sort_request_digest=baseline['request_digest'],
        candidate_settings=_json_safe(RESCUE_RIGID.params()),
        changed_settings={'do_correction': True},
        scope='Full imec0 session, all 384 channels; development, including previously reserved intervals',
        comparison='Same curation and cached 1000-spike QC; matched common-time completeness, coverage, ISI and correspondence; no production promotion')
    path = OUT / 'request.json'
    if path.exists() and json.loads(path.read_text()) != request:
        raise RuntimeError('output belongs to another request')
    path.write_text(json.dumps(request, indent=2)+'\n')
    (OUT / 'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    def stage(name):
        print(name, flush=True)
        (OUT / 'status.json').write_text(json.dumps(dict(stage=name, updated_unix=time.time()))+'\n')
    stage('validating full accepted recording content')
    validate_accepted_recording(BASE / 'recording')
    stage('sorting full session with native rigid KS4')
    result = run_sorter_config(BASE / 'recording', OUT / 'kilosort4', RESCUE_RIGID)
    check_effective_settings('rescue_rigid', result)
    identity = pin_sort_identity(OUT / 'kilosort4', OUT / 'sort_identity.json')
    stage('curation')
    run_curation_stage(OUT / 'kilosort4/sorter_output', OUT / 'cur', identity)
    stage('waveform and amplitude QC')
    run_qc_stage(BASE / 'recording', OUT / 'cur/cur_output', OUT / 'qc', identity)
    stage('sort and QC complete; comparison pending')


if __name__ == '__main__':
    main()
