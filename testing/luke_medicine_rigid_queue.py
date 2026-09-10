"""Persistent dependency controller and development-only external-rigid KS4 run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from testing.luke_full_session_medicine import save, sha, sealed


def service_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState',
                             '-p', 'SubState', '-p', 'MainPID', '-p', 'LoadState', '-p', 'Result'],
                            capture_output=True, text=True)
    value = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not value:
        raise RuntimeError(f'Cannot inspect dependency service: {result.stderr}')
    value['process_exists'] = int(value.get('MainPID', '0')) > 0 and Path('/proc', value['MainPID']).exists()
    return value


def dependency_decision(state, receipt, service_result):
    """No stale success file can release a still-running or failed producer."""
    if receipt and (receipt.get('state') in ['failed', 'wrapper_error'] or
                    receipt.get('returncode') not in [None, 0]):
        raise RuntimeError('Motion-estimation process failed; rigid queue stopped')
    if service_result and (service_result.get('SERVICE_RESULT') != 'success' or
                           service_result.get('EXIT_STATUS') != '0'):
        raise RuntimeError('Motion-estimation service failed/cancelled; rigid queue stopped')
    if state.get('ActiveState') in ['active', 'activating', 'deactivating', 'reloading'] or state.get('process_exists'):
        return 'wait'
    if state.get('ActiveState') == 'failed':
        raise RuntimeError('Motion-estimation service failed')
    if (receipt and receipt.get('state') == 'complete' and receipt.get('returncode') == 0
            and service_result and service_result.get('SERVICE_RESULT') == 'success'
            and service_result.get('EXIT_STATUS') == '0'):
        return 'ready'
    raise RuntimeError('Motion-estimation service stopped without successful terminal receipts')


def read_optional(path):
    return json.loads(path.read_text()) if path.exists() else None


def assert_no_hold(cfg):
    for path in cfg['hold_paths']:
        if Path(path).exists():
            raise RuntimeError(f'New rigid run cancelled/on hold: {path}')


def validate_field(z, fs, samples, boundary_limit_s):
    t = np.asarray(z['time_s'], dtype=float)
    y = np.asarray(z['depth_um'], dtype=float)
    rigid = np.asarray(z['rigid_displacement_um'], dtype=float)
    nonrigid = np.asarray(z['nonrigid_displacement_um'], dtype=float)
    if (t.ndim != 1 or y.ndim != 1 or rigid.shape != t.shape or
            nonrigid.shape != (len(t), len(y)) or len(t) < 2 or len(y) < 2):
        raise ValueError('Invalid candidate field shapes')
    if not all(np.isfinite(a).all() for a in [t, y, rigid, nonrigid]):
        raise ValueError('Nonfinite motion field')
    if not np.allclose(np.diff(t), .25, rtol=0, atol=1e-6) or not np.all(np.diff(y) > 0):
        raise ValueError('Unexpected field sampling or depth order')
    duration = samples/fs
    # Native support can extend epsilon before the first detected event.
    if t[0] < -.0011 or t[-1] > duration or max(t[0], duration-t[-1]) > boundary_limit_s:
        raise ValueError('Field does not cover recording except permitted terminal margins')
    if max(np.max(np.abs(rigid)), np.max(np.abs(nonrigid))) > 500.001:
        raise ValueError('Field exceeds documented centered displacement envelope')
    return t, y, rigid, nonrigid


def common_channel_indices(geometry, depths, rigid, nonrigid, margin):
    """Conservative fixed channel set valid for both correction arms at all times."""
    lo = float(min(rigid.min(), nonrigid.min()))
    hi = float(max(rigid.max(), nonrigid.max()))
    yy = geometry[:, 1]
    keep = ((yy >= max(depths[0], yy.min()+margin-lo)) &
            (yy <= min(depths[-1], yy.max()-margin-hi)))
    indices = np.flatnonzero(keep)
    if len(indices) < 64:
        raise ValueError('Fewer than 64 supported common channels; refuse narrowed sort')
    return indices


def apply_rigid(recording, times, displacement, policy):
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
    if recording.get_num_segments() != 1:
        raise ValueError('Expected one segment')
    origin = float(recording.sample_index_to_time(0))
    # Field is recording-relative; SI trace interpolation queries acquisition time.
    motion = Motion(displacement=[np.asarray(displacement)[:, None]],
                    temporal_bins_s=[np.asarray(times)+origin],
                    spatial_bins_um=np.array([np.mean(recording.get_channel_locations()[:, 1])]),
                    direction='y', interpolation_method='linear')
    return InterpolateMotionRecording(astype(recording, 'float32'), motion,
        border_mode='remove_channels', spatial_interpolation_method='kriging',
        sigma_um=policy['sigma_um'], p=policy['p'], dtype='float32')


def checked_int16(recording):
    """Reject nonfinite/overflow samples rather than wrap during materialization."""
    from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment

    class Segment(BasePreprocessorSegment):
        def get_traces(self, start_frame, end_frame, channel_indices):
            x = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite corrected voltage')
            rounded = np.rint(x)
            if np.any(rounded < -32768) or np.any(rounded > 32767):
                raise ValueError('Corrected voltage would overflow int16; preserve partial')
            return rounded.astype('int16')

    class Recording(BasePreprocessor):
        def __init__(self, source):
            BasePreprocessor.__init__(self, source, dtype='int16')
            for segment in source._recording_segments:
                self.add_recording_segment(Segment(segment))
            self._kwargs = {'source': source}
            self._serializability['json'] = False

    return Recording(recording)


def verify_inputs(cfg):
    for name, digest in cfg['pinned_inputs'].items():
        if sha(name) != digest:
            raise RuntimeError(f'Pinned dependency/input changed: {name}')
    for name, digest in cfg['source_sha256'].items():
        if sha(Path(__file__).resolve().parents[1]/name) != digest:
            raise RuntimeError(f'Frozen queue source changed: {name}')


def execute(cfg, stage):
    from spikeinterface.core import load
    from pipeline.config import fingerprint
    from pipeline.preprocess import validate_accepted_recording
    from pipeline.runtime import validate_production_environment
    from pipeline.sorting import build_kilosort4_params, _json_safe_params, run_kilosort4
    from pipeline.downstream import pin_sort_identity, run_curation_stage, run_qc_stage, run_matlab_export_stage
    from testing.luke_external_warp_pipeline import _materialize_arm

    out, estimation = Path(cfg['output']), Path(cfg['estimation_output'])
    assert_no_hold(cfg); verify_inputs(cfg)
    save(out/'environment.json', validate_production_environment(require_cuda=True))
    summary = json.loads((estimation/'summary.json').read_text())
    source_verified = json.loads((estimation/'source_verified.json').read_text())
    if summary.get('status') != 'complete' or source_verified['recording_content_sha256'] != cfg['recording_content_sha256']:
        raise RuntimeError('Estimation/source completion mismatch')
    sealed(estimation/'fit')
    shared = Path(cfg['shared'])
    field_manifest = json.loads((shared/'field_manifest.json').read_text())
    if (field_manifest.get('status') != 'estimation_artifacts_complete' or
            field_manifest.get('recording_content_sha256') != cfg['recording_content_sha256']):
        raise RuntimeError('Wrong shared field package')
    for name, digest in field_manifest['files'].items():
        if sha(shared/field_manifest['field_package']/name) != digest:
            raise RuntimeError(f'Shared field artifact changed: {name}')
    field_path = estimation/'candidate_fields.npz'
    if sha(field_path) != field_manifest['files']['candidate_fields.npz']:
        raise RuntimeError('Local and shared correction fields differ')

    stage('validate_accepted_source')
    source_dir = Path(cfg['recording'])
    manifest = validate_accepted_recording(source_dir)
    if manifest['recording_content_sha256'] != cfg['recording_content_sha256']:
        raise RuntimeError('Incorrect source recording')
    baseline = json.loads(Path(cfg['baseline_manifest']).read_text())
    params = _json_safe_params(build_kilosort4_params())
    if baseline['sorter_params'] != params or params['do_correction'] is not False:
        raise RuntimeError('Frozen baseline sorter settings differ')
    source = load(source_dir)
    total, fs = source.get_num_samples(), source.get_sampling_frequency()
    if (total, source.get_num_channels()) != (314204894, 384):
        raise RuntimeError('Not the requested full source geometry/duration')
    t, y, rigid, nonrigid = validate_field(np.load(field_path), fs, total, cfg['terminal_margin_limit_s'])
    geometry = source.get_channel_locations()
    indices = common_channel_indices(geometry, y, rigid, nonrigid, cfg['application']['support_margin_um'])
    ids = source.channel_ids[indices]
    origin = float(source.sample_index_to_time(0))
    if abs(float(source.sample_index_to_time(total))-origin-total/fs) > 1e-6:
        raise RuntimeError('Nonuniform recording time mapping')
    contract = dict(schema='luke-improved-medicine-rigid-application-v1',
        user_authorization=cfg['authorization'], development_only=True,
        scientific_status='unqualified_development_comparison',
        estimator_status_preserved=field_manifest['scientific_status'],
        field_sha256=sha(field_path), field_manifest_sha256=sha(shared/'field_manifest.json'),
        source_content_sha256=manifest['recording_content_sha256'],
        source_request_digest=manifest['request_digest'],
        duration_s=total/fs, num_samples=total, acquisition_time_origin_s=origin,
        native_field_recording_support_s=[float(t[0]), float(t[-1])],
        terminal_nearest_value_extension_s=[max(0.,float(t[0])), max(0.,total/fs-float(t[-1]))],
        application=cfg['application'], rigid_gain=1.,
        projection='Use saved seed-referenced equal-depth rigid projection, no new estimation',
        channel_ids=[str(i) for i in ids], channel_indices=indices.tolist(),
        common_with_nonrigid=True, source_channels=384, output_channels=len(ids),
        source_depth_span_um=[float(geometry[:,1].min()), float(geometry[:,1].max())],
        output_depth_span_um=[float(geometry[indices,1].min()), float(geometry[indices,1].max())],
        baseline_comparability='Existing full-probe baseline reused; correction candidates use common supported interior; channel restriction is a comparison limitation',
        sorter_params=params, baseline_manifest_sha256=sha(cfg['baseline_manifest']))
    contract['digest'] = fingerprint(contract)
    save(out/'application_contract.json', contract)
    contract_path = shared/'rigid_application_contract_v1.json'
    if contract_path.exists():
        raise RuntimeError('Shared application contract already exists; preserve it')
    save(contract_path, contract)
    corrected = apply_rigid(source, t, rigid, cfg['application'])
    if not np.isin(ids, corrected.channel_ids).all():
        raise RuntimeError('Operator removed a channel from common domain')
    corrected = corrected.channel_slice(channel_ids=ids)
    if corrected.get_num_samples() != total or not np.array_equal(corrected.get_channel_locations(), geometry[indices]):
        raise RuntimeError('Correction changed time extent or channel identity')
    corrected = checked_int16(corrected)
    # Direct, cheap operator/read checks on endpoints and actual extremal motion.
    stage('check_corrected_voltage')
    centers = sorted(set([0, total-3000, int(t[np.argmax(rigid)]*fs), int(t[np.argmin(rigid)]*fs), round(935*fs)]))
    rows = []
    for center in centers:
        lo = max(0, min(total-3000, center)); hi = min(total, lo+3000)
        traces = corrected.get_traces(start_frame=lo, end_frame=hi)
        rows.append(dict(start_sample=lo, stop_sample=hi, min_counts=int(traces.min()),
                         max_counts=int(traces.max()), finite=True))
    save(out/'voltage_smoke.json', rows)
    required = total*len(ids)*2 + cfg['sort_scratch_reserve_bytes']
    if shutil.disk_usage(out).free < required:
        raise RuntimeError(f'Insufficient local disk: need corrected binary plus scratch, {required} bytes')
    stage('materialize_rigid_corrected_voltage')
    assert_no_hold(cfg)
    request = dict(schema='luke-improved-medicine-rigid-recording-v1',
                   application_contract_digest=contract['digest'], source_content_sha256=manifest['recording_content_sha256'],
                   external_correction=True, field_sha256=contract['field_sha256'],
                   production_environment=cfg['production_environment'])
    _materialize_arm(corrected, out/'recording', source_manifest=manifest, request=request, n_jobs=1)
    del corrected, source
    if shutil.disk_usage(out).free < cfg['sort_scratch_reserve_bytes']:
        raise RuntimeError('Insufficient sorter scratch after materialization')
    assert_no_hold(cfg); verify_inputs(cfg)
    stage('kilosort4_external_rigid_internal_motion_off')
    sort_manifest = run_kilosort4(out/'recording', out/'kilosort4')
    ops = np.load(out/'kilosort4/sorter_output/ops.npy', allow_pickle=True).item()
    if int(ops['nblocks']) != 0 or ops['dshift'] is not None:
        raise RuntimeError('Unexpected internal motion correction')
    if int(ops['n_chan_bin']) != len(ids) or ops['fs'] != fs:
        raise RuntimeError('Incorrect effective sorter input')
    identity = pin_sort_identity(out/'kilosort4', out/'sort_identity.json')
    assert_no_hold(cfg); stage('curation')
    run_curation_stage(out/'kilosort4/sorter_output', out/'cur', identity)
    assert_no_hold(cfg); stage('qc')
    run_qc_stage(out/'recording', out/'cur/cur_output', out/'qc', identity)
    assert_no_hold(cfg); stage('matlab_export')
    run_matlab_export_stage(out/'cur/cur_output', out/'qc', identity)
    save(out/'summary.json', dict(status='complete', development_only=True,
         application_contract_digest=contract['digest'], sort_identity_digest=identity['identity_digest'],
         sort_summary=sort_manifest['summary'], num_samples=total, output_channels=len(ids),
         scientific_status='comparison_pending', correction='external rigid shared MEDiCINe',
         internal_motion_correction=False))
    save(shared/'huklaban1_rigid_result_v1.json', json.loads((out/'summary.json').read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--dummy', action='store_true')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    save(out/'queue_request.json', cfg)

    def stage(name, **kwargs):
        row = dict(stage=name, updated_unix=time.time(), pid=os.getpid(), **kwargs)
        save(out/'status.json', row)
        print(json.dumps(row), flush=True)

    if args.dummy:
        stage('dummy_waiting'); time.sleep(20); stage('dummy_complete')
        save(out/'summary.json', dict(status='dummy_complete'))
        return
    verify_inputs(cfg)
    dep = Path(cfg['estimation_job'])
    stage('queued_waiting_for_medicine')
    while True:
        assert_no_hold(cfg)
        state = service_state(cfg['estimation_unit'])
        decision = dependency_decision(state, read_optional(dep/'receipt.json'), read_optional(dep/'service_result.json'))
        save(out/'dependency_state.json', dict(checked_unix=time.time(), decision=decision, systemd=state))
        if decision == 'ready':
            break
        time.sleep(30)
    execute(cfg, stage)
    stage('complete')


if __name__ == '__main__':
    main()
