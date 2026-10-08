from __future__ import annotations

import hashlib, json, os, sys, threading, time
from pathlib import Path
import pytest
import en_first_medium_trained_pair_launch as pair
from voltage_identity_preflight import verify_voltage_identity

PACKET = Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resources(): return {"wall_seconds_max":2700,"ram_bytes_max":68719476736,"cpu_threads_max":16,"gpu_count_max":1,"persistent_output_bytes_max":17179869184,"minimum_free_bytes":214748364800,"service_restart":"no","service_start_limit":1,"output_monitor_interval_seconds":0.1}

def make_contract(tmp_path, enabled=True):
    bound=tmp_path/"bound"; bound.mkdir(); configs={}
    for name in ("REF384_repeat","repaired_B384","historical_screen"):
        p=bound/f"{name}.json"; p.write_text("{}\n"); configs[name]={"path":str(p),"sha256":sha(p)}
    roles=("pair_launcher","historical_ref_runner","trained_terminal","support_candidate","prewhitening","voltage_preflight","kilosort_io","kilosort_preprocessing","kilosort_spikedetect","kilosort_template_matching","kilosort_run")
    sources={}
    for role in roles:
        p=bound/f"{role}.py"; p.write_text(f"# {role}\n"); sources[role]={"path":str(p),"sha256":sha(p)}
    service=bound/"pair.service"; service.write_text("# reviewed service\n")
    paths={"root":str(tmp_path/"pair_root"),"numba_cache":str(tmp_path/"numba_cache"),"logs":str(tmp_path/"logs"),"staged_source_root":str(tmp_path/"stage"),"attempt_evidence":str(tmp_path/"attempt")}
    contract={"schema":pair.SCHEMA,"status":"enabled" if enabled else "disabled","execution_enabled":enabled,"approval":{"h1_review_manifest_sha256":"a"*64 if enabled else None},"arm_order":list(pair.ARM_ORDER),"configs":configs,"sources":sources,"service":{"unit_name":"pair.service","unit_binding":{"path":str(service),"sha256":sha(service)},"condition_on_start_claim_absence":True},"planned_paths":paths,"resources":resources(),"voltage_identity":{},"stop_conditions":["test"],"checkpoint_policy":"single consumed start; atomic arms","completion_condition":"both arms complete","gates":{"clock_review_verdict":"GO_CLOCK_REPAIR_ONLY","waveform_method_h1_verdict":"GO_METHOD_FREEZE_ONLY","rf_holdout_sealed":True}}
    path=tmp_path/"contract.json"; path.write_text(json.dumps(contract,sort_keys=True)+"\n")
    return path,contract,sha(service)

def pass_voltage(_): return {"status":"PASS","historic_and_current_match":True,"voltage_bytes_read_by_this_compact_preflight":0}
def write_complete(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text('{"status":"complete"}\n'); return {"status":"complete","complete_path":str(path)}

def test_r1_sequential_replay_consumes_start_before_preflight(tmp_path,monkeypatch):
    path,c,service_sha=make_contract(tmp_path); monkeypatch.setenv(pair.PERSISTENT_MARKER,"1"); probes=[]
    def fail(_): probes.append("preflight"); raise RuntimeError("injected GPU preflight failure")
    kw=dict(contract_path=path,expected_contract_sha256=sha(path),expected_service_sha256=service_sha,attempt_evidence_dir=c["planned_paths"]["attempt_evidence"],resource_probe=fail,voltage_probe=pass_voltage)
    with pytest.raises(RuntimeError,match="injected GPU"): pair.run_single_attempt(**kw)
    attempt=Path(c["planned_paths"]["attempt_evidence"])
    assert (attempt/"START_CLAIM.json").is_file() and (attempt/"STARTED.json").is_file()
    assert json.loads((attempt/"FAILURE.json").read_text())["stage"]=="gpu_and_free_space_preflight"
    assert not Path(c["planned_paths"]["root"]).exists()
    with pytest.raises(PermissionError,match="second attempt rejected before preflight"): pair.run_single_attempt(**kw)
    assert probes==["preflight"]

def test_r1_concurrent_replay_allows_one_preflight(tmp_path,monkeypatch):
    path,c,service_sha=make_contract(tmp_path); monkeypatch.setenv(pair.PERSISTENT_MARKER,"1"); barrier=threading.Barrier(2); lock=threading.Lock(); probes=[]; results=[]
    def fail(_):
        with lock: probes.append("preflight")
        time.sleep(.05); raise RuntimeError("injected")
    def worker():
        barrier.wait()
        try: pair.run_single_attempt(contract_path=path,expected_contract_sha256=sha(path),expected_service_sha256=service_sha,attempt_evidence_dir=c["planned_paths"]["attempt_evidence"],resource_probe=fail,voltage_probe=pass_voltage)
        except BaseException as exc:
            with lock: results.append(type(exc).__name__)
    ts=[threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert probes==["preflight"] and sorted(results)==["PermissionError","RuntimeError"]

def claimed_contract(tmp_path,monkeypatch,cap):
    path,_,service_sha=make_contract(tmp_path); c=pair.validate_contract(path,sha(path)); c["resources"]=dict(c["resources"]); c["resources"]["persistent_output_bytes_max"]=cap; monkeypatch.setenv(pair.PERSISTENT_MARKER,"1"); pair.consume_start_claim(c["planned_paths"]["attempt_evidence"],contract_path=path,expected_contract_sha256=sha(path),expected_service_sha256=service_sha); return c

def test_r2_positive_aggregate_cap(tmp_path,monkeypatch):
    c=claimed_contract(tmp_path,monkeypatch,100000); root=Path(c["planned_paths"]["root"])
    def ref(_): (root/"REF384_repeat").mkdir(parents=True); (root/"REF384_repeat/small.bin").write_bytes(b"x"*100); return write_complete(tmp_path/"ref.complete")
    result=pair.execute_pair(c,ref_executor=ref,repaired_executor=lambda _:write_complete(tmp_path/"b.complete"),resource_probe=lambda _:None,voltage_probe=pass_voltage)
    assert result["aggregate_persistent_output_bytes"]<=100000

def test_r2_cap_crossing_stops_before_b(tmp_path,monkeypatch):
    c=claimed_contract(tmp_path,monkeypatch,2000); root=Path(c["planned_paths"]["root"]); called=[]
    def ref(_): called.append("REF"); (root/"REF384_repeat").mkdir(parents=True); (root/"REF384_repeat/over.bin").write_bytes(b"x"*4000); return write_complete(tmp_path/"ref.complete")
    with pytest.raises(pair.OutputCapExceeded,match="cap exceeded"): pair.execute_pair(c,ref_executor=ref,repaired_executor=lambda _:called.append("B"),resource_probe=lambda _:None,voltage_probe=pass_voltage)
    status=json.loads((Path(c["planned_paths"]["attempt_evidence"])/"OUTPUT_CAP_STATUS.json").read_text())
    assert status["within_limit"] is False and called==["REF"] and not (root/"COMPLETE.json").exists()

def test_r2_runtime_monitor_terminates_crossing_child(tmp_path):
    path,_,service_sha=make_contract(tmp_path); c=pair.validate_contract(path,sha(path)); c["resources"]=dict(c["resources"]); c["resources"].update(persistent_output_bytes_max=20000,output_monitor_interval_seconds=.01); attempt=Path(c["planned_paths"]["attempt_evidence"]); pair.consume_start_claim(attempt,contract_path=path,expected_contract_sha256=sha(path),expected_service_sha256=service_sha); root=Path(c["planned_paths"]["root"]); root.mkdir()
    code="import pathlib,time;p=pathlib.Path(r'"+str(root/"growing.bin")+"');f=p.open('wb');[(f.write(b'x'*4096),f.flush(),time.sleep(.01)) for _ in range(20)];f.close()"
    with pytest.raises(pair.OutputCapExceeded,match="reached during fixture"): pair.run_monitored_command([sys.executable,"-c",code],contract=c,attempt_dir=attempt,stage="fixture")
    assert json.loads((attempt/"OUTPUT_CAP_FAILURE.json").read_text())["process_group_terminated"] is True

def test_r2_aggregate_crossing_in_second_arm_preserves_ref_state(tmp_path,monkeypatch):
    c=claimed_contract(tmp_path,monkeypatch,12000); root=Path(c["planned_paths"]["root"])
    def ref(_): (root/"REF384_repeat").mkdir(parents=True); (root/"REF384_repeat/ref.bin").write_bytes(b"r"*3000); return write_complete(tmp_path/"ref.complete")
    def repaired(_): (root/"repaired_B384").mkdir(); (root/"repaired_B384/b.bin").write_bytes(b"b"*10000); return write_complete(tmp_path/"b.complete")
    with pytest.raises(pair.OutputCapExceeded,match="cap exceeded"): pair.execute_pair(c,ref_executor=ref,repaired_executor=repaired,resource_probe=lambda _:None,voltage_probe=pass_voltage)
    assert (root/"launch_evidence/REF384_repeat_EXECUTION_STATE.json").is_file()
    assert not (root/"COMPLETE.json").exists()

def stat_tuple(path):
    v=os.lstat(path); return {k:int(getattr(v,k)) for k in ("st_dev","st_ino","st_mode","st_size","st_mtime_ns","st_ctime_ns")}

def make_voltage_packet(tmp_path):
    tmp_path.mkdir(parents=True,exist_ok=True); binary=tmp_path/"voltage.raw"; binary.write_bytes(b"0123456789abcdef"); digest=sha(binary); os.chmod(binary,0); recording=tmp_path/"recording.json"; recording.write_text(json.dumps({"schema_version":"rescue-recording-manifest-v2","num_samples":8,"num_channels":1,"sampling_frequency_hz":30000.0,"dtype":"int16","selected_start_frame":0,"selected_end_frame":8,"external_voltage_motion_correction":False,"physical_channel_ids":["imec0.ap#AP0"]}))
    historic=tmp_path/"historic"; historic.mkdir(); old=historic/"Q0_FULL_HASH_RECEIPT.json"; old.write_text(json.dumps({"status":"pass","binary_path":str(binary),"bytes_read":16,"expected_sha256":digest,"actual_sha256":digest})); om=historic/"MANIFEST.json"; om.write_text(json.dumps({"files":{old.name:{"bytes":old.stat().st_size,"sha256":sha(old)}}})); oc=historic/"COMPLETE.json"; oc.write_text(json.dumps({"status":"complete","manifest_sha256":sha(om)}))
    current=tmp_path/"current"; current.mkdir(); source_sha="7"*64; cr=current/"CURRENT_VOLTAGE_FULL_HASH_RECEIPT.json"; cr.write_text(json.dumps({"status":"PASS_EXACT_MATCH","binary_path":str(binary),"binary_is_symlink":False,"bytes_read":16,"expected_sha256":digest,"actual_sha256":digest,"stat_unchanged":True,"source_sha256":source_sha,"final_lstat":stat_tuple(binary)})); started=current/"STARTED.json"; started.write_text("{}"); progress=current/"PROGRESS.jsonl"; progress.write_text("{}\n"); cm=current/"MANIFEST.sha256"; cm.write_text("".join(f"{sha(p)}  {p.name}\n" for p in (cr,progress,started))); cc=current/"COMPLETE.json"; cc.write_text(json.dumps({"status":"PASS_EXACT_MATCH","manifest_sha256":sha(cm),"actual_sha256":digest,"bytes_read":16}))
    binding={"binary_path":str(binary),"binary_size_bytes":16,"binary_sha256":digest,"recording_manifest_path":str(recording),"recording_manifest_sha256":sha(recording),"num_samples":8,"num_channels":1,"sampling_frequency_hz":30000.0,"immutable_full_hash_receipt":{"packet_path":str(historic),"packet_manifest_sha256":sha(om),"packet_complete_sha256":sha(oc),"receipt_sha256":sha(old)},"current_full_hash_receipt":{"packet_path":str(current),"packet_manifest_sha256":sha(cm),"packet_complete_sha256":sha(cc),"receipt_sha256":sha(cr),"verification_source_sha256":source_sha,"final_lstat":stat_tuple(binary)}}
    return binding,binary

def test_r3_compact_preflight_reads_zero_voltage(tmp_path):
    binding,binary=make_voltage_packet(tmp_path); result=verify_voltage_identity(binding); assert result["status"]=="PASS" and result["voltage_bytes_read_by_this_compact_preflight"]==0

def test_r3_same_size_replacement_and_symlink_fail(tmp_path):
    binding,binary=make_voltage_packet(tmp_path/"replace"); original=os.lstat(binary); replacement=binary.with_name("replacement.raw"); replacement.write_bytes(b"fedcba9876543210"); os.utime(replacement,ns=(original.st_atime_ns,original.st_mtime_ns)); os.replace(replacement,binary)
    with pytest.raises(RuntimeError,match="stat identity changed"): verify_voltage_identity(binding)
    binding,binary=make_voltage_packet(tmp_path/"symlink"); target=binary.with_name("target.raw"); binary.rename(target); binary.symlink_to(target)
    with pytest.raises(RuntimeError,match="non-symlink regular file"): verify_voltage_identity(binding)

def test_r3_voltage_preflight_precedes_ref(tmp_path,monkeypatch):
    c=claimed_contract(tmp_path,monkeypatch,100000); order=[]
    def voltage(_): order.append("voltage"); return pass_voltage(None)
    def ref(_): order.append("REF"); return write_complete(tmp_path/"ref.complete")
    pair.execute_pair(c,ref_executor=ref,repaired_executor=lambda _:write_complete(tmp_path/"b.complete"),resource_probe=lambda _:None,voltage_probe=voltage); assert order==["voltage","REF"]

def test_actual_repair_packet_validate_only_and_preserves_entries():
    path=PACKET/"contract/en_first_medium_pair.execution_disabled.r1_r3.v2.json"; c=pair.validate_contract(path,sha(path)); ref=json.loads((PACKET/"config/REF384_repeat.execution_disabled.frozen.v2.json").read_text()); repaired=json.loads((PACKET/"config/repaired_B384.execution_disabled.clock_v5.json").read_text()); assert c["execution_enabled"] is False and c["approval"]["h1_review_manifest_sha256"] is None; assert c["service"]["enabled_binding_intentionally_absent"] is True; assert ref["effective_requirements"]["support_policy_enabled"] is False; assert ref["output_clock"]==repaired["output_clock"]; assert repaired["trained"]["allow_saved_smoke_bank"] is False; assert c["gates"]["blocking_h1_review_manifest_sha256"]=="9842cca05ee6762aa5af26138b04e2edd792bf2f1504b19485870747837e1381"
