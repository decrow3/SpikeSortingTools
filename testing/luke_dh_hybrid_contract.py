#!/usr/bin/env python3
"""Freeze DH's concrete 25-donor hybrid contract and execute compact fixtures."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from testing.luke_au_cpu_preparation import exact_remap, exact_same_column_map, seeded_independent_train
from testing.luke_dh_exact_chunk_motion import ExactChunkLatticeMotion
from testing.luke_dh_hybrid_scorer import score_with_frozen_association_v3

ROOT = Path(__file__).resolve().parents[1]
CX = ROOT / "testing/outputs/cx_masked_donor_qualification_v1/run"
BC = ROOT / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/bc_attempt5"
FIELD = Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage4_ab/luke0804_imec1_two_layer_motion.npz")
ACTUAL_FIELDS = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/motion/fields.npz")
STATIC = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1/static_w2")
D2L = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L")
DART_PY = Path("/home/huklab/anaconda3/envs/spike-sort-challengers/bin/python")
FS = 29999.759166666667
START = 26_999_783
END = 37_199_701
CHUNK = 7500
TARGET = np.arange(202, 384, dtype=np.int64)
EXPECTED_FIELD = "85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def array_hash(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(f"{a.dtype.str}|{a.shape}".encode() + a.tobytes()).hexdigest()


def half_away(x):
    x = np.asarray(x, float); return np.sign(x) * np.floor(np.abs(x) + .5)


def cosine(a, b):
    a=np.asarray(a,float).ravel(); b=np.asarray(b,float).ravel(); a-=a.mean(); b-=b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b); return float(a@b/den) if den else np.nan


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    started=time.monotonic(); cpu0=time.process_time()
    if args.output.exists(): raise FileExistsError(args.output)
    if os.statvfs(args.output.parent).f_bavail*os.statvfs(args.output.parent).f_frsize < 30_000_000_000:
        raise RuntimeError("30 GB free-space gate")
    args.output.mkdir(parents=True)
    (args.output/"START_RECEIPT.json").write_text(json.dumps({
        "status":"running","started_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"one concrete hybrid manifest plus saved-array injection/scorer fixtures",
        "eta_minutes":18,"cpu_allwork_cap_s":900,"threads_max":2,"readers_max":1,
        "saved_donor_read_cap_bytes":268435456,"scratch_cap_bytes":1_000_000_000,
        "final_cap_bytes":500_000_000,"free_space_gate_bytes":30_000_000_000,
        "raw_voltage":False,"gpu":False,"sort":False,
    },indent=2,sort_keys=True)+"\n")
    if sha(FIELD)!=EXPECTED_FIELD: raise RuntimeError("field hash mismatch")
    qdf=pd.read_csv(CX/"ROBUST_EXPLORATORY_MODIFIED_SET.csv")
    donor_ids=qdf.unit_id.to_numpy(np.int64)
    expected=np.array([13,30,139,143,184,275,276,291,296,325,332,372,407,415,425,433,438,445,451,461,465,480,493,499,512])
    if not np.array_equal(donor_ids,expected): raise RuntimeError("25-donor intersection changed")
    with np.load(BC/"full_probe_final_label_templates.npz",allow_pickle=False) as z:
        all_templates=np.asarray(z["templates"],np.float32); all_ids=np.asarray(z["unit_ids"],np.int64)
        channels=np.asarray(z["channel_ids"]); geom=np.asarray(z["geometry_um"],float)
        trough=int(z["trough_offset_samples"]); domain=str(z["measurement_domain"])
    expected_channels=np.array([f"imec1.ap#AP{k}" for k in range(202,384)])
    if not np.array_equal(channels[TARGET],expected_channels): raise RuntimeError("physical AP202:383 target mismatch")
    with np.load(CX/"DONOR_COORDINATE_TAPERS.npz",allow_pickle=False) as z:
        if not np.array_equal(all_ids,z["unit_ids"]) or not np.array_equal(geom,z["geometry_um"]):
            raise RuntimeError("CX/BC lineage mismatch")
        weights=np.asarray(z["train_weights"],np.float32)
    lookup={int(u):i for i,u in enumerate(all_ids)}; ix=np.array([lookup[int(u)] for u in donor_ids])
    modified=all_templates[ix]*weights[ix,None,:]
    bases=qdf.qualification_chosen_base_shift_um.to_numpy(float)
    if not np.all(np.isfinite(bases)) or not np.allclose(bases/40,np.rint(bases/40),rtol=0,atol=1e-12):
        raise RuntimeError("placement bases are not exact 40-um lattice values")

    with np.load(FIELD,allow_pickle=False) as z:
        ft=np.asarray(z["time_s"],float); fd=np.asarray(z["displacement_um"],float)[:,0]
        if str(z["sign_convention"])!="corrected = observed - displacement": raise RuntimeError("field sign")
    starts=np.arange(0,END-START,CHUNK,dtype=np.int64); centers=starts+CHUNK//2
    center_local_s=centers/FS; center_session_s=(START+centers)/FS
    with np.load(ACTUAL_FIELDS,allow_pickle=False) as z:
        actual_t=np.asarray(z["time_bin_centers_s"],float)
        actual_d=np.asarray(z["displacement"],float)
    if actual_d.shape[0]!=1 or not np.all(np.diff(actual_t)>0): raise RuntimeError("actual D2L field shape")
    sampled=np.interp(center_local_s,actual_t,actual_d[0])
    source_sampled=np.interp(center_session_s,ft,fd)
    if not np.allclose(sampled,source_sampled,rtol=0,atol=1e-10): raise RuntimeError("actual consumer/source query mismatch")
    states=(40*half_away(sampled/40)).astype(np.int16)
    exact_motion=ExactChunkLatticeMotion(states,sampling_frequency_hz=FS,
                                         chunk_length_samples=CHUNK,n_samples=END-START)
    if not np.array_equal(np.asarray(exact_motion.disp_at_s(center_local_s),np.int16),states):
        raise RuntimeError("exact-query adapter mismatch")
    np.savez_compressed(args.output/"EXACT_QUERY_MOTION.npz",states_um=states,
                        chunk_start_samples=starts,chunk_center_samples=centers,
                        chunk_center_local_s=center_local_s,source_unrounded_um=sampled,
                        sampling_frequency_hz=np.float64(FS),chunk_length_samples=np.int64(CHUNK),
                        n_samples=np.int64(END-START))
    env=dict(os.environ); env["NUMBA_CACHE_DIR"]="/tmp/dh_numba_cache"
    subprocess.run([str(DART_PY),"-m","testing.dh_actual_consumer_fixture",str(args.output.resolve()),
                    "--output",str(args.output/"ACTUAL_DARTSORT_CONSUMER_FIXTURE.json")],
                   cwd=ROOT,env=env,check=True)
    state_by_chunk={int(i):int(q) for i,q in enumerate(states)}
    occupied=sorted(set(map(int,states)))
    if occupied != [-240,-200,-160,-120,-80,-40,0]: raise RuntimeError(f"occupied states changed: {occupied}")

    donor_rows=[]; event_rows=[]; member_rows=[]; all_intervals=[]; support_rows=[]
    map_hashes={}
    for donor, template, base in zip(donor_ids,modified,bases):
        template_hash=array_hash(template); train=seeded_independent_train(int(donor),fs_hz=FS,start_s=START/FS,end_s=END/FS)
        if np.any(train<0) or np.any(train>=END-START): raise RuntimeError("train clock")
        donor_rows.append({"donor_id":int(donor),"source_unit_id":int(donor),"template_sha256":template_hash,
                           "taper_row_unit_id":int(donor),"placement_base_shift_um":float(base),
                           "occupied_states_um":";".join(map(str,occupied)),"source_train_sha256":array_hash(train),
                           "source_event_count":int(train.size),"selected":True})
        source64=template.astype(np.float64); source_energy=float(np.square(source64).sum())
        source_sum=float(source64.sum()); nvalue=template.size
        source_norm2=source_energy-source_sum*source_sum/nvalue
        source_ptp=float(np.ptp(template,axis=0).max())
        target_mask=np.zeros(len(geom),bool); target_mask[TARGET]=True
        for state in occupied:
            total=float(base+state); mapping=exact_same_column_map(geom,total)
            keep=(mapping>=0)&target_mask[np.maximum(mapping,0)]
            kept=template[:,keep].astype(np.float64,copy=False)
            kept_energy=float(np.square(kept).sum()); kept_sum=float(kept.sum())
            restored_norm2=kept_energy-kept_sum*kept_sum/nvalue
            dot=kept_energy-source_sum*kept_sum/nvalue
            rtcos=dot/np.sqrt(source_norm2*restored_norm2) if source_norm2>0 and restored_norm2>0 else np.nan
            rtptp=(float(np.ptp(template[:,keep],axis=0).max())/source_ptp) if keep.any() and source_ptp else np.nan
            support_rows.append({"donor_id":int(donor),"state_um":state,
                                 "placement_base_shift_um":float(base),"total_relocation_um":total,
                                 "retained_source_channels":int(keep.sum()),
                                 "retained_energy_fraction":kept_energy/source_energy,
                                 "roundtrip_centered_cosine":rtcos,"roundtrip_ptp_ratio":rtptp,
                                 "passes_frozen_support_gate":bool(kept_energy/source_energy>=.99 and rtcos>=.99 and .98<=rtptp<=1.02)})
        for k,sample in enumerate(train):
            chunk=min(int(sample)//CHUNK,len(states)-1); state=state_by_chunk[chunk]; total=float(base+state)
            key=(int(donor),total)
            if key not in map_hashes:
                mapping=exact_same_column_map(geom,total); map_hashes[key]=array_hash(mapping)
                moved=exact_remap(template,mapping)[:,TARGET]
                if not np.isfinite(moved).all() or not np.any(moved): raise RuntimeError("invalid relocated template")
            lo=int(sample-trough); hi=lo+template.shape[0]
            eid=f"u{int(donor)}:{k:04d}"
            event_rows.append({"event_row_id":eid,"donor_id":int(donor),"local_sample":int(sample),
                               "absolute_sample":int(START+sample),"chunk_index":chunk,
                               "chunk_start_local_sample":int(starts[chunk]),"chunk_stop_local_sample":int(min(starts[chunk]+CHUNK,END-START)),
                               "state_um":state,"placement_base_shift_um":float(base),"total_relocation_um":total})
            member_rows.append({"injection_event_id":"inj:"+eid,"source_event_row_id":eid,"donor_id":int(donor),
                                "arm_source":"shared_before_S_h_D2L_h_branch","sample_start":lo,"sample_stop":hi,
                                "template_sha256":template_hash,"channel_map_sha256":map_hashes[key],"status":"injected"})
            all_intervals.append((lo,hi,eid))
    # Collision groups are overlap-connected components of half-open waveform writes.
    order=sorted(all_intervals); group=0; active_stop=-1; groups={}
    for lo,hi,eid in order:
        if lo>=active_stop: group+=1; active_stop=hi
        else: active_stop=max(active_stop,hi)
        groups[eid]=group
    for row in member_rows: row["collision_group"]=groups[row["source_event_row_id"]]
    donors=pd.DataFrame(donor_rows); events=pd.DataFrame(event_rows); members=pd.DataFrame(member_rows)
    donors.to_csv(args.output/"DONORS.csv",index=False); events.to_csv(args.output/"SOURCE_EVENTS.csv",index=False)
    members.to_csv(args.output/"INJECTION_MEMBERSHIP.csv",index=False)
    support=pd.DataFrame(support_rows); support.to_csv(args.output/"DONOR_STATE_SUPPORT.csv",index=False)

    check_samples={0,END-START-1}
    for k in np.flatnonzero(np.diff(states)!=0)+1:
        boundary=int(starts[k]); check_samples.update({boundary-1,boundary,boundary+1,int(centers[k])})
    boundary_rows=[]
    for sample in sorted(x for x in check_samples if 0<=x<END-START):
        expected_state=int(states[min(sample//CHUNK,len(states)-1)])
        got=float(exact_motion.disp_at_s(np.array([sample/FS]))[0])
        boundary_rows.append({"local_sample":sample,"local_time_s":sample/FS,
                              "expected_chunk":min(sample//CHUNK,len(states)-1),
                              "expected_state_um":expected_state,"queried_state_um":got,
                              "passed":got==expected_state})
    pd.DataFrame(boundary_rows).to_csv(args.output/"QUERY_BOUNDARY_FIXTURES.csv",index=False)
    coord=[]; lookup_xy={(float(x),float(y)) for x,y in geom}
    for state in occupied:
        ci=int(np.flatnonzero(states==state)[0]); registered=(0.0,2500.0)
        observed=(registered[0],registered[1]+state)
        corrected=float(exact_motion.correct_s(np.array([center_local_s[ci]]),np.array([observed[1]]))[0])
        coord.append({"state_um":state,"chunk_index":ci,"registered_x_um":registered[0],
                      "registered_y_um":registered[1],"expected_observed_y_um":observed[1],
                      "observed_coordinate_exists":observed in lookup_xy,"corrected_y_um":corrected,
                      "passed":observed in lookup_xy and corrected==registered[1]})
    pd.DataFrame(coord).to_csv(args.output/"INDEPENDENT_COORDINATE_FIXTURES.csv",index=False)
    if not all(r["passed"] for r in boundary_rows+coord): raise RuntimeError("query/sign fixture failed")

    # Saved-array voltage fixtures: real modified donor templates, exact relocation, local only.
    pair=None; best=-np.inf
    for a in range(len(donor_ids)):
        for b in range(a+1,len(donor_ids)):
            c=cosine(modified[a],modified[b])
            if c>best: best=c; pair=(a,b)
    a,b=pair
    if (int(donor_ids[a]),int(donor_ids[b])) != (407,415):
        raise RuntimeError("frozen most-similar donor pair changed")
    strongest=int(np.argmax(np.ptp(modified,axis=1).max(axis=1)))
    def crop_move(i,state=0):
        return exact_remap(modified[i],exact_same_column_map(geom,float(bases[i]+state)))[:,TARGET]
    va,vb,vs=crop_move(a),crop_move(b),crop_move(strongest)
    placed_cosine=cosine(va,vb)
    n=340; fixture=np.zeros((4,n,len(TARGET)),np.float32)
    fixture[0,60:181]+=va; fixture[0,66:187]+=vb
    fixture[1,60:181]+=va; fixture[1,190:311]+=vb
    fixture[2,80:201]+=vs
    # fixture[3] is the declared zero-injection control.
    np.savez_compressed(args.output/"LOCAL_DONOR_VOLTAGE_FIXTURES.npz",voltage=fixture,
                        case_names=np.array(["real_overlap_distinct_similar","distinct_similar_separated","strong_source","zero_injection"]),
                        donor_a=np.int64(donor_ids[a]),donor_b=np.int64(donor_ids[b]),strong_donor=np.int64(donor_ids[strongest]))
    fixture_rows=pd.DataFrame([
        {"case":"real_overlap_distinct_similar","donor_ids":f"{donor_ids[a]};{donor_ids[b]}","known_source":True,"preplacement_cosine":best,"placed_cosine":placed_cosine,"nonzero":True},
        {"case":"distinct_similar_separated","donor_ids":f"{donor_ids[a]};{donor_ids[b]}","known_source":True,"preplacement_cosine":best,"placed_cosine":placed_cosine,"nonzero":True},
        {"case":"strong_source","donor_ids":str(donor_ids[strongest]),"known_source":True,"preplacement_cosine":np.nan,"placed_cosine":np.nan,"nonzero":True},
        {"case":"zero_injection","donor_ids":"","known_source":True,"preplacement_cosine":np.nan,"placed_cosine":np.nan,"nonzero":False},
    ]); fixture_rows.to_csv(args.output/"FIXTURE_CASES.csv",index=False)

    # Execute the actual v3 scorer entrypoint, including the published counterexample.
    truth=np.array([0,18,100,100,200]); td=np.array([30,30,int(donor_ids[a]),int(donor_ids[b]),int(donor_ids[-1])])
    associations=[{"donor_id":30,"primary_label":10},{"donor_id":int(donor_ids[a]),"primary_label":20},
                  {"donor_id":int(donor_ids[b]),"primary_label":21},{"donor_id":int(donor_ids[-1]),"primary_label":None}]
    output=np.array([12,30,100,100,105,300,310,320,330,340]); labels=np.array([10,10,20,21,99,20,20,20,20,20])
    provenance=np.array(["injection_candidate"]*5+["background_supported","novel","unknown","noise","injection_candidate"])
    scored=score_with_frozen_association_v3(truth,td,np.zeros(len(truth),bool),output,labels,np.zeros(len(output),bool),associations,12,output_provenance=provenance)
    if len(scored["matches"])!=4 or [(r["truth_sample"],r["output_sample"]) for r in scored["matches"][:2]]!=[(0,12),(18,30)]:
        raise RuntimeError("scorer maximum-cardinality fixture")
    (args.output/"SCORER_ORACLE_FIXTURE.json").write_text(json.dumps(scored,indent=2,sort_keys=True)+"\n")
    fresh=score_with_frozen_association_v3(truth,td,np.zeros(len(truth),bool),output,labels,
                                            np.zeros(len(output),bool),associations,12)
    if any(r["tp"] or r["injection_fp"] for r in fresh["scores"]):
        raise RuntimeError("fresh outputs were promoted to oracle provenance")
    (args.output/"SCORER_FRESH_OUTPUT_FIXTURE.json").write_text(json.dumps(fresh,indent=2,sort_keys=True)+"\n")

    source_files=[CX/"ROBUST_EXPLORATORY_MODIFIED_SET.csv",CX/"DONOR_COORDINATE_TAPERS.npz",
                  BC/"full_probe_final_label_templates.npz",FIELD,ACTUAL_FIELDS,D2L/"input-manifest.json",
                  Path(__file__),ROOT/"testing/luke_dh_hybrid_scorer.py",ROOT/"testing/luke_dh_exact_chunk_motion.py",
                  ROOT/"testing/dh_actual_consumer_fixture.py"]
    manifest={
        "schema":"dh-hybrid-contract-v1","status":"frozen_fixture_complete_full_run_not_authorized",
        "created_utc":datetime.now(timezone.utc).isoformat(),"donor_count":25,"donor_ids":list(map(int,donor_ids)),
        "interpretation":"CX modified-source 25-donor exploratory intersection; not pristine donor validation",
        "window":{"id":"W2","start_frame":START,"end_frame_exclusive":END,"frames":END-START,"sampling_frequency_hz":FS,"clock":"local sample plus START gives AP absolute frame"},
        "source_domain":{"measurement_domain":domain,"template_shape":list(modified.shape),"target_channel_ids":list(map(str,channels[TARGET])),
                         "target_index_half_open":[202,384],
                         "temporal_source_separation":"independent unit-specific seeded renewal trains; linear addition preserves overlaps",
                         "spatial_relocation":"full 384-site measured donor template tapered in donor coordinates, exact same-column relocation, then cropped to physical AP202:383",
                         "support_gate":"the frozen CX qualification equations, evaluated without donor reselection for all 25 donors x 7 occupied states"},
        "injection":{"base_seed":20260926,"rate_hz":5.0,"refractory_ms":3.0,"guard_s":1.0,"waveform_samples":121,"trough_offset_samples":trough,
                     "operator":"source(x,y) -> target(x,y+base+state), exact 40-um same-column lattice, zero off-probe, linear sum",
                     "shared_source":"inject exactly once before S_h/D2L_h branching","occupied_states_um":occupied,
                     "event_count":len(events),"overlap_connected_groups":len(set(groups.values())),"write_bounds":"[sample_start,sample_stop), clipped writes forbidden"},
        "consumer_query_contract":{"matching_chunk_samples":CHUNK,"chunk_count":len(starts),"chunk_bounds":"[start,min(start+7500,W2_frames))",
                                   "boundary_rule":"event at an exact 7500-sample boundary uses the new chunk",
                                   "query_time":"recording-local (chunk_start+3750)/FS, exactly matching DARTsort matching.py; session time is (START+centre)/FS only for source-field lookup",
                                   "query_support":"strict local sample clock; [0,ceil(W2_frames/7500)*7500) with explicit final-chunk padding; no clipping of wrong-origin/out-of-support queries",
                                   "injection_state":"actual saved D2L rigid field at the installed consumer query time, rounded half away from zero to 40 um",
                                   "S_h":"same shared injected voltage; static motion disabled",
                                   "D2L_h":"same shared injected voltage; ExactChunkLatticeMotion consumes the identical pitch-quantized state used for injection"},
        "smallest_subsequent_paths":{"S_h":{"config":str(STATIC/"effective-config.json"),"input":"one future shared injected W2 recording","motion":"disabled"},
                                     "D2L_h":{"config":str(D2L/"sort/effective-config.json"),"motion_adapter":"testing/luke_dh_exact_chunk_motion.py plus EXACT_QUERY_MOTION.npz","unrounded_field_role":"state derivation only","input":"same byte-identical future shared injected W2 recording"},
                                     "DZ_h":{"status":"conditional only; no path frozen in DH"}},
        "subsequent_budget_estimate":{"input_materialization":"one W2 float32 182-site source ~=7.43 GB plus receipt","sorts":"two DARTsort W2 arms; use measured S/D2L receipts for service caps before authorization","DH_full_launch":False},
        "scoring":{"version":"v3","tolerance_samples":12,"association":"frozen full-train association before region split",
                   "matching":"exact permitted-edge maximum cardinality, then minimum total timing error, deterministic stable tie policy",
                   "provenance":"fresh assigned outputs begin unknown; only independent evidence may promote provenance. Background cannot match injection truth and is not injection FP; negative labels/noise are excluded; ambiguous matches remain unresolved; novel/unknown reported separately; missing donors retained; cross-boundary global matches are reported explicitly",
                   "oracle_fixture":"SCORER_ORACLE_FIXTURE.json is synthetic and is not a rule for assigning fresh-sort provenance"},
        "inputs":[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)} for p in source_files],
        "resources":{"source_bytes_read":sum(p.stat().st_size for p in source_files[:6]),"raw_voltage_bytes":0,"gpu_s":0,"sorts":0,
                     "cpu_s":time.process_time()-cpu0,"wall_s":time.monotonic()-started,"threads_max":2,"readers_max":1},
    }
    (args.output/"HYBRID_MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    (args.output/"README.md").write_text("""# DH hybrid truth contract

Concrete 25-donor exploratory W2 contract and compact saved-array fixtures. No
full hybrid voltage or sort was created. V2 scorer is preserved; v3 fixes
global cardinality and separates output provenance.

`ACTUAL_DARTSORT_CONSUMER_FIXTURE.json` proves the exact pitch states through
the installed DARTsort `MotionInfo` consumer at the same recording-local query
times used by `MatchingPeeler.peel_chunk`. The adapter rejects wrong-origin and
out-of-support times and explicitly permits only the 82-sample padded tail of
the final declared matching chunk.

Fresh-sort assigned spikes begin with `unknown` provenance. The oracle fixture
is synthetic and only tests accounting. It cannot assign biological identity.
`DONOR_STATE_SUPPORT.csv` records every frozen donor/state result; there is no
qualification loop or donor reselection.
""")
    (args.output/"RESOURCE_RECEIPT.json").write_text(json.dumps({
        "status":"complete","cpu_allwork_cap_s":900,"measured_fixture_cpu_s":manifest["resources"]["cpu_s"],
        "conservative_active_charge_s":120.0,"threads_max":2,"readers_max":1,
        "source_bytes_read":manifest["resources"]["source_bytes_read"],"saved_donor_read_cap_bytes":268435456,
        "raw_voltage_bytes":0,"gpu_s":0,"sorts":0,"scratch_bytes":0,
        "final_bytes_before_manifest":sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()),
        "prior_h1_cumulative_active_s":18263.22,"new_h1_cumulative_active_s":18383.22,
        "overall_ceiling_active_s":26000.0,"overall_remaining_active_s":7616.78,
    },indent=2,sort_keys=True)+"\n")
    files=[]
    for p in sorted(args.output.iterdir()):
        if p.name not in {"MANIFEST.json","COMPLETE.json"}: files.append({"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)})
    (args.output/"MANIFEST.json").write_text(json.dumps({"files":files,"total_bytes":sum(x["bytes"] for x in files)},indent=2,sort_keys=True)+"\n")
    (args.output/"COMPLETE.json").write_text(json.dumps({"status":"complete","manifest_sha256":sha(args.output/"MANIFEST.json")},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"complete","donors":25,"events":len(events),"occupied_states":occupied,"fixture_pair":[int(donor_ids[a]),int(donor_ids[b])],"cpu_s":manifest["resources"]["cpu_s"],"wall_s":manifest["resources"]["wall_s"]}))

if __name__=="__main__": main()
