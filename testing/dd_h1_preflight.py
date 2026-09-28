"""DD h1 rescue-KS input, geometry, q-table and configuration preflight."""
from __future__ import annotations
import hashlib,json,os,shutil,time,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from testing.dd_lattice_remap import build_knot_table,coordinate_map,pair_hash,sample_q

FS=29999.759166666667; START=26999783; END=37199701
ROOT=Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dd_lattice_w2_20260928/host_h1")
RESCUE=Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec1")
FIELD=Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage4_ab/luke0804_imec1_two_layer_motion.npz")
MASK=Path("/mnt/NPX/Luke/DARTsort_motion_experiments/ck_input_intervals_20260927/censor_mask_v1.csv")

def sha(p):
 h=hashlib.sha256();
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()

def wj(p,x):
 q=p.with_suffix(p.suffix+".partial"); q.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n"); os.replace(q,p)

def main():
 started=time.perf_counter(); out=ROOT/"preflight"
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True); (out/"source").mkdir()
 for p in (Path(__file__),Path(__file__).with_name("dd_lattice_remap.py"),Path(__file__).with_name("test_dd_lattice_remap.py")): shutil.copy2(p,out/"source"/p.name)
 if sha(FIELD)!="85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9" or sha(MASK)!="86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55": raise ValueError("field/mask hash")
 rec_manifest=json.loads((RESCUE/"recording/rescue_recording_manifest.json").read_text()); sort_manifest=json.loads((RESCUE/"kilosort4/rescue_sort_manifest.json").read_text()); params=json.loads((RESCUE/"kilosort4/spikeinterface_params.json").read_text())["sorter_params"]
 from spikeinterface.core import load
 rec=load(RESCUE/"recording"); geom=np.asarray(rec.get_channel_locations(),float); ids=np.asarray(rec.get_channel_ids()).astype(str)
 if rec.get_num_channels()!=384 or not np.isclose(rec.get_sampling_frequency(),FS,rtol=0,atol=1e-9): raise ValueError("recording identity")
 knots,reference=build_knot_table(FIELD,MASK,START,END,FS); knots.to_csv(out/"W2_KNOT_Q.csv",index=False,float_format="%.12g")
 q=sample_q(END-START,START,FS,knots); qvals,qcounts=np.unique(q,return_counts=True)
 bad_id="imec1.ap#AP191"; bad_index=int(np.flatnonzero(ids==bad_id)[0]); mapping=[]; total_zero=0; total_bad=0
 for value,count in zip(qvals,qcounts):
  m=coordinate_map(geom,int(value)); off=int((m<0).sum()); bad=int((m==bad_index).sum())
  mapping.append({"q_um":int(value),"samples":int(count),"valid_targets":int((m>=0).sum()),"off_probe_targets":off,"bad_source_targets":bad})
  total_zero+=int(count)*off; total_bad+=int(count)*bad
 if not np.array_equal(coordinate_map(geom,0),np.arange(384)): raise ValueError("q0 is not identity")
 ops=np.load(RESCUE/"kilosort4/sorter_output/ops.npy",allow_pickle=True).item(); effective=dict(ops.get("settings",{})); effective.update(ops)
 critical={k:effective.get(k) for k in ("nblocks","do_CAR","highpass_cutoff","whitening_range")}
 checks={"do_correction_requested_false":params["do_correction"] is False,"effective_nblocks_zero":int(effective["nblocks"])==0,"do_CAR_true":bool(effective["do_CAR"]),"highpass_300":float(effective["highpass_cutoff"])==300.,"whitening_range_32":int(effective["whitening_range"])==32,"external_filter_none":rec_manifest["external_filter"] is None,"external_reference_none":rec_manifest["external_reference"] is None,"external_motion_false":rec_manifest["external_voltage_motion_correction"] is False,"bad_channel_exact":rec_manifest["bad_channel_ids"]==[bad_id],"q0_identity":True,"sample_clock_preserved":True,"geometry_unique":len({tuple(x) for x in geom})==384}
 if not all(checks.values()): raise ValueError(checks)
 receipt={"status":"pass","actual_start_utc":datetime.now(timezone.utc).isoformat(),"source_frames_half_open":[START,END],"sampling_frequency_hz":FS,"samples":END-START,"channels":384,"source_recording":str(RESCUE/"recording"),"source_graph":rec_manifest["graph"],"input_dtype":str(rec.dtype),"bad_channel_id":bad_id,"bad_channel_index":bad_index,"reference_median_um":reference,"knot_rows":len(knots),"pair_line_knot_q_sha256":pair_hash(knots),"full_knot_csv_sha256":sha(out/"W2_KNOT_Q.csv"),"q_sample_counts":dict(zip(map(str,qvals),map(int,qcounts))),"mapping_support":mapping,"zero_filled_values":total_zero,"interpolated_bad_source_values":total_bad,"checks":checks,"effective_kilosort":critical,"sorter_thresholds":{"Th_universal":params["Th_universal"],"Th_learned":params["Th_learned"]},"preprocessing_boundary":"reuse accepted phase/blank/interpolate int16; lattice permute; Kilosort highpass/CAR/whitening once","elapsed_s":time.perf_counter()-started,"canonical_h5_comparison":"pending source publication; compare identical synthetic fixture before materialization"}
 wj(out/"INPUT_PREFLIGHT.json",receipt); wj(out/"COMPLETE.json",{"status":"complete_pre_materialization","receipt_sha256":sha(out/"INPUT_PREFLIGHT.json"),"written_last":True})

if __name__=="__main__": main()
