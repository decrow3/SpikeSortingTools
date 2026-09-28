#!/usr/bin/env python3
"""Publish DH source and compact non-voltage results under standing authority."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, shutil

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"testing/outputs/dh_hybrid_truth_contract_20260928"
DST=Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dh_hybrid_truth_contract_20260928/host_h1")
ALLOW=["START_RECEIPT.json","DONORS.csv","SOURCE_EVENTS.csv","INJECTION_MEMBERSHIP.csv",
       "DONOR_STATE_SUPPORT.csv","EXACT_QUERY_MOTION.npz","QUERY_BOUNDARY_FIXTURES.csv",
       "INDEPENDENT_COORDINATE_FIXTURES.csv","ACTUAL_DARTSORT_CONSUMER_FIXTURE.json",
       "FIXTURE_CASES.csv","SCORER_ORACLE_FIXTURE.json","SCORER_FRESH_OUTPUT_FIXTURE.json",
       "HYBRID_MANIFEST.json","RESOURCE_RECEIPT.json","README.md","MANIFEST.json"]
SOURCES=[ROOT/"testing/luke_dh_hybrid_contract.py",ROOT/"testing/luke_dh_hybrid_scorer.py",
         ROOT/"testing/luke_dh_exact_chunk_motion.py",ROOT/"testing/dh_actual_consumer_fixture.py",
         ROOT/"testing/test_luke_dh_hybrid_scorer.py",ROOT/"testing/test_luke_dh_exact_chunk_motion.py"]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()

def main():
 if json.loads((SRC/"COMPLETE.json").read_text())["status"]!="complete": raise RuntimeError("local incomplete")
 DST.mkdir(parents=True,exist_ok=True); (DST/"source").mkdir(exist_ok=True)
 rows=[]
 for source,dest in [(SRC/n,DST/n) for n in ALLOW]+[(p,DST/"source"/p.name) for p in SOURCES]:
  if dest.exists() and sha(dest)!=sha(source): raise FileExistsError(dest)
  if not dest.exists():
   tmp=dest.with_suffix(dest.suffix+".partial");shutil.copy2(source,tmp);os.replace(tmp,dest)
  rows.append({"path":str(dest.relative_to(DST)),"bytes":dest.stat().st_size,"sha256":sha(dest)})
 manifest={"status":"complete","published_utc":datetime.now(timezone.utc).isoformat(),"files":rows,
           "excluded":["LOCAL_DONOR_VOLTAGE_FIXTURES.npz","all raw voltage","all BH payload"],
           "note":"source plus compact tables/receipts only; local donor-voltage fixture deliberately not transferred"}
 tmp=DST/"SHARED_MANIFEST.json.partial";tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");os.replace(tmp,DST/"SHARED_MANIFEST.json")
 complete={"status":"complete","shared_manifest_sha256":sha(DST/"SHARED_MANIFEST.json"),"written_last":True}
 tmp=DST/"COMPLETE.json.partial";tmp.write_text(json.dumps(complete,indent=2,sort_keys=True)+"\n");os.replace(tmp,DST/"COMPLETE.json")
if __name__=="__main__":main()
