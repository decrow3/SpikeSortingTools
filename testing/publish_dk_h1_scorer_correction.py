#!/usr/bin/env python3
"""Publish the immutable compact DK scorer correction packet."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, shutil

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"testing/outputs/dk_h1_scorer_boundary_correction_20260928"
DST=Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dk_hybrid_w2_20260928/host_h1_scorer_correction_v2")
SOURCES=[ROOT/"testing/dk_h1_scorer_boundary_correction.py",ROOT/"testing/luke_dh_hybrid_scorer.py",ROOT/"testing/test_luke_dh_hybrid_scorer.py"]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 c=json.loads((SRC/'COMPLETE.json').read_text())
 if not c['status'].startswith('complete'): raise RuntimeError('local incomplete')
 if DST.exists(): raise FileExistsError(DST)
 DST.mkdir(parents=True); (DST/'source').mkdir()
 pairs=[]
 for p in sorted(SRC.iterdir()):
  if not p.is_file() or p.name=='COMPLETE.json': continue
  dest_name='SOURCE_MANIFEST.json' if p.name=='MANIFEST.json' else p.name
  pairs.append((p,DST/dest_name))
 pairs += [(p,DST/'source'/p.name) for p in SOURCES]
 rows=[]
 for source,dest in pairs:
  partial=dest.with_suffix(dest.suffix+'.partial'); shutil.copy2(source,partial); os.replace(partial,dest)
  rows.append({'path':str(dest.relative_to(DST)),'bytes':dest.stat().st_size,'sha256':sha(dest)})
 manifest={'status':c['status'],'published_utc':datetime.now(timezone.utc).isoformat(),'files':rows}
 (DST/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 complete={'status':c['status'],'manifest_sha256':sha(DST/'MANIFEST.json'),'written_last':True}
 partial=DST/'COMPLETE.json.partial'; partial.write_text(json.dumps(complete,indent=2,sort_keys=True)+'\n'); os.replace(partial,DST/'COMPLETE.json')

if __name__=='__main__': main()
