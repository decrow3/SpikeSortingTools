#!/usr/bin/env python3
"""Add a non-destructive invalidation marker to the bad DK v1 publication."""
from pathlib import Path
import json
import os

V1=Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dk_hybrid_w2_20260928/host_h1_scorer_correction_v1")
MARKER=V1/"INVALIDATED.json"

def main():
 if MARKER.exists(): raise FileExistsError(MARKER)
 value={
  "status":"invalid_publication_manifest_collision",
  "reason":"The publication manifest replaced a copied local MANIFEST.json while listing the copied file's prior hash. Scientific payload hashes other than that self-collision matched.",
  "use_instead":"/mnt/NPX/Luke/DARTsort_motion_experiments/dk_hybrid_w2_20260928/host_h1_scorer_correction_v2/",
  "v2_manifest_sha256":"32214f6c7fc843ac9b4c70f297fee45178a8f9fcb3a5d68d1022d361ac7fb959",
  "do_not_use_v1":True,
 }
 partial=MARKER.with_suffix('.json.partial'); partial.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(partial,MARKER)

if __name__=='__main__': main()
