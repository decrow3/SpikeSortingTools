"""Export only the directly approved complete CX masked-donor packet."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
SRC=Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools/testing/outputs/cx_masked_donor_qualification_v1/run')
DST=Path('/mnt/NPX/Luke/DARTsort_motion_experiments/cx_masked_donor_qualification_20260928')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def wj(p,x):
 q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def main():
 manifest=json.loads((SRC/'MANIFEST.json').read_text());complete=json.loads((SRC/'COMPLETE.json').read_text())
 if sha(SRC/'MANIFEST.json')!=complete['manifest_sha256']:raise ValueError('source manifest self-hash')
 for row in manifest['files']:
  p=SRC/row['path']
  if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError(row['path'])
 taper=SRC/'DONOR_COORDINATE_TAPERS.npz'
 if taper.stat().st_size!=121434 or sha(taper)!='9b2593cd0081490c47a833df56b896bf5b1ba9044a4db27517bda244017ad250':raise ValueError('taper')
 DST.mkdir(parents=True,exist_ok=True)
 for row in manifest['files']:
  source=SRC/row['path'];dest=DST/row['path'];dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and (dest.stat().st_size!=row['bytes'] or sha(dest)!=row['sha256']):raise FileExistsError(f'distinct destination collision: {dest}')
  if not dest.exists():shutil.copy2(source,dest)
 shutil.copy2(SRC/'MANIFEST.json',DST/'SOURCE_MANIFEST.json');shutil.copy2(SRC/'COMPLETE.json',DST/'SOURCE_COMPLETE.json')
 destination=[]
 for row in manifest['files']:
  p=DST/row['path'];destination.append({'path':row['path'],'bytes':p.stat().st_size,'sha256':sha(p)})
 if destination!=manifest['files']:raise ValueError('destination manifest verification')
 receipt={'status':'export_complete','exported_utc':datetime.now(timezone.utc).isoformat(),'authority':'direct user approval relayed 2026-09-28 06:45 UTC','source':str(SRC),'destination':str(DST),'source_manifest_sha256':sha(SRC/'MANIFEST.json'),'destination_source_manifest_sha256':sha(DST/'SOURCE_MANIFEST.json'),'files_verified':len(destination),'total_bytes':sum(x['bytes'] for x in destination),'taper_bytes':121434,'taper_sha256':sha(DST/'DONOR_COORDINATE_TAPERS.npz'),'preserved_existing_distinct_files':['EXPORT_BLOCKER.json','START_RECEIPT.json','dartsort_cx_masked_donor_qualification_scope_20260928.md'],'excluded':'no raw voltage; no BH payload; no unrelated files','scientific_interpretation':'modified-full-source 25/90 exploratory; original 5/90 failure unchanged'}
 wj(DST/'EXPORT_RECEIPT.json',receipt);wj(DST/'COMPLETE.json',{'status':'complete_approved_export','export_receipt_sha256':sha(DST/'EXPORT_RECEIPT.json'),'source_manifest_sha256':receipt['source_manifest_sha256'],'taper_bytes':121434,'written_last':True})
if __name__=='__main__':main()
