"""Wait independently for remaining arms, verify sealed fits and combine reports."""
import json,time
import pandas as pd
from testing.luke_ap_methods_sweep_v1 import ROOT,valid,sha
from testing import luke_ap_methods_sweep_report_v1 as report
SRC=ROOT/'testing/outputs';NEW=SRC/'luke_ap_methods_remaining_v1';OUT=SRC/'luke_ap_methods_completed40_v1'
def main():
 for _ in range(60):
  path=NEW/'progress.json';rows=json.loads(path.read_text()) if path.exists() else []
  if any(r['returncode'] for r in rows):raise RuntimeError('Remaining arm failed: preserve evidence; no full report')
  if len(rows)==8:break
  time.sleep(30)
 else:raise RuntimeError('Timeout waiting for remaining arms')
 m=pd.read_csv(SRC/'luke_ap_methods_sweep_v1/manifest.csv');assert len(m)==32
 for r in m.itertuples():assert valid(__import__('pathlib').Path(r.field).parent)
 for r in rows:
  assert valid(__import__('pathlib').Path(r['field']).parent)
  m.loc[len(m)]=[r['input'],r['method'],False,r['seconds'],r['field']]
 assert len(m)==40 and not m.duplicated(['input','method']).any()
 OUT.mkdir(exist_ok=True);m.to_csv(OUT/'manifest.csv',index=False);report.OUT=OUT;report.report()
 (OUT/'continuation_provenance.json').write_text(json.dumps(dict(original_completed=32,new_completed=8,offsets_unchanged=True,original_failure_preserved=True,original_failure_cause='Unresolved; separate retry passed unchanged strict assertion',source_script_sha256={str(p):sha(p) for p in [__import__('pathlib').Path(__file__),ROOT/'testing/luke_ap_methods_remaining_fit_v1.py',ROOT/'testing/luke_dredge_bounded_250_audit_v1.py']}),indent=2))
if __name__=='__main__':main()
