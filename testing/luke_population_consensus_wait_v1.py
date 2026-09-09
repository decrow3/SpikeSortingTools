"""Independent downstream report job; wait for successful managed extraction."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'testing/outputs'

def main():
 receipt=SRC/'luke_population_depth_v2_job/receipt.json'
 source=SRC/'luke_population_depth_v2'
 report_script=ROOT/'testing/luke_population_consensus_v1.py'
 digest=hashlib.sha256(report_script.read_bytes()).hexdigest()
 previous=None
 for _ in range(2160):
  state=subprocess.run(['systemctl','--user','show','luke-population-depth-v2.service','-p','ActiveState','-p','MainPID'],capture_output=True,text=True,check=True).stdout.strip()
  r=json.loads(receipt.read_text()) if receipt.exists() else {}
  progress=(r.get('state'),len(list(source.glob('s*.complete.json'))),state)
  if progress!=previous:print('Extraction:',progress,flush=True);previous=progress
  if r.get('state')=='complete':
   if r.get('returncode')!=0:raise RuntimeError('Extraction failed; preserve its evidence; report not run.')
   break
  if 'ActiveState=active' not in state and 'ActiveState=activating' not in state:
   raise RuntimeError('Extraction service is not live and has no successful completion receipt.')
  time.sleep(10)
 else:raise TimeoutError('No successful extraction within six hours; no report generated.')
 assert json.loads((source/'summary.json').read_text())['status']=='complete'
 assert hashlib.sha256(report_script.read_bytes()).hexdigest()==digest,'Report script changed while waiting; review and relaunch downstream job.'
 from testing.luke_population_depth_v2 import valid
 assert valid('templates') and all(valid(f's{s}') for s in range(930,1030,10)), 'Extraction checkpoint validation failed'
 print('Extraction stages verified; rendering population consensus.',flush=True)
 subprocess.run([sys.executable,'-m','testing.luke_population_consensus_v1'],cwd=ROOT,check=True)
 print('Population figures and report complete.',flush=True)
if __name__=='__main__':main()
