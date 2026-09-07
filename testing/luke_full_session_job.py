"""Persistent-service entry point: sort/QC, then comparison, with exit receipts."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'testing/outputs/luke_full_session_rigid_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = dict(started_unix=time.time(), state='running')
    path = OUT / 'job.json'
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    for module, log in [('testing.luke_full_session_rigid', 'run.log'),
                        ('testing.luke_full_session_compare', 'comparison.log')]:
        with (OUT/log).open('ab') as stream:
            result = subprocess.run([sys.executable, '-u', '-m', module],
                                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        receipt.update(last_module=module, returncode=result.returncode, updated_unix=time.time())
        if result.returncode:
            receipt['state']='failed'
            path.write_text(json.dumps(receipt, indent=2)+'\n')
            (OUT/'status.json').write_text(json.dumps(dict(stage='failed', **receipt))+'\n')
            return 1
    receipt['state']='complete'
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
