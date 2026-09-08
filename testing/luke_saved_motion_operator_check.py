"""CPU-only application-sign check on the saved geometry; never sorts."""
import argparse
import numpy as np,json
from pathlib import Path
from kilosort.preprocessing import get_drift_matrix
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--group-root',type=Path,required=True)
p.add_argument('--output-root',type=Path,required=True)
a=p.parse_args()
root=a.group_root
out=a.output_root
o=np.load(root/'arms/rescue_12_9_native_rigid/sort/sorter_output/ops.npy',allow_pickle=True).item()
o['iKxx']=torch.from_numpy(o['iKxx'])
y=o['yc'];x=o['xc'];f=lambda delta:np.exp(-((y-(1880+delta))/25)**2/2-((x-24)/35)**2/2)
target=f(0);moved=f(20)
errors={}
for d in [-20,0,20]:
 mat=get_drift_matrix(o,np.array([d]),device=torch.device('cpu')).numpy()
 errors[str(d)]=float(np.linalg.norm(mat@moved-target)/np.linalg.norm(target))
assert errors['-20']<errors['0']<errors['20']
print(errors)
(out/'operator_sign_check.json').write_text(json.dumps(dict(physical_displacement_um=20,relative_l2_errors=errors,description='Analytic smooth spatial profile on saved geometry; checks application sign, not real waveform fidelity.'),indent=2)+'\n')
