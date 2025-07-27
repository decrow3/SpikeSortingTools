from scipy.io import savemat
import numpy as np
import glob
import os
npzFiles = glob.glob("/mnt/NPX/Rocky/20240704/RedoneSorts/pipeline_results_Rocky20240704_V1V2_g0_imec0/cur/cur_sorter_output/ops.npy")


for f in npzFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f,allow_pickle=True)
    xc=d.item()['xc']
    yc=d.item()['yc']
    matout={"xc":xc,"yc":yc}
    savemat(fm, matout)
    print('generated ', fm, 'from', f)
