from scipy.io import savemat
import numpy as np
import glob
import os
npzFiles = glob.glob("/home/huklab/Documents/RyanSorting/SpikeSortingTools/pipeline_results_Rocky20240826_V1V2_g0_imec1_locar_40_75/cur/cur_sorter_output/ops.npy")

for f in npzFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f,allow_pickle=True)
    xc=d.item()['xc']
    yc=d.item()['yc']
    matout={"xc":xc,"yc":yc}
    savemat(fm, matout)
    print('generated ', fm, 'from', f)
