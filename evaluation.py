import h5py
import os
import glob
import numpy as np


# numpy version
def mrae_loss(outputs, label):
    """Computes the rrmse value"""
    diff = label-outputs
    abs_diff = np.abs(diff)
    relative_abs_diff = np.divide(abs_diff, label+np.finfo(float).eps)
    return np.mean(relative_abs_diff)


# numpy version
def rmse_loss(outputs, label):
    """Computes the rmse value"""
    diff = label - outputs
    square_diff = np.power(diff, 2)
    return np.sqrt(np.mean(square_diff))


result_path = 'results'
ground_path = 'NTIRE2018_Validate_Spectral'

result_name = glob.glob(os.path.join(result_path, '*.mat'))
result_name.sort()
ground_name = glob.glob(os.path.join(ground_path, '*.mat'))
ground_name.sort()

record_mrae, record_rmse = [], []
for i in range(len(ground_name)):
    # load rusults hyper
    hs = h5py.File(ground_name[i], 'r')
    ground = np.float32(hs.get('rad'))   # 31,1300,1392
    hs = h5py.File(result_name[i], 'r')
    result = np.float32(hs.get('rad'))   # 31,1300,1392
    mrae = mrae_loss(result, ground)
    rmse = rmse_loss(result, ground)
    record_mrae.append(mrae)
    record_rmse.append(rmse)
print("Average mrae_loss [%.4f], average rmse_loss [%.2f]" % (np.array(record_mrae).mean(), np.array(record_rmse).mean()))
