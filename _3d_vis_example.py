import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from evals import evaluator
from evals import eval_utils

# 1. MICCAI BTCV dataset 3d visualization example
path = "/home/user/DISK/data/Jing/data/2015_MICCAI_BTCV/Train_Sets/img/img0001.nii.gz"
label_path = "/home/user/DISK/data/Jing/data/2015_MICCAI_BTCV/Train_Sets/label/label0001.nii.gz"
data = nib.load(label_path).get_data()

evaluate = evaluator.build_evaluator()
evaluate.visualize_in_3d({"test": np.int32(data==1)}, raw_data_path=path)


# 2. ISBI CHAOS dataset 3d visualization example
path = "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/Train_Sets/MR/1/T2SPIR/DICOM_anon/"
label_path = "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/Train_Sets/MR/1/T2SPIR/Ground/"
files = os.listdir(label_path)
files.sort()
labels = [np.int32(
    sitk.GetArrayFromImage(
        sitk.ReadImage(
            label_path+f)))[...,np.newaxis] for f in files]
data =  np.concatenate(labels, axis=2)

evaluate = evaluator.build_dicom_evaluator()
evaluate.visualize_in_3d({"test": np.int32(data==63)}, raw_data_path=path)



