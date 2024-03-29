# -*- coding: utf-8 -*-
"""
This example shows the evaluation process used in CHAOS challenge. 
A sample data is shared with original DICOM images, its ground truth 
and an example of segmentation result.

Ground truth volume is used as reference to evaluate sample segmentation.

DICOM folder is used to transform voxel values into real world coordinates.
Hence, ASSD and MSSD metrics are calculated with millimeter.

09/07/2019
@author: Ali Emre Kavur, emrekavur@gmail.com
"""

import os
from CHAOSmetrics import png_series_reader
from CHAOSmetrics import evaluate

# ======= Directories =======
# cwd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
cwd = os.path.normpath(os.getcwd() + os.sep)
print(cwd)
ground_dir = os.path.normpath(cwd + '/Data_3D/Ground')
seg_dir = os.path.normpath(cwd + '/Data_3D/Segmentation')
dicom_dir = os.path.normpath(cwd + '/Data_3D/DICOM_anon')

# ======= Volume Reading =======
Vref = png_series_reader(ground_dir)
Vseg = png_series_reader(seg_dir)
print(Vseg.shape)
print('Volumes imported.')
# ======= Evaluation =======
print('Calculating...')
[dice, ravd, assd ,mssd]=evaluate(Vref,Vseg,dicom_dir)
print('DICE=%.3f RAVD=%.3f ASSD=%.3f MSSD=%.3f' %(dice, ravd, assd ,mssd))

# import nibabel as nib
# print(nib.load('/home/user/DISK/data/Jing/data/Training/raw/img0001.nii.gz').affine)
