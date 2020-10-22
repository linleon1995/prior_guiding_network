import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom

from evals import metrics
from evals import eval_utils


_ALL_METRICS = {"RAVD": metrics.RAVD,
                "ASSD": metrics.ASSD,
                "MSSD": metrics.MSSD,
                "ASSD_and_MSSD": metrics.ASSD_and_MSSD,
                "DSC": metrics.DICE,
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "IoU": metrics.compute_mean_iou,
                "image_dice": metrics.compute_mean_dsc,
                "Accuracy": metrics.compute_accuracy}

class build_evaluator(object):
    """
    nifti
    """
    default_metrics = ["RAVD", 
                       "ASSD", 
                       "MSSD", 
                       "DSC", 
                       "Precision", 
                       "Recall", 
                       "IoU", 
                       "image_dice", 
                       "Accuracy"]
    def __init__(self, metrics=None):
        self.metrics = []
        if metrics is not None:
            for m in metrics:
                self.metrics.append(m)
        else:
            for m in self.default_metrics:
                self.metrics.append(m)
        
        if "ASSD" in self.metrics and "MSSD" in self.metrics:
            self.metrics.remove("ASSD")
            self.metrics.remove("MSSD")
            self.metrics.append("ASSD_and_MSSD")
                                
        self.transform_flag = False
        for _3d_m in ["DSC", "RAVD", "MSSD", "ASSD"]:
            if _3d_m in self.metrics:
                self.transform_flag = True
                break

    def __call__(self, ref, test, **metrics_kwargs):
        results = {}
        # Provide affine matrix for metrics that need transformation
        if self.transform_flag:
            raw_data_path = metrics_kwargs.pop("raw_data_path")
            affine = self.get_affine(raw_data_path)
            metrics_kwargs["affine"] = affine
        
        # Check the shape consistency between prediction and ground truth    
        self.check_shape(ref, test)
        
        # Get all the results of assign maetrics
        for m in self.metrics:
            if m == "ASSD_and_MSSD":
                results["ASSD"], results["MSSD"] = _ALL_METRICS[m](ref, test, **metrics_kwargs)
            else:    
                results[m] = _ALL_METRICS[m](ref, test, **metrics_kwargs)
        return results

    def get_affine(self, raw_data_path):
        affine = nib.load(raw_data_path).affine.copy()
        return affine

    def check_shape(self, ref, test):
        assert ref.shape == test.shape
        # These metrics require 3D data
        for _3d_m in ["DSC", "RAVD", "MSSD", "ASSD"]:
            if _3d_m in self.metrics:
                if ref.ndim != 3 or test.ndim != 3:
                    raise ValueError("Incorrect shape for %s, this metrics require 3d data" %_3d_m)
                break

    def visualize_in_3d(self, ref, test, raw_data_path):
        affine = self.get_affine(raw_data_path)
        eval_utils.show_3d({"Reference": ref, "Segmentation": test}, affine)
        

class build_dicom_evaluator(build_evaluator):
    def get_affine(self, dicom_dir):
        # dicom_file_list=glob.glob(dicom_dir + '/*.dcm')
        dicom_file_list = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
        dicom_file_list.sort()
        #Read position and orientation info from first image
        ds_first = pydicom.dcmread(dicom_file_list[0])
        img_pos_first=list( map(float, list(ds_first.ImagePositionPatient)))
        img_or=list( map(float, list(ds_first.ImageOrientationPatient)))
        pix_space=list( map(float, list(ds_first.PixelSpacing)))
        #Read position info from first image from last image
        ds_last = pydicom.dcmread(dicom_file_list[-1])
        img_pos_last=list( map(float, list(ds_last.ImagePositionPatient)))

        T1=img_pos_first
        TN=img_pos_last
        X=img_or[:3]
        Y=img_or[3:]
        deltaI=pix_space[0]
        deltaJ=pix_space[1]
        N=len(dicom_file_list)
        affine=np.array([[X[0]*deltaI,Y[0]*deltaJ,(T1[0]-TN[0])/(1-N),T1[0]], [X[1]*deltaI,Y[1]*deltaJ,(T1[1]-TN[1])/(1-N),T1[1]], [X[2]*deltaI,Y[2]*deltaJ,(T1[2]-TN[2])/(1-N),T1[2]], [0,0,0,1]])
        return affine





