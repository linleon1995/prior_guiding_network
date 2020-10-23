# -*- coding: utf-8 -*-
"""
Created on 09/07/2019

@author: Ali Emre Kavur
"""
import os
import pydicom
import numpy as np
import glob
import cv2
import SimpleITK as sitk
from scipy import ndimage
from sklearn.neighbors import KDTree
import nibabel as nib
import matplotlib.pyplot as plt

from evals.chaos_eval import CHAOSmetrics
ori_RAVD = CHAOSmetrics.RAVD
ori_DICE = CHAOSmetrics.DICE


def RAVD(ref, test, **metrics_kwargs):
    """Relative Absolute Volume Difference"""
    return ori_RAVD(ref, test)

def DICE(ref, test, **metrics_kwargs):
    """Dice Score Coefficient"""
    return ori_DICE(ref, test)


def ASSD_and_MSSD(ref, test, **metrics_kwargs):
    """Average Symmetric Surface Distance"""
    affine = metrics_kwargs.pop("affine")
    num_class = metrics_kwargs.pop("num_class")
    ASSD, MSSD = [], []
    for c in range(1, num_class):
        ref_c = np.int32(ref==c)
        test_c = np.int32(test==c)
        if np.sum(ref_c) == 0 or np.sum(test_c) == 0:
          continue
        assd, mssd = SSD(ref_c, test_c, affine)
        ASSD.append(assd)
        MSSD.append(mssd)
    return np.array(ASSD), np.array(MSSD)

       
def ASSD(ref, test, **metrics_kwargs):
    """Average Symmetric Surface Distance"""
    return ASSD_and_MSSD(ref, test, **metrics_kwargs)[0]


def MSSD(ref, test, **metrics_kwargs):
    """Maximum Symmetric Surface Distance"""
    return ASSD_and_MSSD(ref, test, **metrics_kwargs)[1]


def SSD(Vref, Vseg, affine):
    struct = ndimage.generate_binary_structure(3, 1)

    ref_border=Vref ^ ndimage.binary_erosion(Vref, structure=struct, border_value=1)
    ref_border_voxels=np.array(np.where(ref_border))

    seg_border=Vseg ^ ndimage.binary_erosion(Vseg, structure=struct, border_value=1)
    seg_border_voxels=np.array(np.where(seg_border))

    ref_border_voxels_real=transformToRealCoordinates(ref_border_voxels, affine)
    seg_border_voxels_real=transformToRealCoordinates(seg_border_voxels, affine)

    tree_ref = KDTree(np.array(ref_border_voxels_real))
    dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels_real)
    tree_seg = KDTree(np.array(seg_border_voxels_real))
    dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels_real)

    assd=(dist_seg_to_ref.sum() + dist_ref_to_seg.sum())/(len(dist_seg_to_ref)+len(dist_ref_to_seg))
    mssd=np.concatenate((dist_seg_to_ref, dist_ref_to_seg)).max()
    return assd, mssd

def transformToRealCoordinates(indexPoints, affine):
    """
    This function transforms index points to the real world coordinates
    according to DICOM Patient-Based Coordinate System
    The source: DICOM PS3.3 2019a - Information Object Definitions page 499.

    In CHAOS challenge the orientation of the slices is determined by order
    of image names NOT by position tags in DICOM files. If you need to use
    real orientation data mentioned in DICOM, you may consider to use
    TransformIndexToPhysicalPoint() function from SimpleITK library.
    """
    M = affine
    realPoints=[]
    for i in range(len(indexPoints[0])):
        # TODO: check correctness
        # P=np.array([indexPoints[1,i],indexPoints[2,i],indexPoints[0,i],1])
        P=np.array([indexPoints[0,i],indexPoints[1,i],indexPoints[2,i],1])
        R=np.matmul(M,P)
        realPoints.append(R[0:3])

    return realPoints


def precision(ref=None, test=None, **metrics_kwargs):
    total_cm = metrics_kwargs.pop("total_cm")
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    precision = cm_diag / sum_over_row
    return precision


def recall(ref=None, test=None, **metrics_kwargs):
    total_cm = metrics_kwargs.pop("total_cm")
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    precision = cm_diag / sum_over_col
    return precision


def compute_mean_dsc(ref=None, test=None, **metrics_kwargs):
    """Compute the mean intersection-over-union via the confusion matrix."""
    total_cm = metrics_kwargs.pop("total_cm")
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    dscs = 2*cm_diag / denominator

    print('Dice Score Simililarity for each class:')
    for i, dsc in enumerate(dscs):
        print('    class {}: {:.4f}'.format(i, dsc))

    # If the number of valid entries is 0 (no classes) we return 0.
    m_dsc = np.where(
        num_valid_entries > 0,
        np.sum(dscs) / num_valid_entries,
        0)
    print('mean Dice Score Simililarity: {:.4f}'.format(float(m_dsc)))
    return m_dsc, dscs


def compute_mean_iou(ref, test, **metrics_kwargs):
    """Compute the mean intersection-over-union via the confusion matrix."""
    total_cm = metrics_kwargs.pop("total_cm")
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    ious = cm_diag / denominator

    print('Intersection over Union for each class:')
    for i, iou in enumerate(ious):
        print('    class {}: {:.4f}'.format(i, iou))

    # If the number of valid entries is 0 (no classes) we return 0.
    m_iou = np.where(
        num_valid_entries > 0,
        np.sum(ious) / num_valid_entries,
        0)
    print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))
    return m_iou


def compute_accuracy(ref, test, **metrics_kwargs):
    """Compute the accuracy via the confusion matrix."""
    total_cm = metrics_kwargs.pop("total_cm")
    denominator = total_cm.sum().astype(float)
    cm_diag_sum = np.diagonal(total_cm).sum().astype(float)

    # If the number of valid entries is 0 (no classes) we return 0.
    accuracy = np.where(
        denominator > 0,
        cm_diag_sum / denominator,
        0)
    print('Pixel Accuracy: {:.4f}'.format(float(accuracy)))
    return accuracy
