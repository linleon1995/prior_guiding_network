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
from medpy import metric
from mpl_toolkits.mplot3d import Axes3D
"""
def evaluate(Vref,Vseg,dicom_dir):
    dice=DICE(Vref,Vseg)
    ravd=RAVD(Vref,Vseg)
    return dice, ravd
"""
# TODO: Align coordinate between Simple iTK and nibabel
def show_3d(ref, seg):
    Rx, Ry, Rz = [], [] , []
    Sx, Sy, Sz = [], [] , []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for r, s in zip(ref, seg):       
        Rx.append(r[0])
        Ry.append(r[1])
        Rz.append(r[2])
        
        Sx.append(s[0])
        Sy.append(s[1])
        Sz.append(s[2])
        
    ax.scatter(Rx, Ry, Rz, c='r', marker='.')
    ax.scatter(Sx, Sy, Sz, c='g', marker='.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    
def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)
    
    
def evaluate(Vref,Vseg,dicom_dir, j, ax):
    dice=DICE(Vref,Vseg)
    ravd=RAVD(Vref,Vseg)
    [assd, mssd]=SSD(Vref,Vseg,dicom_dir, j, ax)
    return dice, ravd, assd ,mssd

def DICE(Vref,Vseg):
    dice=2*(Vref & Vseg).sum()/(Vref.sum() + Vseg.sum())
    return dice

def RAVD(Vref,Vseg):
    ravd=(abs(Vref.sum() - Vseg.sum())/Vref.sum())*100
    return ravd

def SSD(Vref,Vseg,dicom_dir, j, ax):  
    struct = ndimage.generate_binary_structure(3, 1)  
    
    ref_border=Vref ^ ndimage.binary_erosion(Vref, structure=struct, border_value=1)
    ref_border_voxels=np.array(np.where(ref_border))
    # print(ref_border_voxels.shape, ref_border.shape)
    """
    print(struct, ref_border)
    for i in range(3):
      for j in range(3):
        for k in range(3):
          print(struct[i,j,k])
    print(struct.shape, ref_border.shape)    
    import matplotlib.pyplot as plt
    plt.imshow(np.int32(ref_border[30]), "gray")
    plt.show()
    plt.imshow(np.int32(Vref[30]), "gray")
    plt.show()
    plt.imshow(np.int32(ndimage.binary_erosion(Vref, structure=struct, border_value=1)[30]), "gray")
    plt.show()
    """
    
    seg_border=Vseg ^ ndimage.binary_erosion(Vseg, structure=struct, border_value=1)
    seg_border_voxels=np.array(np.where(seg_border))  
    
    ref_border_voxels_real=transformToRealCoordinates(ref_border_voxels,dicom_dir)
    seg_border_voxels_real=transformToRealCoordinates(seg_border_voxels,dicom_dir)    
    
    
    show_3d(ref_border_voxels_real, seg_border_voxels_real)
      
    tree_ref = KDTree(np.array(ref_border_voxels_real))
    dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels_real)
    tree_seg = KDTree(np.array(seg_border_voxels_real))
    dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels_real)   
    
    assd=(dist_seg_to_ref.sum() + dist_ref_to_seg.sum())/(len(dist_seg_to_ref)+len(dist_ref_to_seg))
    mssd=np.concatenate((dist_seg_to_ref, dist_ref_to_seg)).max()    
    print(np.std(dist_seg_to_ref), np.std(dist_ref_to_seg))
    return assd, mssd

def transformToRealCoordinates(indexPoints,dicom_dir):
    """
    This function transforms index points to the real world coordinates
    according to DICOM Patient-Based Coordinate System
    The source: DICOM PS3.3 2019a - Information Object Definitions page 499.
    
    In CHAOS challenge the orientation of the slices is determined by order
    of image names NOT by position tags in DICOM files. If you need to use
    real orientation data mentioned in DICOM, you may consider to use
    TransformIndexToPhysicalPoint() function from SimpleITK library.
    """
    M = dicom_dir.affine.copy()
    print(M)
    realPoints=[]
    for i in range(len(indexPoints[0])):
        # P=np.array([indexPoints[1,i],indexPoints[2,i],indexPoints[0,i],1])
        P=np.array([indexPoints[0,i],indexPoints[1,i],indexPoints[2,i],1])
        R=np.matmul(M,P)
        realPoints.append(R[0:3])
 
    return realPoints

def png_series_reader(dir):
    V = []
    png_file_list=glob.glob(dir + '/*.png')
    png_file_list.sort()
    for filename in png_file_list: 
        image = cv2.imread(filename,0)
        V.append(image)
    V = np.array(V,order='A')
    V = V.astype(bool)
    return V

if __name__ == '__main__':

    
    # ======= Directories =======
    # cwd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    cwd = os.path.normpath(os.getcwd() + os.sep)
    print(cwd)
    ground_dir = os.listdir("/home/user/DISK/data/Jing/data/2015_MICCAI_BTCV/Train_Sets/label/")
    seg_dir = os.listdir("/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_057/model.ckpt-best/nii_files_val/")
    dicom_dir = os.listdir("/home/user/DISK/data/Jing/data/2015_MICCAI_BTCV/Train_Sets/raw/")
    ground_dir.sort()
    dicom_dir.sort()
    seg_dir.sort()
    ground_dir = ground_dir[24:]
    dicom_dir = dicom_dir[24:]
    print(ground_dir)
    # ======= Volume Reading =======
    def read_medical_images(file):
      # image = sitk.ReadImage(file)
      # image_arr = sitk.GetArrayFromImage(image)
      image_arr = nib.load(file).get_data()
      image_arr = np.int32(image_arr)
      
      return image_arr
    def r(image_path):
      # Read the medical image (.nii.gz .dcm) containing the volume with SimpleITK
      image = sitk.ReadImage(image_path)
      # Access the numpy array:
      image_arr = sitk.GetArrayFromImage(image)
      return np.int32(image_arr)
      
    total_dd, total_rr, total_aa, total_mm = [], [], [], []
    for i in range(6):
      dd, rr, aa, mm = [], [], [], []
      Vref = read_medical_images("/home/user/DISK/data/Jing/data/2015_MICCAI_BTCV/Train_Sets/label/"+ground_dir[i])
      Vseg = read_medical_images("/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_057/model.ckpt-best/nii_files_val/"+seg_dir[i]) 
      ddd = nib.load("/home/user/DISK/data/Jing/data/2015_MICCAI_BTCV/Train_Sets/raw/"+dicom_dir[i]) 
      # plt.imshow(Vref[...,60])
      # plt.show()
      # dd2 = r("/home/user/DISK/data/Jing/data/2015_MICCAI_BTCV/Train_Sets/raw/"+dicom_dir[i])
      # print(dd2.shape, ddd.shape)
      """
      x = 60
      plt.imshow(Vref[...,x])
      plt.show()
      plt.imshow(Vseg[...,x])
      plt.show()
      plt.imshow(ddd.get_data()[...,x])
      plt.show()
      print(Vref.shape, Vseg.shape, ddd.get_data().shape)
      """
      # hausdorff_distance(test=Vseg, reference=Vref, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs)
      
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for j in range(1,13):
        v_ref = Vref
        v_ref = np.int32(v_ref==j)
        v_seg = Vseg
        v_seg = np.int32(v_seg==j)
        print('Volumes imported.')
        # ======= Evaluation =======
        print('Calculating...')
        if np.sum(v_seg) == 0 or np.sum(v_ref) == 0:
          continue
        print(v_ref.shape)
        
        [dice, ravd, assd ,mssd]=evaluate(v_ref,v_seg,ddd, j, ax)
        print('Class%d DICE=%.3f RAVD=%.3f ASSD=%.3f MSSD=%.3f' %(j, dice, ravd, assd ,mssd))
        dd.append(dice)
        rr.append(ravd)
        aa.append(assd)
        mm.append(mssd)
        print('DICE=%.3f RAVD=%.3f ASSD=%.3f MSSD=%.3f' %(sum(dd)/len(dd), sum(rr)/len(rr), sum(aa)/len(aa), sum(mm)/len(mm)))
      total_dd.append(sum(dd)/len(dd))    
      total_rr.append(sum(rr)/len(rr)) 
      total_aa.append(sum(aa)/len(aa)) 
      total_mm.append(sum(mm)/len(mm)) 
      print('DICE=%.3f RAVD=%.3f ASSD=%.3f MSSD=%.3f' %(sum(total_dd)/len(total_dd), sum(total_rr)/len(total_rr), sum(total_aa)/len(total_aa), sum(total_mm)/len(total_mm)))
      
      
      