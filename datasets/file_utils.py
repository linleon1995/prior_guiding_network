#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020
@author: Jing-Siang, Lin
"""

import os
import numpy as np
import SimpleITK as sitk
import cv2
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
        
def read_medical_images(file):
    image = sitk.ReadImage(file)
    image_arr = sitk.GetArrayFromImage(image)
    image_arr = np.int32(image_arr)
    return image_arr


def write_medical_images(imgs, out_dir, image_format, file_name="img", saving_data_type="3d"):
    if not isinstance(imgs, list):
        raise ValueError("imgs should be a list")

    if saving_data_type == "2d":
        for i, img in enumerate(imgs):
            out = sitk.GetImageFromArray(img)
            sitk.WriteImage(out, os.path.join(out_dir, file_name, str(i).zfill(4), saving_data_type))
    elif saving_data_type == "3d":
        imgs = np.stack(imgs, axis=0)
        print(np.shape(imgs))
        out = sitk.GetImageFromArray(imgs)
        sitk.WriteImage(out, os.path.join(out_dir, file_name+image_format))
    else:
        raise ValueError("Unknown saving_data_type")


def get_file_list(path, fileStr=[], fileExt=[], sort_files=True, file_idx=None):
    file_list = []
    def get_judge(key):
        if isinstance(key, str):
            key = [key]
        judge = True
        if key is not None:
            if len(key) == 0:
                judge = False
        else:
            judge = False
        return judge, key
    
    str_judge, fileStr = get_judge(fileStr)
    ext_judge, fileExt = get_judge(fileExt)
    
    for f in os.listdir(path):
        Str, Ext = False, False
        if str_judge:
            for file_start in fileStr:
                if f.startswith(file_start):
                    Str = True
                    break
        else:
            Str = True
        
        if ext_judge:
            for file_end in fileExt:
                if f.endswith(file_end):
                    Ext = True
                    break
        else:
            Ext = True

        if Str and Ext:
            file_list.append(os.path.join(path,f))
                   
    if len(file_list) == 0:
        raise ValueError("No file exist in %s" %path)

    if file_idx is not None:
        tmp = [file_list[i] for i in file_idx]
        file_list = tmp
        
    if sort_files:
        file_list.sort()
    return file_list


def convert_label_value(data, convert_dict):
    for k, v in convert_dict.items():
        data[data==k] = v
    return data


def save_in_image(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, file_name), data)
    
    