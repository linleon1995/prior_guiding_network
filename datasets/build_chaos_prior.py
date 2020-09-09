#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020
@author: Jing-Siang, Lin
"""

import os
import numpy as np
import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import file_utils
import build_prior
import dataset_infos
build_medical_images_prior = build_prior.build_medical_images_prior
_ISBI_CHAOS_INFORMATION_MR_T2 = dataset_infos._ISBI_CHAOS_INFORMATION_MR_T2


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='2019 ISBI CHAOS dataset root folder.')

parser.add_argument('--output_dir', type=str, required=True,
                    help='Path to save CHAOS dataset prior.')

parser.add_argument('--num_subject', nargs='+', type=int, required=True,
                    help='The number of subject join the prior')

parser.add_argument('--prior_slice', nargs='+', type=int, required=True,
                    help='The number of slice (level) for prior third dimension')

parser.add_argument('--modality', nargs='+', required=True,
                    help='The modality of data')

parser.add_argument('--remove_zeros', type=str, default=True,
                    help='The boolean flag to decide whether remove zeros slices (no foreground)')

parser.add_argument('--save_prior_in_npy', type=bool, default=True,
                    help='')

parser.add_argument('--save_prior_in_images', type=bool, default=True,
                    help='')


class build_chaos_prior(build_medical_images_prior):
    def __init__(self, output_dir, num_subject, num_slice, num_class, modality, save_prior_in_npy=True,
                 save_prior_in_img=False, remove_zeros=True):
        self.modality = modality
        output_dir = os.path.join(output_dir, self.modality)
        super().__init__(
            output_dir, num_subject, num_slice, num_class, save_prior_in_npy, save_prior_in_img, remove_zeros)

    def load_data_func(self, f):
        image_arr = load_chaos_data(f, self.modality)
        return image_arr


def load_chaos_data(path, modality):
    def load_data(path):
        file_list = file_utils.get_file_list(path)
        data = []
        for f in file_list:
            data.append(file_utils.read_medical_images(f))
        return np.stack(data, axis=2)

    if modality == "CT":
        path = os.path.join(path, "Ground")
    elif modality == "MR_T1":
        path = os.path.join(path, "T1DUAL", "Ground")
    elif modality == "MR_T2":
        path = os.path.join(path, "T2SPIR", "Ground")
    else:
        raise ValueError("Unknown Data Moality")

    img = load_data(path)

    # TODO: convert label,  combine with build_chaos_data.py
    if modality == "CT":
        convert_dict = {255: 1}
    elif modality in ("MR_T1", "MR_T2"):
        img = cv2.resize(img, (_ISBI_CHAOS_INFORMATION_MR_T2.height,_ISBI_CHAOS_INFORMATION_MR_T2.width),
                         interpolation=cv2.INTER_NEAREST)
        convert_dict = {63: 1, 126: 2, 189: 3, 252: 4}

    img = file_utils.convert_label_value(img, convert_dict)
    return img


def main(unused_argv):
    for modality in FLAGS.modality:
        if modality == "CT":
            num_class = dataset_infos._ISBI_CHAOS_INFORMATION_CT.num_classes
        elif modality == "MR_T1":
            num_class = dataset_infos._ISBI_CHAOS_INFORMATION_MR_T1.num_classes
        elif modality == "MR_T2":
            num_class = dataset_infos._ISBI_CHAOS_INFORMATION_MR_T2.num_classes
        else:
            raise ValueError("Unknown data modality.")

        for sub in FLAGS.num_subject:
            for slices in FLAGS.prior_slice:
                prior_generator = build_chaos_prior(output_dir=FLAGS.output_dir,
                                                    num_subject=sub,
                                                    num_slice=slices,
                                                    num_class=num_class,
                                                    modality=modality,
                                                    save_prior_in_npy=FLAGS.save_prior_in_npy,
                                                    save_prior_in_img=FLAGS.save_prior_in_images)
                data_dir = os.path.join(FLAGS.data_dir, "Train_Sets", modality.split("_")[0])
                file_list = file_utils.get_file_list(data_dir, None, None)
                _ = prior_generator(file_list)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)

