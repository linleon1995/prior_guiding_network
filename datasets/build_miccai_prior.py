#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020
@author: Jing-Siang, Lin
"""


import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
import file_utils
import build_prior
import dataset_infos
build_medical_images_prior = build_prior.build_medical_images_prior

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='2015 MICCAI BTCV dataset root folder.')

parser.add_argument('--output_dir', type=str, required=True,
                    help='Path to save BTCV dataset prior.')

parser.add_argument('--num_subject', nargs='+', type=int, required=True,
                    help='The number of subject join the prior')

parser.add_argument('--prior_slice', nargs='+', type=int, required=True,
                    help='The number of slice (level) for prior third dimension')

parser.add_argument('--remove_zeros', type=str, default=True,
                    help='The boolean flag for removing zero slices')

parser.add_argument('--save_prior_in_npy', type=bool, default=True,
                    help='')

parser.add_argument('--save_prior_in_images', type=bool, default=True,
                    help='')


class build_miccai_prior(build_medical_images_prior):
    def load_data_func(self, f):
        print(f)
        image_arr = super().load_data_func(f)
        image_arr = image_arr[:,::-1]
        image_arr = np.swapaxes(image_arr, 0, 1)
        image_arr = np.swapaxes(image_arr, 1, 2)
        return image_arr


def main(unused_argv):
    for sub in FLAGS.num_subject:
        for slices in FLAGS.prior_slice:
            prior_generator = build_miccai_prior(output_dir=FLAGS.output_dir,
                                                 num_subject=sub,
                                                 num_slice=slices,
                                                 num_class=dataset_infos._MICCAI_ABDOMINAL_INFORMATION.num_classes,
                                                 save_prior_in_npy=FLAGS.save_prior_in_npy,
                                                 save_prior_in_img=FLAGS.save_prior_in_images)
            data_dir = os.path.join(FLAGS.data_dir, "Train_Sets", "label/")                                     
            file_list = file_utils.get_file_list(path=data_dir,
                                                 fileStr=None,
                                                 fileExt=None)
            _ = prior_generator(file_list)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)