import os
import numpy as np
import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
import file_utils, build_prior

build_medical_images_prior = build_prior.build_medical_images_prior

DATA_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/label/'
OUTPUT_DIR = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/2013_MICCAI_BTCV/'

NUM_SUBJECT = [1,10,20]
NUM_SUBJECT = [1, 2]
OUTPUT_PRIOR_SLICE = [1,2,4,6,8,10]
OUTPUT_PRIOR_SLICE = [1]

parser = argparse.ArgumentParser()

parser.add_argument('--num_subject', nargs='+', type=int, default=NUM_SUBJECT,
                    help='')

parser.add_argument('--prior_slice', nargs='+', type=int, default=OUTPUT_PRIOR_SLICE,
                    help='')

parser.add_argument('--num_class', type=int, default=14,
                    help='The number of segmentation categories')

parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                    help='MICCAI 2013 dataset root folder.')

parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                    help='')

parser.add_argument('--remove_zeros', type=str, default=True,
                    help='The boolean flag for removing zero slices')

parser.add_argument('--save_prior_in_npy', type=bool, default=True,
                    help='')

parser.add_argument('--save_prior_in_images', type=bool, default=True,
                    help='')


class build_miccai_prior(build_medical_images_prior):
    def load_data_func(self, f):
        image_arr = super().load_data_func(f)
        image_arr = image_arr[:,::-1]
        image_arr = np.swapaxes(image_arr, 0, 1)
        image_arr = np.swapaxes(image_arr, 1, 2)
        return image_arr


def main(unused_argv):
    for sub in FLAGS.num_subject:
        for slices in FLAGS.prior_slice:
            prior_generator = build_miccai_prior(FLAGS.output_dir,
                                                sub,
                                                slices,
                                                FLAGS.num_class,
                                                save_prior_in_npy=FLAGS.save_prior_in_npy,
                                                save_prior_in_img=FLAGS.save_prior_in_images)
            file_list = file_utils.get_file_list(path=FLAGS.data_dir,
                                                 fileStr=None,
                                                 fileExt=["nii.gz"])
            _ = prior_generator(file_list)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)