import os
import numpy as np
import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import file_utils, build_prior, dataset_infos

build_medical_images_prior = build_prior.build_medical_images_prior
_ISBI_CHAOS_INFORMATION_MR_T2 = dataset_infos._ISBI_CHAOS_INFORMATION_MR_T2

DATA_DIR = '/home/acm528_02/Jing_Siang/data/2019_ISBI_CHAOS/Train_Sets/CT/'
OUTPUT_DIR = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/'

NUM_SUBJECT = [1,10,20]
NUM_SUBJECT = [1, 2]
OUTPUT_PRIOR_SLICE = [1,2,4,6,8,10]
OUTPUT_PRIOR_SLICE = [1]

# TODO: In the end, most of the default should be require
parser = argparse.ArgumentParser()

parser.add_argument('--num_subject', nargs='+', type=int, default=NUM_SUBJECT,
                    help='')

parser.add_argument('--prior_slice', nargs='+', type=int, default=OUTPUT_PRIOR_SLICE,
                    help='')

parser.add_argument('--num_class', type=int, default=None,
                    help='The number of segmentation categories')

parser.add_argument('--modality', type=str, default=None,
                    help='The modality of data')

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
        # TODO: MR need resize image
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
    for sub in FLAGS.num_subject:
        for slices in FLAGS.prior_slice:
            prior_generator = build_chaos_prior(FLAGS.output_dir,
                                                sub,
                                                slices,
                                                FLAGS.num_class,
                                                FLAGS.modality,
                                                save_prior_in_npy=FLAGS.save_prior_in_npy,
                                                save_prior_in_img=FLAGS.save_prior_in_images)
            file_list = file_utils.get_file_list(FLAGS.data_dir, None, None)
            _ = prior_generator(file_list)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
    
    