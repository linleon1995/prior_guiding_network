import os
import numpy as np
import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
import file_utils

class build_medical_images_prior(object):
    def __init__(self, output_dir, num_subject, num_slice, num_class, save_prior_in_npy=True,
                 save_prior_in_img=False, remove_zeros=True):
        self.output_dir = output_dir
        self.img_output_dir = os.path.join(self.output_dir, "priors_img")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.img_output_dir):
            os.makedirs(self.img_output_dir, exist_ok=True)
        self.num_subject = num_subject
        self.num_slice = num_slice
        self.num_class = num_class
        self.remove_zeros = remove_zeros
        self.prior_name = get_prior_name(self.num_slice, self.num_subject)

    def __call__(self, file_list):
        prior = self.get_prior(file_list)
        self.save_prior_in_npy(prior)
        for i in range(self.num_slice):
            for j in range(self.num_class):
                self.img_name = self.prior_name + "-depth%03d-class%03d" %(i, j)
                img = np.int32(255*prior[:,:,j,i])
                self.save_prior_in_img(img)
        return prior

    def get_prior(self, file_list):
        return merge_training_seg(
            self.load_data_func, file_list, self.num_subject, self.num_slice, self.num_class, self.remove_zeros)

    def load_data_func(self, f):
        return file_utils.read_medical_images(f)

    def save_prior_in_npy(self, prior):
        np.save(os.path.join(self.output_dir, self.prior_name+".npy"), prior)

    def save_prior_in_img(self, img):
        file_name = os.path.join(self.img_output_dir, self.img_name+".png")
        cv2.imwrite(file_name, img)

    # def save_prior(self, prior):
    #     if not os.path.exists(self.output_dir):
    #         os.makedirs(self.output_dir, exist_ok=True)
    #     prior_name = get_prior_name(self.num_slice, self.num_subject)

    #     if self.save_prior_in_npy:
    #         np.save(os.path.join(self.output_dir, prior_name+".npy"), prior)

    #     if self.save_prior_in_img:
    #         img_output_dir = os.path.join(self.output_dir, "priors_img")
    #         if not os.path.exists(img_output_dir):
    #             os.makedirs(img_output_dir, exist_ok=True)

    #         for i in range(self.num_slice):
    #             for j in range(self.num_class):
    #                 img_name = prior_name + "-depth%03d-class%03d" %(i, j)
    #                 file_name = os.path.join(img_output_dir, img_name+".png")
    #                 cv2.imwrite(file_name, np.int32(255*prior[:,:,j,i]))


def get_prior_name(num_slice, num_subject):
    return "train-slice%03d-subject%03d" %(num_slice, num_subject)


def np_onehot(data, num_class=None):
    """Numpy based one-hot encoding in the last dimmension"""
    if num_class is not None:
        num_class = np.max(data) + 1
    onehot_data = np.eye(num_class)[data]
    return onehot_data


def remove_zeros_slice(data):
    """
    one-hot data in specific class: [H,W,C]
    Return
        [H,W,C'] which includes all non-zero frame
    """
    nonzero_idx = np.nonzero(np.sum(np.sum(data, axis=0), axis=0))
    nonzero_idx = nonzero_idx[0]
    start, end = nonzero_idx[0], nonzero_idx[-1]
    output = data[...,start:end+1]
    return output


def normalize_slice(data, num_slice):
    """
    data: [H,W,C]
    num_slice: C'
    Return:
        normalized_data: [H,W,C']
    """
    c = data.shape[2]
    seg = np.linspace(0, c+1, num_slice+1)
    seg = np.int32(seg)

    normalized_data = []
    # for i, _ in enumerate(seg):
    for i in range(seg.shape[0]-1):
        if i <= num_slice:
            region = data[...,seg[i]:seg[i+1]]
            norm_slice = np.sum(region, axis=2)

            normalized_data.append(norm_slice)
    normalized_data = np.stack(normalized_data, axis=2)
    return normalized_data


def normalize_value(data):
    normalized_term = np.max(np.max(data, axis=0), axis=0)
    data /= normalized_term[np.newaxis,np.newaxis]
    return data


def merge_training_seg(load_func, file_list, num_subject, num_slice, num_class, remove_zeros):
    """
    data_list: [data1, data2,..., dataN]
    data: [H,W,C]
    Returns:
        [H,W,C,K]
    """
    norm_data_list = []
    merge_list = []
    assert num_subject <= len(file_list)
    file_list = file_list[:num_subject]

    print("Prior Information: Subject number {}  Slice number {}  Class {}"
              .format(num_subject, num_slice, num_class))
    for idx, f in enumerate(file_list):
        print("Processing Data {}/{}".format(idx+1, len(file_list)))

        # Access data from dir
        data = load_func(f)

        # Remove slice which not contain any organ
        onehot_data = np_onehot(data, num_class)
        norm_data_list = []
        for k in range(onehot_data.shape[3]):
            class_data = onehot_data[...,k]
            # Process when organ exist
            if np.sum(class_data) != 0:
                if remove_zeros:
                    class_data = remove_zeros_slice(class_data)
                norm_data = normalize_slice(class_data, num_slice)
            else:
                norm_data = np.zeros(list(class_data.shape[:2])+[num_slice])
            norm_data_list.append(norm_data)
        final_norm_data = np.stack(norm_data_list, axis=2)
        merge_list.append(final_norm_data)
    merge_data = sum(merge_list)
    merge_data = normalize_value(merge_data)
    return merge_data
