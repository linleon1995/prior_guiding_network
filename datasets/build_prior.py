import os 
import numpy as np
import nibabel as nib
import argparse

DATA_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/label/'
OUTPUT_DIR = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/'

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                    help='MICCAI 2013 dataset root folder.')

parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                    help='')

parser.add_argument('--num_subject', type=str, default=1,
                    help='The number of training data be used')

parser.add_argument('--num_class', type=str, default=14,
                    help='The number of segmentation categories')


def merge_training_seg(data_list):
    # TODO: Merge correctly
    data = data_list[0]
    data = np.float32(data)
    merge_data = np.sum(data, axis=0)
    normalized_term = np.max(np.max(merge_data, axis=0), axis=0)
    merge_data /= normalized_term[np.newaxis,np.newaxis]
    return merge_data

    
def load_nibabel_data_v2(file_list, num_of_class=None):
    # TODO: use SimpleITK to replace nibabel
    # seg_itk = sitk.ReadImage(seg_file)
    # seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    # TODO: Rethinking the design
    imgs = []
    for f in file_list:
        sample = nib.load(f).get_data()
        sample = np.flip(np.swapaxes(sample, 0, -1), 1)
        if num_of_class is not None:
            sample = np.eye(num_of_class)[sample]
            sample = np.uint8(sample)
        imgs.append(sample)
    return imgs
    
    
def get_file_list(path, num_file=None, file_format="nii"):
    # Check existence and number of files
    file_list = os.listdir(path)
    new_file_list = []
    for i, f in enumerate(file_list):
        if os.path.isfile(os.path.join(path, f)) and f.split(".")[1]==file_format:
            new_file_list.append(os.path.join(path, f))
    file_list = new_file_list
           
    if len(file_list) == 0:
        raise ValueError("No file exist")  
    
    # Determine the data for building prior
    file_list.sort()
    if num_file is not None:
        if num_file > len(file_list):
            raise ValueError("Out of Range Error") 
        file_list = file_list[0:num_file]
        
    return file_list


def build_priors(data_dir, output_dir, num_subject=None, num_of_class=None):
    file_list = get_file_list(data_dir, num_subject) 
    data_list = load_nibabel_data_v2(file_list, num_of_class)
    priors = merge_training_seg(data_list)
    
    # Check prior shape
    assert len(priors.shape) == 3
    
    np.save(os.path.join(output_dir, 
    "training_seg_merge_"+str(num_subject).zfill(3)+".npy"), priors)    
    
    
def main(unused_argv):
    build_priors(data_dir=FLAGS.data_dir, 
                 output_dir=FLAGS.output_dir,
                 num_subject=FLAGS.num_subject, 
                 num_of_class=FLAGS.num_class)    
    
    
if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)