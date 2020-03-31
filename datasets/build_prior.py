import os 
import numpy as np
import nibabel as nib
import argparse
import matplotlib.pyplot as plt

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

parser.add_argument('--num_slice', type=str, default=1,
                    help='The slice number (layer) of output prior')

parser.add_argument('--save_prior_in_npy', type=bool, default=True,
                    help='')

parser.add_argument('--save_prior_in_images', type=bool, default=True,
                    help='')

# TODO: Optimize
def np_onehot(data, num_class=None):
    # TODO: axis params
    if num_class is not None:
        num_class = np.max(data) + 1
    onehot_data = np.eye(num_class)[data]
    return onehot_data


def remove_zeros_slice(data):
    """
    one-hot data in specific class: [H,W,C]
    Return
        [class1, class2,..., classK]
        [[H,W,C1],[H,W,C2],...,[H,W,Ck]]
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


def merge_training_seg2(data_list, num_slice, num_class):
    """
    data_list: [data1, data2,..., dataN]
    data: [H,W,C]
    Returns:
        [H,W,C,K]
    """
    norm_data_list = []
    merge_list = []
    for idx, data in enumerate(data_list):
        print("Processing Data %03d" %idx)
        # Remove slice which not contain any organ
        onehot_data = np_onehot(data, num_class)
        norm_data_list = []
        for k in range(onehot_data.shape[3]):
            class_data = onehot_data[...,k]
            # Process when organ exist
            if np.sum(class_data) != 0:
                nonzero_data = remove_zeros_slice(class_data)
                norm_data = normalize_slice(nonzero_data, num_slice)
            else:
                norm_data = np.zeros(list(class_data.shape[:2])+[num_slice])
            norm_data_list.append(norm_data)    
        final_norm_data = np.stack(norm_data_list, axis=2)
        merge_list.append(final_norm_data)
    merge_data = sum(merge_list)   
    merge_data = normalize_value(merge_data)
    return merge_data


def merge_training_seg(data_list):
    # TODO: Merge correctly
    merge_data = 0
    for data in data_list:
        merge_data += np.sum(data, axis=0)

    merge_data = np.float32(merge_data)
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
        sample = np.transpose(sample[:,::-1], axes=[1,0,2])
        if num_of_class is not None:
            sample = np.eye(num_of_class)[sample]
            sample = np.uint8(sample)
        imgs.append(sample)
    return imgs
    
    
def get_file_list(path, num_file=None, file_format="nii"):
    # Check existence and number of files
    file_list = os.listdir(path)
    new_file_list = []
    for _, f in enumerate(file_list):
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


def get_prior_name(data_split, num_slice, num_subject):
    file_name = "-".join([data_split,
                "slice%03d" %num_slice,
                "subject%03d" %num_subject,])
    print("File nmae: %s" %file_name)
    return file_name


def build_priors(data_dir, output_dir, num_slice, num_of_class, 
                 num_subject=None, save_prior_in_npy=True, save_prior_in_images=True):
    # TODO: logging prior information
    # for sub in [1,10,20]:
    #     for slices in [2,4,6,8,10]:
    file_list = get_file_list(data_dir, num_subject) 
    data_list = load_nibabel_data_v2(file_list)
    priors = merge_training_seg2(data_list, num_slice, num_of_class)
    
    # Check prior shape
    assert len(priors.shape) == 4
    
    prior_img_dir = output_dir + "priors_img/"
    for i in range(num_slice):
        for j in range(num_of_class):
            if save_prior_in_images:
                plt.imshow(priors[:,:,j,i])
                # plt.show()
                img_file = get_prior_name("train", num_slice, num_subject)
                img_file = img_file + "-depth%03d" %i + "-class%03d" %j + ".png"
                plt.savefig(os.path.join(prior_img_dir, img_file))
    if save_prior_in_npy:
        file_name = get_prior_name("train", num_slice, num_subject)
        file_name = file_name + ".npy"
        np.save(os.path.join(output_dir, file_name), priors)    
    print("Finish Prior Building!")
    

def main(unused_argv):
    for sub in [1,10,20]:
        for slices in [1,2,4,6,8,10]:
            build_priors(data_dir=FLAGS.data_dir, 
                        output_dir=FLAGS.output_dir,
                        num_slice=slices,
                        num_of_class=FLAGS.num_class,
                        num_subject=sub, 
                        save_prior_in_npy=FLAGS.save_prior_in_npy,
                        save_prior_in_images=False)  
    # build_priors(data_dir=FLAGS.data_dir, 
    #              output_dir=FLAGS.output_dir,
    #              num_slice=FLAGS.num_slice,
    #              num_of_class=FLAGS.num_class,
    #              num_subject=FLAGS.num_subject, 
    #              save_prior_in_npy=FLAGS.save_prior_in_npy,
    #              save_prior_in_images=FLAGS.save_prior_in_images)    
    
    
if __name__ == '__main__':
    a = np.load("/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/train-slice002-subject001.npy")
    b=0
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)