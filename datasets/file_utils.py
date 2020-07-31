import os
import numpy as np
import SimpleITK as sitk
import cv2

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
    
def get_file_list(path, fileStr=[], fileExt=[], sort_files=True, num_file=None, exc_key="$$"):
    # TODO: fileStr, fileExt are empty list, string, None condition
    # TODO: make exclusive key right
    file_list = []
    for _ in os.listdir(path):
        for file_start in fileStr:
            for file_end in fileExt:
                if _.startswith(file_start) and _.endswith(file_end) and exc_key not in _:
                    file_list.append(os.path.join(path,_))
                    
                
    if len(file_list) == 0:
        raise ValueError("No file exist")  
    
    # Determine the number of files to load
    if sort_files:
        file_list.sort()
    if num_file is not None:
        if num_file > len(file_list):
            raise ValueError("Out of Range Error") 
        file_list = file_list[0:num_file]
        
    return file_list


def convert_label_value(data, convert_dict):
    # TODO: optimize
    for k, v in convert_dict.items():
        data[data==k] = v
    return data
 
 
def save_in_image(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, file_name), data)