#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020
@author: Jing-Siang, Lin
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy import ndimage
from evals import metrics

    
def show_3d(volume_dict, affine, file_name=None):
    """
    Scatter plot for 3d visualization
    Args:
        volume_dict: Input dictionary contains all volumes that need to be visualized. 
                     The key of volume_dict is the label of the plot, and the value of 
                     volume_dict is a 3d NumPy array.
        affine: Affine transform parameters extracted from raw data which helps to 
                transform volume to real-world coordinate
        file_name: The file name for figure saving
    """
    struct = ndimage.generate_binary_structure(3, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for label, v in volume_dict.items():
        # Access border points to get better running performance
        border=v ^ ndimage.binary_erosion(v, structure=struct, border_value=1)
        border_voxels=np.array(np.where(border))  
        border_voxels_real=metrics.transformToRealCoordinates(border_voxels,affine)
        Sx, Sy, Sz = [], [], []
        for s in  border_voxels_real:
            Sx.append(s[0])
            Sy.append(s[1])
            Sz.append(s[2])
        ax.scatter(Sx, Sy, Sz, marker='.', label=label)
    ax.legend()
    if file_name is not None:
        plt.savefig(file_name+".png")
    plt.show()
    
    
class Build_Pyplot_Subplots(object):
    def __init__(self, saving_path, is_showfig, is_savefig, subplot_split, type_list):
        self.fig, self.ax = plt.subplots(figsize=(9,3),*subplot_split)
        # TODO: figsize decided by subplot_split
        plt.tight_layout()
        # self.fig.figsize = (6,2)
        self.x_axis = subplot_split[0]
        self.y_axis = subplot_split[1]
        self.saving_path = saving_path
        self.is_showfig = is_showfig
        self.is_savefig = is_savefig
        self.type_list = type_list
        self.num_plot = np.prod(self.ax.shape)
        assert self.num_plot == len(type_list)
        # TODO: color_map
        # TODO: condition of is_showfig and issavefig
        # TODO: remove coreordinate
        # TODO: Solve subplots(2,2)--> ax is a numpy array not suitble for using for loop directly
        # TODO: Make a judgement about existence of directory

    def set_title(self, title_list):
        assert self.num_plot == len(title_list)
        # for x in range(self.x_axis):
        #     for y in range(self.y_axis):
        #         sub_ax.set_title(title)
        for sub_ax, title in zip(self.ax, title_list):
            sub_ax.set_title(title)

    def set_axis_off(self,):
        for sub_ax in self.ax:
            sub_ax.set_axis_off()

    def display_figure(self, file_name, value_list, parameters=None, saved_format='.png'):
        assert self.num_plot == len(value_list)
        if parameters is None:
            parameters = self.num_plot*[None]

        for sub_ax, plot_type, value, p in zip(self.ax, self.type_list, value_list, parameters):
            if plot_type == "plot":
                if p is not None:
                    sub_ax.plot(*value, **p)
                else:
                    sub_ax.plot(*value)
            elif plot_type == "img":
                if p is not None:
                    sub_ax.imshow(value, **p)
                else:
                    sub_ax.imshow(value)
            else:
                raise ValueError("Unkown ploting type")
        if self.is_showfig:
            plt.show()
        if self.is_savefig:
            self.fig.savefig(self.saving_path+file_name+saved_format)

        plt.close(self.fig)


def compute_params_and_flops(graph):
    """Measure the model parameter and operation"""
    flops_proto = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params_proto = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    flops = flops_proto.total_float_ops
    params = params_proto.total_parameters/(1024**2)
    print("FLOPs: {}; GFLOPs: {}".format(flops, flops/1e9))
    print("Trainable params:{} MB".format(params))
    return flops, params


def load_model(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def plot_confusion_matrix(cm, 
                          num_class, 
                          filename='Confusion_Matrix',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(num_class))
    plt.xticks(tick_marks, num_class, rotation=45)
    plt.yticks(tick_marks, num_class)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, filename+".png"))
    else:
        plt.show()


def inference_segmentation(logits, dim):
    """To get classification result by using softmax and argmax"""
    prediction = tf.nn.softmax(logits, axis=dim)
    prediction = tf.argmax(prediction, axis=dim)
    prediction = tf.cast(prediction, tf.int32)
    return prediction


def get_label_range(image):
    """
    This function help to make user understand the nonzero range of input image.
    By given one image, the function will extract the height and with of nonzero range.
    Args:
        image: image in shape [height,width] or [height,width,1].
    Returns:
        nonzero_range: The List contains nonzero range of input image.
    """
    if len(np.shape(image)) == 3:
        image = image[...,0]
    elif len(np.shape(image)) == 2:
        pass
    else:
        raise ValueError("Unknown image shape")

    if np.sum(image) == 0:
        nonzero_range = 4*[0]
    else:
        fg = np.where(image!=0)
        h_min = np.min(fg[0])
        h_max = np.max(fg[0])
        w_min = np.min(fg[1])
        w_max = np.max(fg[1])
        nonzero_range = [h_min,w_min,h_max,w_max]
    return nonzero_range

