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
class_to_organ = {0: "background", 1: "spleen", 2: "right kidney", 3: "left kidney", 4: "gallblader",
                  5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta", 9: "IVC",
                  10: "PS", 11: "pancreas", 12: "RAG", 13: "LAG"}


def print_checkpoint_tensor_name(checkpoint_dir):
    # TODO: all_tensors,  all_tensor_names for parameters
    """Print all tensor name from graph"""
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    # List ALL tensors example output
    print_tensors_in_checkpoint_file(file_name=checkpoint_dir, tensor_name='', all_tensors=False,
                                    all_tensor_names=True)


def eval_flol_model(dsc_in_diff_th, threshold):
    """
    """
    ll = []
    _, ax = plt.subplots(1,1)
    num_class = len(dsc_in_diff_th)
    for i, c in enumerate(dsc_in_diff_th):
        if i < num_class-1:
            label = class_to_organ[i+1]
        else:
            label = "mean"
        if i+1 > 10:
            ax.plot(threshold, c, "*-", label=label)
        else:
            ax.plot(threshold, c, label=label)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    ax.set_xticks(threshold, minor=False)
    ax.grid(True)
    ax.set_title("Threshold to DSC for each calss")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Dice Score")
    plt.show()


# def show_3d(ref, seg):
#     Rx, Ry, Rz = [], [] , []
#     Sx, Sy, Sz = [], [] , []
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     for r, s in zip(ref, seg):
#         Rx.append(r[0])
#         Ry.append(r[1])
#         Rz.append(r[2])

#         Sx.append(s[0])
#         Sy.append(s[1])
#         Sz.append(s[2])

#     ax.scatter(Rx, Ry, Rz, c='r', marker='.')
#     ax.scatter(Sx, Sy, Sz, c='g', marker='.')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()

    
def show_3d(volume_dict, affine, file_name=None):
    """
    Scatter plot for 3d visualization
    Args:
        volume_dict: Input dictionary contains all volumes that need to be visualized. 
                     The key of volume_dict is the label of the plot, and the value of 
                     volume_dict is a 3d NumPy array.
        affine: Affine transform parameters extracted from raw data which helps to 
                transform volume to real-world coordinate
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

def plot_histogram(path):
    pass

def plot_box_diagram(path):
    # spread = np.random.rand(50) * 100
    # center = np.ones(25) * 50
    # flier_high = np.random.rand(10) * 100 + 100
    # flier_low = np.random.rand(10) * -100
    # data = np.concatenate((spread, center, flier_high, flier_low))

    # # Fixing random state for reproducibility
    # # np.random.seed(19680801)

    fig1, ax = plt.subplots(1,4)
    # ax1.set_title('Basic Plot')
    # ax1.boxplot(data)

    # ?置?形的?示?格
    # plt.style.use('ggplot')

    # ?置中文和??正常?示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # ??：整体乘客的年?箱??
    ax[0].boxplot(x = np.arange(2,20), # 指定???据
                patch_artist=False, # 要求用自定??色填充盒形?，默?白色填充
                showmeans=True, # 以?的形式?示均值
                boxprops = {'color':'black'}, # ?置箱体?性，填充色和?框色
                flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # ?置异常值?性，?的形?、填充色和?框色
                meanprops = {'marker':'D','markerfacecolor':'indianred'}, # ?置均值?的?性，?的形?、填充色
                medianprops = {'linestyle':'-','color':'orange'}) # ?置中位??的?性，?的?型和?色
    # ?置y?的范?
    plt.ylim(0,85)

    # 去除箱??的上?框与右?框的刻度??
    plt.tick_params(top='off', right='off', left='off')
    # ?示?形
    plt.show()


def compute_params_and_flops(graph):
    flops_proto = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params_proto = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    flops = flops_proto.total_float_ops
    params = params_proto.total_parameters/(1024**2)
    print("FLOPs: {}; GFLOPs: {}".format(flops, flops/1e9))
    print("Trainable params:{} MB".format(params))
    return flops, params


def save_evaluation():
    pass

def display_classify_performance(label, pred):
    """
    label: binary mask in shape [N,H,W,C]
    pred: corresponding shape and data_type with label
    3: tp, 2: fp, 1: fn, 0: tn
    """
    return label + 2*pred


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
    prediction = tf.nn.softmax(logits, axis=dim)
    # prediction = tf.identity(prediction, name=common.OUTPUT_TYPE)
    prediction = tf.argmax(prediction, axis=dim)
    prediction = tf.cast(prediction, tf.int32)
    return prediction


def get_label_range(label, height, width):
    """
    HW1, HW
    """
    if len(np.shape(label)) == 3:
        label = label[...,0]
    elif len(np.shape(label)) == 2:
        pass
    else:
        raise ValueError("Unknown label shape")

    if np.sum(label) == 0:
        return 4*[0]
    else:
        fg = np.where(label!=0)
        h_min = np.min(fg[0])
        h_max = np.max(fg[0])
        w_min = np.min(fg[1])
        w_max = np.max(fg[1])

        # h_min = np.min(np.min(fg, axis=0))
        # w_min = np.min(np.min(fg, axis=1))
        # h_max = np.max(np.max(fg, axis=0))
        # w_max = np.max(np.max(fg, axis=1))
        return [h_min,w_min,h_max,w_max]

