#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:05:16 2019

@author: acm528_02
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
class_to_organ = {0: "background", 1: "spleen", 2: "right kidney", 3: "left kidney", 4: "gallblader", 
                  5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta", 9: "IVC", 
                  10: "PS", 11: "pancreas", 12: "RAG", 13: "LAG"}
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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
    
    # 设置图形的显示风格
    # plt.style.use('ggplot')

    # 设置中文和负号正常显示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 绘图：整体乘客的年龄箱线图
    ax[0].boxplot(x = np.arange(2,20), # 指定绘图数据
                patch_artist=False, # 要求用自定义颜色填充盒形图，默认白色填充
                showmeans=True, # 以点的形式显示均值
                boxprops = {'color':'black'}, # 设置箱体属性，填充色和边框色
                flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
                meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色
                medianprops = {'linestyle':'-','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
    # 设置y轴的范围
    plt.ylim(0,85)

    # 去除箱线图的上边框与右边框的刻度标签
    plt.tick_params(top='off', right='off', left='off')
    # 显示图形
    plt.show()


def compute_params_and_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print("FLOPs: {}; GFLOPs: {}".format(flops.total_float_ops, flops.total_float_ops / 1e9))
    print("Trainable params:{} MB".format(params.total_parameters/(1024**2)))
    return flops, params


def save_evaluation():
    pass

def compute_mean_dsc(total_cm):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = np.sum(total_cm, axis=0).astype(float)
      sum_over_col = np.sum(total_cm, axis=1).astype(float)
      cm_diag = np.diagonal(total_cm).astype(float)
      denominator = sum_over_row + sum_over_col
    
      # The mean is only computed over classes that appear in the
      # label or prediction tensor. If the denominator is 0, we need to
      # ignore the class.
      num_valid_entries = np.sum((denominator != 0).astype(float))
    
      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = np.where(
          denominator > 0,
          denominator,
          np.ones_like(denominator))
    
      dscs = 2*cm_diag / denominator
    
      print('Dice Score Simililarity for each class:')
      for i, dsc in enumerate(dscs):
        print('    class {}: {:.4f}'.format(i, dsc))
    
      # If the number of valid entries is 0 (no classes) we return 0.
      m_dsc = np.where(
          num_valid_entries > 0,
          np.sum(dscs) / num_valid_entries,
          0)
      m_dsc = float(m_dsc)
      print('mean Dice Score Simililarity: {:.4f}'.format(float(m_dsc)))
      return m_dsc, dscs
  
def precision_and_recall(total_cm):
    """"""    
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    precision = cm_diag / sum_over_row
    recall = cm_diag / sum_over_col
    
    print('Recall and Precision')
    print(30*"=")
    i = 0
    for p, r in zip(precision, recall):
        print('    class {}: precision: {:.4f}  recall: {:.4f}'.format(i, p, r))
        i += 1
    p_mean = np.mean(precision)
    p_std = np.std(precision)
    r_mean = np.mean(recall)
    r_std = np.std(recall)
    print('    precision: mean {:.4f}  std {:.4f}'.format(p_mean, p_std))
    print('    recall: mean {:.4f}  std {:.4f}'.format(r_mean, r_std))
    return  precision, recall


def compute_mean_iou(total_cm):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = np.sum(total_cm, axis=0).astype(float)
      sum_over_col = np.sum(total_cm, axis=1).astype(float)
      cm_diag = np.diagonal(total_cm).astype(float)
      denominator = sum_over_row + sum_over_col - cm_diag
    
      # The mean is only computed over classes that appear in the
      # label or prediction tensor. If the denominator is 0, we need to
      # ignore the class.
      num_valid_entries = np.sum((denominator != 0).astype(float))
    
      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = np.where(
          denominator > 0,
          denominator,
          np.ones_like(denominator))
    
      ious = cm_diag / denominator
    
      print('Intersection over Union for each class:')
      for i, iou in enumerate(ious):
        print('    class {}: {:.4f}'.format(i, iou))
    
      # If the number of valid entries is 0 (no classes) we return 0.
      m_iou = np.where(
          num_valid_entries > 0,
          np.sum(ious) / num_valid_entries,
          0)
      m_iou = float(m_iou)
      print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))
      return m_iou


def compute_accuracy(total_cm):
      """Compute the accuracy via the confusion matrix."""
      denominator = total_cm.sum().astype(float)
      cm_diag_sum = np.diagonal(total_cm).sum().astype(float)

      # If the number of valid entries is 0 (no classes) we return 0.
      accuracy = np.where(
          denominator > 0,
          cm_diag_sum / denominator,
          0)
      accuracy = float(accuracy)
      print('Pixel Accuracy: {:.4f}'.format(float(accuracy)))
      return accuracy
    
    
def load_model(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def plot_confusion_matrix(cm, classes,
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
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()  
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "CM.png"))
    else:
        plt.show()