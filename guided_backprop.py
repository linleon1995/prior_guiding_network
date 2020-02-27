#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:27:56 2019

@author: EE_ACM528_04
"""

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tf_tesis2  import unet_multi_task3, stn
from tf_tesis2.eval_utils import compute_mean_dsc, compute_mean_iou, compute_accuracy, load_model, plot_confusion_matrix
from tf_tesis2.dense_crf import crf_inference
#from tf_unet_multi_task  import util_multi_task 
import numpy as np
import CT_scan_util_multi_task
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tf_unet_multi_task import unet_multi_task 
#from tf_unet import unet_from_lab_server
#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import nibabel as nib
import glob

import argparse

N_CLASS = 14
RAW_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/raw/'
MASK_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/label/'
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_unet_multi_task/unet_mt_trained/unet32_sub25_labelmean/run_006/model.ckpt-55"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_008/model.ckpt-80"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_010/model.ckpt-40"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_022/model.ckpt-40"

#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_002/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_069/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_055/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_034/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_040/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_044/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_048/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_008/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/unet_pretrained/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_051/model.ckpt.best"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_077/model.ckpt.best"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_083/model.ckpt.best"

DATA_LIST_PATH = './dataset/val.txt'
IGNORE_LABEL = 255
#NUM_CLASSES = 14
NUM_STEPS = 1449 # Number of images in the validation set.
Z_CLASS = 3
SUBJECT_LIST = np.arange(25, 26)
SHUFFLE = False
IMG_SHAPE = [1, 256, 256, 1]
LABEL_SHAPE = [1, 256, 256, N_CLASS]
PI_ON_180 = 0.017453292519943295


#@tf.RegisterGradient("GuidedRelu")
#def _GuidedReluGrad(op, grad):
#    gate_g = tf.cast(grad > 0, "float32")
#    gate_y = tf.cast(op.outputs[0] > 0, "float32")
#    return grad * gate_g * gate_y

def transform_global_prior(label_mean, angle_label, label_exist, z_label):
    # select the z-level from z label
    label_transform = [tf.expand_dims(label_mean[z_label[i,0]],0) for i in range(1)]
    label_transform = tf.concat(label_transform, 0)
    
    # select class from class label
    label_transform = tf.multiply(label_transform, label_exist)
        
    rad = tf.expand_dims(angle_label * PI_ON_180, 2)
    cos_rad = tf.cos(rad)
    sin_rad = tf.sin(rad)
    t1 = tf.concat([cos_rad, -sin_rad, tf.zeros_like(cos_rad)], 2)
    t2 = tf.concat([sin_rad, cos_rad, tf.zeros_like(cos_rad)], 2)
    theta = tf.concat([t1, t2], 1)

    batch_grids = stn.affine_grid_generator(256, 256, theta)
    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    background = stn.bilinear_sampler(1-label_transform[...,0:1], x_s, y_s)
    background = 1 - background
    foreground = stn.bilinear_sampler(label_transform[...,1:], x_s, y_s)
    label_transform = tf.concat([foreground, background], -1)
#    label_transform = tf.concat([background, foreground], -1)
#    label_transform=    stn.bilinear_sampler(label_transform, x_s, y_s)
    return label_transform


def build_crf(origin_image, featemap, n_class, crf_config):
    def crf(featemap,image):
#        crf_config = {"g_sxy":3/12,"g_compat":3,"bi_sxy":80/12,"bi_srgb":13,"bi_compat":10,"iterations":5}
        batch_size = featemap.shape[0]
        image = image.astype(np.uint8)
        ret = np.zeros(featemap.shape,dtype=np.float32)
        for i in range(batch_size):
            ret[i,:,:,:] = crf_inference(featemap[i],image[i],crf_config,n_class)

        ret[ret < 0.0001] = 0.0001
        ret /= np.sum(ret,axis=3,keepdims=True)
        ret = np.log(ret)
        return ret.astype(np.float32)

    crf_result = tf.py_func(crf,[featemap,origin_image],tf.float32) # shape [N, h, w, C], RGB or BGR doesn't matter for crf
    return crf_result
    



def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="U-net multi-task")
    parser.add_argument("--data-dir", type=str, default=RAW_PATH,
                        help="...")
    parser.add_argument("--label-dir", type=str, default=MASK_PATH,
                        help="...")
    parser.add_argument("--model-dir", type=str, default=MODEL_PATH,
                        help="...")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--n-class", type=int, default=N_CLASS,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--z-class", type=int, default=Z_CLASS,
                        help="...")
    parser.add_argument("--shuffle", type=bool, default=SHUFFLE,
                        help="...")

    return parser.parse_args()
    

def model_prepare(x, z, args):
    z_flag = True
    angle_flag = True
    class_flag = True
    output, _, convs = unet_multi_task3.create_conv_net_upsample_multi_task_angle(x, 
                                                                               keep_prob=1,  
                                                                               is_training=False,
                                                                               channels=1, 
                                                                               n_class=args.n_class, 
                                                                               z_flag=z_flag, 
                                                                               angle_flag=angle_flag, 
                                                                               class_flag=class_flag, 
                                                                               layers=5, 
                                                                              features_root=32,  
                                                                              summaries=False, 
                                                                              z_class=args.z_class,
                                                                               )


    logits = output['output_map']
#    z_logits = output['z_output']
#    angle_output = output['angle_output']
#    return logits, z_logits, angle_output, convs
    return logits, convs
    
def main(crf_config):
    """Create the model and start the evaluation process."""
    args = get_arguments()
    tf.reset_default_graph()
    
    
    # Load reader.
    data_provider = CT_scan_util_multi_task.MedicalDataProvider(
                                      raw_path=args.data_dir,
                                      mask_path=args.label_dir,
                                      shuffle_data=args.shuffle,
                                      subject_list=SUBJECT_LIST,
                                      resize_ratio=0.5,
                                      data_aug=False,
                                      cubic=False,
                                      z_class=args.z_class,
                                      )


    # Create network.
    x = tf.placeholder("float", shape=[None, 256, 256, 1])
    y = tf.placeholder("float", shape=[None, 256, 256, args.n_class])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    z = tf.placeholder("int32", shape=[None, 1], name='z_label')
    
    #
#    angle = tf.placeholder("float", shape=[None, 1], name='angle_label')
#    class_label = tf.placeholder("float", shape=[None, args.n_class], name='class_label')
#
#    label_exist = tf.reshape(class_label, [1, 1, 1, args.n_class])
#
#    label_top = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_top_new.npy')
#    label_mid = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_middle_new.npy')
#    label_bot = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_bottom_new.npy')
#    ref_model = np.concatenate([label_bot, label_mid, label_top], 0)
#    label_mean = tf.convert_to_tensor(ref_model)
#    
#    label_transform = transform_global_prior(label_mean, angle, label_exist, z)
#    foreground = label_transform[...,1:]
#
#    x_new = tf.concat([x, foreground], axis=-1)
    x_new = x
    #
    
    
    
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        try:
            logits, convs = model_prepare(x_new, z, args)
        except ValueError:
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                scope.reuse_variables()
                logits, convs = model_prepare(x_new, z, args)
    
        # get intermedia layer
#        act_list = []
        c = convs[-1][0]
        act_list = [c[...,i][...,np.newaxis] for i in range(args.n_class)]
#        for c in convs:
#            for layer in c:
#                act = tf.reduce_max(layer, [0,1,2])
#                print(act)
#                max_act = tf.argmax(act, -1, output_type=tf.int32)
#                print(max_act)
#                max_f = layer[...,max_act]
#                max_f = tf.expand_dims(max_f, 3)
#                act_list.append(max_f)
#                print(max_f)

        with tf.name_scope('guided_back_pro_map'):
            guided_back_pro_list = []
            for class_act in act_list:
                guided_back_pro = tf.gradients(
                    class_act, x)
                guided_back_pro_list.append(guided_back_pro)


    
    
#    # Prediction       
#    raw_output = tf.nn.softmax(logits)
#    
#    # erosion
##    raw_output = tf.argmax(raw_output, dimension=3)
##    e = tf.one_hot(indices=raw_output,
##                                depth=int(args.n_class),
##                                on_value=1,
##                                off_value=0,
##                                axis=-1,
##                                )
##    e = tf.reshape(e, [1, 256, 256, args.n_class])
##    print(e)
##    e = tf.nn.erosion2d(e, [3,3,14], [1,1,1,1], [1,1,1,1], "SAME")
##    raw_output = e
#    
#    prediction = tf.argmax(raw_output, dimension=3)
#    prediction = tf.expand_dims(prediction, dim=3) # Create 4-d tensor.
#    
#    z_output = tf.nn.softmax(z_logits)
#    z_output = tf.argmax(z_output, dimension=1)
#    z_prediction = tf.expand_dims(z_output, dim=1) # Create 4-d tensor.
#
#    # CRF
##    crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":50,"bi_srgb":11,"bi_compat":3,"iterations":10}
##    crf_config = {"g_sxy":1,"g_compat":4,"bi_sxy":49,"bi_srgb":13,"bi_compat":5,"iterations":10}
##    birgb = np.arange(3, 11, 1)
##    bisxy = np.arange(50, 110, 10)
#    x_r = 255.0*tf.tile(x, [1,1,1,3])
##    for i in birgb:
##        for j in bisxy:
##            crf_config['bi_rgb'] = i
##            crf_config['bi_sxy'] = j
#    crf_output = build_crf(x_r, raw_output, args.n_class, crf_config)
#    crf_output_b = tf.exp(crf_output)
#    crf_output = tf.argmax(crf_output_b, dimension=3)
#    crf_output = tf.expand_dims(crf_output, dim=3)
#    
#    # mIoU
#    gt = tf.argmax(y, -1)
#    gt = tf.reshape(gt, [-1,])
##    z_gt = tf.argmax(z, -1)
#    z_gt = tf.reshape(z, [-1,])
#    
#    cm = tf.confusion_matrix(gt, tf.reshape(prediction, [-1,]), num_classes=args.n_class)
#    cm_z = tf.confusion_matrix(z_gt, tf.reshape(z_prediction, [-1,]), num_classes=args.z_class)
#    cm_crf = tf.confusion_matrix(gt, tf.reshape(crf_output, [-1,]), num_classes=args.n_class)


    # Set up tf session and initialize variables.  
    sess = tf.Session()
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    
    # Load weights.
    loader = tf.train.Saver()
    if args.model_dir is not None:
        load_model(loader, sess, args.model_dir)


    # Iterate over training steps.
    total_pred, total_pred_z, total_logits, x_test, y_test, z_test = [], [], [], [], [], []
    total_co = []
    slice_DSC = []
    slice_DSC_crf = []
    
    total_crf_out = []
    sum_cm = 0
    sum_cm_z = 0
    sum_cm_crf = 0
    num_steps = np.sum(data_provider.n_frames)
    print('num-steps: {}'.format(num_steps))
    
    label_mean, label_top, label_mid, label_bot = 0, 0, 0, 0
    
    gg = []
    
    for step in range(num_steps):
        print('step {:d}'.format(step))
        image, label, z_label, angle_label, _class_label= data_provider(1)
        feed_dict = {x: image, y: label, keep_prob: 1., is_training: False, 
#                     z: z_label,  angle: angle_label,  class_label: _class_label
                     }
        gradients = sess.run(guided_back_pro_list, feed_dict)
#        _logits, _pred, _z_pred, np_cm, np_cm_z, np_cm_crf, crf_out, class_out, gradients =  \
#        sess.run([logits, prediction, z_prediction, cm, cm_z, cm_crf, crf_output_b, co, guided_back_pro], feed_dict) 
        gg.append(gradients)
    cc=sess.run([convs], feed_dict)
    return gg, cc
#        total_co.append(class_out)
#        sum_cm += np_cm
#        sum_cm_z += np_cm_z
#        sum_cm_crf += np_cm_crf
#        total_pred.append(_pred)
#        total_logits.append(_logits)
#        total_pred_z.append(_z_pred)
#        x_test.append(image)
#        y_test.append(label)
#        z_test.append(z_label)
#        total_crf_out.append(crf_out)
#        if z_label[0,0] == 0:
#            label_bot += label
#        elif z_label[0,0] == 1:
#            label_mid += label
#        elif z_label[0,0] == 2:
#            label_top += label
#        label_mean += label              
#        
#        print('step {:d}'.format(step))
##        slice_DSC.append(compute_mean_dsc(np_cm))
##        slice_DSC_crf.append(compute_mean_dsc(np_cm_crf))
#    plot_confusion_matrix(sum_cm, classes=np.arange(args.n_class), normalize=True,
#                          title='Confusion matrix, without normalization')
#    plt.show()
#    plot_confusion_matrix(sum_cm_crf, classes=np.arange(args.n_class), normalize=True,
#                          title='Confusion matrix, without normalization')
#    plt.show()
#    total_mIoU = compute_mean_iou(sum_cm)
#    total_DSC = compute_mean_dsc(sum_cm)          
#    total_acc = compute_accuracy(sum_cm)
#    
#    total_DSC_crf = compute_mean_dsc(sum_cm_crf)  
#    
#    plot_confusion_matrix(sum_cm_z, classes=np.arange(args.z_class), normalize=True,
#                          title='Confusion matrix, without normalization')
#    plt.show()
#    z_acc = compute_accuracy(sum_cm_z)
#
#    return x_test, y_test, z_test, total_pred, total_pred_z, total_logits, total_mIoU, total_acc, \
#    total_DSC, sum_cm, sum_cm_z, z_acc, label_mean, total_co, total_DSC_crf, total_crf_out


if __name__ == '__main__':
    crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":1,"bi_srgb":110,"bi_compat":3,"iterations":10}
    g, c = main(crf_config)
#    crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":1,"bi_srgb":110,"bi_compat":3,"iterations":10}
#    bisrgb = np.arange(80, 110, 10)
#    bisxy = np.arange(0.1, 1, 0.2)
#
#    idx=-1
#    
#    crf_table = np.zeros(bisrgb.shape + bisxy.shape)
#    for i in bisrgb:
#        idx += 1
#        jdx = -1
#        for j in bisxy:
#            jdx += 1
#            print(i, j)
#            crf_config['bi_srgb'] = i
#            crf_config['bi_sxy'] = j
#            x_test, y_test, z_test, prediction, z_pred, logits, total_mToU, total_acc, total_DSC, sum_cm, \
#            sum_cm_z, z_acc, label_mean, total_co, total_DSC_crf, total_crf_out = main(crf_config)
#            crf_table[idx,jdx] = total_DSC_crf
            
#    cmap = plt.cm.jet    
#    for i in range(len(x_test)):
##        i=i+100
#        print('sample: {}, zlabel: {}, z_prediction: {}'.format(i, int(z_test[i][0,0]), z_pred[i][0]))
#        fig, (ax11, ax12, ax13) = plt.subplots(1,3)
#
##        ax11.imshow(x_test[i][0,...,0], 'gray')
#        ax11.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
#        ax11.set_axis_off()     
#
##        ax12.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
#        ax12.imshow(prediction[i][0,...,0], vmin=0, vmax=13)
#        ax12.set_axis_off()
#
#        ax13.imshow(np.argmax(total_crf_out[i][0], -1), vmin=0, vmax=13)
##        ax13.imshow(np.argmax(prediction[i][0], -1), vmin=0, vmax=13)
#        ax13.set_axis_off()
#        plt.show()
#        plt.close(fig)

    