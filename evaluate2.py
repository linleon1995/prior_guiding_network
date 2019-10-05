#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:37:31 2019

@author: acm528_02
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:27:56 2019

@author: EE_ACM528_04
"""

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tf_tesis2  import unet_multi_task2
from tf_tesis2.eval_utils import compute_mean_dsc, compute_mean_iou, compute_accuracy, load_model, plot_confusion_matrix
#from tf_unet_multi_task  import util_multi_task 
import numpy as np
import CT_scan_util_multi_task
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tf_unet_multi_task import unet_multi_task 
#from tf_unet import unet_from_lab_server
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
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
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_049/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_051/model.ckpt.best"

DATA_LIST_PATH = './dataset/val.txt'
IGNORE_LABEL = 255
#NUM_CLASSES = 14
NUM_STEPS = 1449 # Number of images in the validation set.
Z_CLASS = 3
SUBJECT_LIST = np.arange(25, 30)
SHUFFLE = False
IMG_SHAPE = [1, 256, 256, 1]
LABEL_SHAPE = [1, 256, 256, N_CLASS]





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
    logits, z_logits, angle_output, _ = unet_multi_task2.create_conv_net_upsample_multi_task_angle(x, 
                                                                                                   keep_prob=1,  
                                                                                                   is_training=False,
                                                                                                   channels=1, 
                                                                                                   n_class=args.n_class, 
                                                                                                   layers=5, 
                                                                                                  features_root=32,  
                                                                                                  summaries=False, 
                                                                                                  z_class=args.z_class,
                                                                                                   )


#    logits, z_logits, _ = unet_multi_task2.create_conv_net_upsample_multi_task(x, 
##                                                                                      z_label=z,                       
#                                                                                      keep_prob=1, 
#                                                                                      is_training=False, 
#                                                                                      channels=1, 
#                                                                                      n_class=args.n_class, 
#                                                                                      layers=5, 
#                                                                                      features_root=32,  
#                                                                                      summaries=False, 
#                                                                                      z_class=args.z_class,
#                                                                                      )
    return logits, z_logits, z_logits
  

#class Evaluate():
#    def __init__(self, ):
#        self.args = get_arguments()
#        
#    def __call__(self, ):
#        # create model and get the prediction or other model results
#        self.create_model()
#        
#        # evaluate
#        
#        # show results
#        
#    def create_model(self, ):
#        # Create network.
#        x = tf.placeholder("float", shape=[None, 256, 256, 1])
#        self.y = tf.placeholder("float", shape=[None, 256, 256, self.args.n_class])
#        self.keep_prob = tf.placeholder(tf.float32)
#        self.is_training = tf.placeholder(tf.bool)
#        z = tf.placeholder("int32", shape=[None, 1], name='z_label')
#        
#        logits, z_logits, co = model_prepare(x, z, self.args)
#        
#        # Prediction       
#        raw_output = tf.nn.softmax(logits)
#        raw_output = tf.argmax(raw_output, dimension=3)
#        self.prediction = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
#        
#        z_output = tf.nn.softmax(z_logits)
#        z_output = tf.argmax(z_output, dimension=1)
#        self.z_prediction = tf.expand_dims(z_output, dim=1) # Create 4-d tensor.
#        return 
#    
#    def get_metrices(self, ):
#        pass
#    def show_results(self, ):
#        pass


def main():
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
    
    logits, z_logits, co = model_prepare(x, z, args)
    
    
    # Prediction       
    raw_output = tf.nn.softmax(logits)
    raw_output = tf.argmax(raw_output, dimension=3)
    prediction = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    z_output = tf.nn.softmax(z_logits)
    z_output = tf.argmax(z_output, dimension=1)
    z_prediction = tf.expand_dims(z_output, dim=1) # Create 4-d tensor.

    # CRF
    
    # mIoU
    pred = tf.reshape(prediction, [-1,])
    z_pred = tf.reshape(z_prediction, [-1,])
    gt = tf.argmax(y, -1)
    gt = tf.reshape(gt, [-1,])
#    z_gt = tf.argmax(z, -1)
    z_gt = tf.reshape(z, [-1,])
    
    cm = tf.confusion_matrix(gt, pred, num_classes=args.n_class)
    cm_z = tf.confusion_matrix(z_gt, z_pred, num_classes=args.z_class)


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
    total_mIoU, total_acc, total_pred, total_pred_z, total_logits, x_test, y_test, z_test = [], [], [], [], [], [], [], []
    total_co = []
    slice_DSC = []
    sum_cm = 0
    sum_cm_z = 0
    num_steps = np.sum(data_provider.n_frames)
    print('num-steps: {}'.format(num_steps))
    
    label_mean, label_top, label_mid, label_bot = 0, 0, 0, 0
    for step in range(num_steps):
        image, label, z_label, _, _ = data_provider(1)
        feed_dict = {x: image, y: label, keep_prob: 1., is_training: False, z: z_label}
        _logits, _pred, _z_pred, np_cm, np_cm_z, class_out = sess.run([logits, prediction, z_pred, cm, cm_z, co], feed_dict) 
        
        total_co.append(class_out)
        sum_cm += np_cm
        sum_cm_z += np_cm_z
        total_pred.append(_pred)
        total_logits.append(_logits)
        total_pred_z.append(_z_pred)
        x_test.append(image)
        y_test.append(label)
        z_test.append(z_label)
        if z_label[0,0] == 0:
            label_bot += label
        elif z_label[0,0] == 1:
            label_mid += label
        elif z_label[0,0] == 2:
            label_top += label
        label_mean += label              
        
        print('step {:d}'.format(step))
        slice_DSC.append(compute_mean_dsc(np_cm))
    plot_confusion_matrix(sum_cm, classes=np.arange(args.n_class), normalize=True,
                          title='Confusion matrix, without normalization')
    plt.show()
    total_mIoU = compute_mean_iou(sum_cm)
    total_DSC = compute_mean_dsc(sum_cm)          
    total_acc = compute_accuracy(sum_cm)
    
    plot_confusion_matrix(sum_cm_z, classes=np.arange(args.z_class), normalize=True,
                          title='Confusion matrix, without normalization')
    plt.show()
    z_acc = compute_accuracy(sum_cm_z)

    return x_test, y_test, z_test, total_pred, total_pred_z, total_logits, total_mIoU, total_acc, 
    total_DSC, sum_cm, slice_DSC, sum_cm_z, z_acc, label_mean, total_co


if __name__ == '__main__':
    x_test, y_test, z_test, prediction, z_pred, logits, total_mToU, total_acc, total_DSC, sum_cm, slice_DSC, sum_cm_z, z_acc, label_mean, total_co = main()

#    cmap = plt.cm.jet    
#    for i in range(len(x_test)):
#        print('sample: {}, zlabel: {}, z_prediction: {}'.format(i, int(z_test[i][0,0]), z_pred[i][0]))
#        fig, (ax11, ax12, ax13) = plt.subplots(1,3)
#
#        ax11.imshow(x_test[i][0,...,0], 'gray')
#        ax11.set_axis_off()     
#
#        ax12.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
#        ax12.set_axis_off()
#
#        ax13.imshow(prediction[i][0,...,0], vmin=0, vmax=13)
#        ax13.set_axis_off()
#        plt.show()
#        plt.close(fig)

    