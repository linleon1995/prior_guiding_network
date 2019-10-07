#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:19:36 2019

@author: acm528_02
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
import matplotlib.pyplot as plt
#from tf_unet_multi_task import util
import inspect
import time
VGG_MEAN = [103.939, 116.779, 123.68]
#from tf_tesis2.layer_multi_task import (weight_variable, weight_variable_deconv, bias_variable, 
#                            conv2d, deconv2d, upsampling2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
#                            cross_entropy, batch_norm, softmax, fc_layer, new_conv_layer_bn, new_conv_layer, upsampling_layer)
from tf_tesis2.network import (crn_encoder_sep, crn_decoder_sep, crn_atrous_encoder_sep, crn_atrous_decoder_sep, crn_atrous_decoder_sep2, 
                               crn_encoder_sep_com, crn_decoder_sep_com, crn_encoder_sep_resnet50, crn_decoder_sep_resnet50,
                               crn_encoder_sep_new_aggregation, crn_decoder_sep_new_aggregation, crn_encoder, crn_decoder)
from tf_tesis2 import stn
from tf_tesis2.dense_crf import crf_inference
from scipy.misc import imresize, imrotate
from tf_tesis2.utils import (conv2d)
from tf_tesis2 import train_utils, module, input_preprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
PI_ON_180 = 0.017453292519943295
# TODO: get_optimizer and loss
# TODO: tensorboard
# TODO: build folder??


def _average_gradients(tower_grads):
  """Calculates average of gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list is
      over individual gradients. The inner list is over the gradient calculation
      for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
       across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads, variables = zip(*grad_and_vars)
    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)

    # All vars are of the same value, using the first tower here.
    average_grads.append((grad, variables[0]))

  return average_grads


class Unet(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._loss_utils for more options
    """
    
    def __init__(self, model_flag, nx, ny, channels=1, n_class=2, cost="cross_entropy", prior=None, 
                 batch_size=None, seq_length=None, cost_kwargs={}, **kwargs):
        tf.reset_default_graph()

        self.seq_length = seq_length
        self.n_class = n_class
        if self.seq_length is not None:
            self.batch_size = batch_size*self.seq_length
        else:
            self.batch_size = batch_size
        self.prior = tf.convert_to_tensor(prior)
        self.z_class = kwargs.get("z_class", 5)
        self.data_aug = kwargs.get("data_aug", None)       
        self.summaries = kwargs.get("summaries", True)
        self.z_flag = model_flag['zlevel_classify']
        self.angle_flag = model_flag['rotate_module']
        self.nx = nx
        self.ny = ny
        
        self.x = tf.placeholder("float", shape=[None, self.nx, self.ny, channels], name='x')
        self.y = tf.placeholder("float", shape=[None, self.nx, self.ny, n_class], name='y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout (keep probability)
        self.is_training = tf.placeholder(tf.bool)

        self.seq_length = seq_length
#        if self.z_flag: self.z_label = tf.placeholder("int32", shape=[None, self.n_class], name='z_label')
        if self.z_flag: self.z_label = tf.placeholder("int32", shape=[None, 1], name='z_label')
        if self.angle_flag: self.angle_label = tf.placeholder("float", shape=[None, 2, 3], name='angle_label')

        self.output, self.layer_dict = self._model(cost_kwargs, kwargs)
        self.logits = self.output['output_map']
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        # TODO: we don't need loss when doing evaluate
#        if self.is_training:
        self.total_loss = self._get_loss(cost, cost_kwargs, kwargs)
        self.gradients_node = tf.gradients(self.total_loss, self.variables)
        
        if self.summaries:
#            self.y_for_show = colorize(tf.expand_dims(tf.argmax(self.y[0:1], 3), 3), cmap='viridis')
#            self.pred_for_show = colorize(tf.expand_dims(tf.argmax(self.predicter[0:1], 3), 3), cmap='viridis')
#            print(y_for_show, pred_for_show, 30*'w')
#            y_for_show = tf.multiply(tf.divide(tf.expand_dims(tf.argmax(self.y[0:1], 3), 3), n_class), 255)
#            pred_for_show = tf.multiply(tf.divide(tf.expand_dims(tf.argmax(self.predicter[0:1], 3), 3), n_class), 255)

#            tf.summary.image('summary_raw', get_image_summary(self.x[0:1,...,0:1]))
#            tf.summary.image('summary_label', get_image_summary(self.y_for_show))
#            tf.summary.image('summary_predict', get_image_summary(self.pred_for_show))
            
#            for k in range(self.n_class):
#                tf.summary.image('prior_transform_{}'.format(k), get_image_summary(colorize(self.prior_transform[0:1,...,k:k+1], cmap='viridis')))
            
#            for k in range(self.n_class):
#                tf.summary.image('predict_class_{}'.format(k), get_image_summary(colorize(self.predicter[0:1,...,k:k+1], cmap='viridis')))
                
            for k in self.layer_dict.keys():
                tf.summary.image('intermedia_{}'.format(k), get_image_summary(self.layer_dict[k][...,0:1]))
            
            for k in self.layer_dict.keys():
                tf.summary.histogram("histogram_{}".format(k), self.layer_dict[k])
                
 
        
        
        
    def _get_loss(self, cost, cost_kwargs, kwargs):
        if self.seq_length is not None:
            labels = self.y
            labels = [self.y[i:i+1] for i in range(self.seq_length//2,self.batch_size,self.seq_length)]
            labels = tf.concat(labels, axis=0)
        else:
            labels = self.y
        
        self.seg_loss = self._loss_utils(self.logits, labels, cost, cost_kwargs)
        total_loss = self.seg_loss

        if self.z_flag: 
            self.z_cost = self._loss_utils(self.z_pred, self.z_label, 'MSE', cost_kwargs)
            lambda_z = kwargs.get("lambda_z", 1e-2)
            total_loss += lambda_z * self.z_cost

        if kwargs.get("lambda_guidance", None) is not None:
            # get all guidance
            guidance = {}
            for layer_name in self.output:
                if "guidance" in layer_name:
                    guidance[layer_name] = self.output[layer_name]

            # calcuate guidance loss
            self.guidance_loss = 0
            for g in guidance.values():
                ny_g = g.get_shape()[1]
                nx_g = g.get_shape()[2]
                ys = tf.image.resize_bilinear(self.y, [ny_g, nx_g])
                self.guidance_loss += self._loss_utils(g, ys, cost, cost_kwargs)
            lambda_guidance = kwargs.get("lambda_guidance", None)
            total_loss += lambda_guidance * self.guidance_loss

        self.accuracy = 1-self.seg_loss
        return total_loss

    def _model(self, cost_kwargs, kwargs):
        """
        """
        output, layer_dict = crn_atrous_encoder_sep(self.x, self.n_class, self.z_class, self.batch_size, self.seq_length, is_training=True )

        self.z_output = output['z_output']
#        self.z_pred = self.z_output
#        self.z_pred = tf.nn.softmax(self.z_output, dim=1)
        self.z_pred = tf.nn.sigmoid(self.z_output)
        
        
            
        if self.prior is not None:
#            self.prior_transform = self.y
            indices = tf.cast(tf.floor(self.z_pred), tf.int32)
            prior_transform = self.indexing(self.prior, indices)
            self.prior_transform  = tf.one_hot(indices=prior_transform,
                                depth=int(self.n_class),
                                on_value=1,
                                off_value=0,
                                axis=-1,
                                )

            # TODO: modify transform_global_prior function
#            self.prior_transform = self.transform_global_prior(self.prior)
#            self.foreground = self.prior_transform[...,1:]
        
        # temporal information (bi GRU)
        if self.seq_length is not None:
            conv_output = tf.split(layer_dict['pool4'], self.seq_length, axis=0)
            with tf.variable_scope("RNN"):
                rnn_output = module.bidirectional_GRU(features=conv_output,
                                                  batch_size=self.batch_size,
                                                  nx=self.nx//8,
                                                  ny=self.ny//8,
                                                  n_class=self.n_class,
                                                  seq_length=self.seq_length,
                                                  is_training=True)
    
#                self.logits = conv2d(rnn_output, [1,1,32,self.n_class], activate=None, scope="logits", bn_flag=False)
#                self.logits = rnn_output
            layer_dict['pool4'] = rnn_output
            
#        output, layer_dict, _ = unet_prior_guide_prof_decoder( output, self.prior_transform, self.batch_size, layer_dict, is_training=True )
#        output, layer_dict, _ = crn_decoder( output, self.prior_transform, self.batch_size, layer_dict, is_training=True )
#        output, layer_dict, _ = crn_atrous_decoder_sep( output, self.prior_transform, self.batch_size, layer_dict, is_training=True )
        output, layer_dict, _ = crn_atrous_decoder_sep2( output, self.prior_transform, self.batch_size, layer_dict, is_training=True )
#        output, layer_dict, _ = crn_decoder_sep_new_aggregation( output, self.prior_transform, self.n_class, self.batch_size, layer_dict, is_training=True )
#        output, layer_dict, _ = crn_decoder_sep( output, self.prior_transform, self.batch_size, layer_dict, is_training=True )
#        output, layer_dict, _ = crn_decoder_sep_com( output, self.prior_transform, self.n_class, self.batch_size, layer_dict, is_training=True )
#        output, layer_dict, _ = crn_decoder_sep_resnet50( output, self.prior_transform, self.batch_size, layer_dict, is_training=True )
        return output, layer_dict

    def _loss_utils(self, logits, labels, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """
        if cost_name == "cross_entropy":
            flat_logits = tf.reshape(logits, [-1, logits.get_shape()[-1]])
            flat_labels = tf.reshape(labels, [-1, labels.get_shape()[-1]])
            
            class_weights = cost_kwargs.pop("class_weights", None)
            
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
        
                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)
        
                loss = tf.reduce_mean(weighted_loss)
                
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
                                                                              labels=flat_labels))

                
        elif cost_name == "KL_divergence":
            eps = 1e-5
            labels = tf.exp(labels)
            loss = tf.reduce_mean(tf.reduce_sum(labels * tf.log((eps+labels)/(eps+logits)), axis=3))
            
        elif cost_name == "cross_entropy_sigmoid":
            flat_logits = tf.reshape(logits, [-1, logits.get_shape()[-1]])
            flat_labels = tf.reshape(labels, [-1, labels.get_shape()[-1]])
            
            class_weights = cost_kwargs.pop("class_weights", None)
            
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
        
                loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits,
                                                                   labels=flat_labels)
                weighted_loss = tf.multiply(loss_map, weight_map)
        
                loss = tf.reduce_mean(weighted_loss)
                
            else:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, 
                                                                              labels=flat_labels))
                
        elif cost_name == "cross_entropy_zlabel":
            class_weights = cost_kwargs.pop("class_weights", None)
            
            labels = tf.one_hot(indices=labels,
                                depth=int(self.z_class),
                                on_value=1,
                                off_value=0,
                                axis=-1,
                                )
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                weight_map = tf.multiply(labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
        
                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels)
                weighted_loss = tf.multiply(loss_map, weight_map)
        
                loss = tf.reduce_mean(weighted_loss)
                
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                                              labels=labels))
        
        elif cost_name == "cross_entropy_new_zlabel":
            class_weights = cost_kwargs.pop("class_weights", None)
            
#            labels = tf.expand_dims(labels, -1)
            labels = tf.one_hot(indices=labels,
                                depth=int(self.z_class+1),
                                on_value=1,
                                off_value=0,
                                axis=-1,
                                )
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                weight_map = tf.multiply(labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)
        
                loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels)
                weighted_loss = tf.multiply(loss_map, weight_map)
        
                loss = tf.reduce_mean(weighted_loss)
                
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                                              labels=labels))
                
        elif cost_name == "mean_dice_coefficient":       
            eps = 1e-5
            gt = tf.reshape(labels, [-1, labels.get_shape()[-1]])
            prediction = tf.nn.softmax(logits)
            prediction = tf.reshape(prediction, [-1, logits.get_shape()[-1]])
            
            intersection = tf.reduce_sum(gt*prediction, 0)
            union = tf.reduce_sum(gt, 0) + tf.reduce_sum(prediction, 0)

            loss = (2*intersection+eps) / (union+eps)
            loss = 1 - tf.reduce_mean(loss)

            
        elif cost_name == "seed_loss":
            eps = 1e-5
            
            count = tf.reduce_sum(labels,axis=(1,2,3),keep_dims=True)
            count = tf.cast(count, dtype=tf.float32)
            labels = tf.cast(labels, dtype=tf.float32)
            loss = -tf.reduce_mean(tf.reduce_sum( labels*tf.log(self.predicter+eps), axis=(1,2,3), keep_dims=True)/count)
         
            
        elif cost_name == "MSE":
            loss = tf.losses.mean_squared_error(
                                            labels,
                                            logits,
                                            )
            
        else:
            raise ValueError("Unknown cost function: "%cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)

            
        return loss
    
    
    def transform_global_prior(self, prior):
#        if self.z_flag:
        prior = tf.reshape(prior, [self.z_class, -1])
        prior_transform = tf.matmul(self.z_pred, prior)
        prior_transform = tf.reshape(prior_transform, [self.batch_size, 512,512,self.n_class])
        if self.angle_flag:
            self.theta = self.angle_label
            batch_grids = stn.affine_grid_generator(self.ny, self.nx, self.theta)
            x_s = batch_grids[:, 0, :, :]
            y_s = batch_grids[:, 1, :, :]
        
            background = stn.bilinear_sampler(1-prior_transform[...,0:1], x_s, y_s)
            background = -(background-1)
            foreground = stn.bilinear_sampler(prior_transform[...,1:], x_s, y_s)
            prior_transform = tf.concat([background, foreground], -1)
            
        return prior_transform
    
    def indexing(self, prior, z):
        return tf.gather(params=prior, indices=z[:,0])

    
    def predict(self, model_path, x_test, is_training=False):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
#            self.restore(sess, model_path)
            
            saver = tf.train.import_meta_graph(model_path+'/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction,logits = sess.run([self.predicter, self.logits], feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1., self.net.is_training: is_training})
            
        return prediction, logits
    
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path, var_list=None):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        if var_list is not None:
            saver = tf.train.Saver(var_list=var_list)
        else:
            saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)



class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    """
    
    verification_batch_size = 4
    
    def __init__(self, net, batch_size=1, norm_grads=False, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.power = opt_kwargs.pop("power", 0.7)
        self.epochs = opt_kwargs.pop("epochs", 100)
        
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                       **self.opt_kwargs).minimize(self.net.seg_loss, 
                                                                                    global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
#            decay_rate = self.opt_kwargs.pop("decay_rate", 0.96)
#            self.learning_rate_node = tf.Variable(learning_rate)
            self.training_number_of_steps = self.epochs*training_iters
            self.learning_rate_node = tf.train.polynomial_decay(learning_rate=learning_rate, global_step=global_step, 
                                                                decay_steps=self.training_number_of_steps, end_learning_rate=1e-2*learning_rate, 
                                                                power=self.power, cycle=False)

            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                                   **self.opt_kwargs).minimize(self.net.total_loss,
                                                                         global_step=global_step)

#            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
#                                               **self.opt_kwargs)
    
        return optimizer
    
    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)
#        global_step = tf.train.get_or_create_global_step()
        
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]))

        if self.net.summaries and self.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)
        if self.net.summaries:
            tf.summary.scalar('loss', self.net.total_loss)
            tf.summary.scalar('dsc_loss', self.net.seg_loss)
            if self.net.z_flag:
                tf.summary.scalar('z_loss', self.net.z_cost)
#            if self.net.angle_flag:
#                tf.summary.scalar('angle_loss', self.net.angle_loss)
    #        tf.summary.scalar('cross_entropy', self.net.cross_entropy)
#            tf.summary.scalar('guidance_loss', self.net.guidance_loss)
            tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        
        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
        
        return init

    

    def train(self, data_provider, valid_provider, output_path, training_iters=10, dropout=0.75, display_step=1, 
              restore=False, write_graph=False, prediction_path = 'prediction', is_training=True):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        save_path = os.path.join(output_path, 'model.ckpt')
#        save_path = output_path
        if self.epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, output_path, restore, prediction_path)
        
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    if self.net.seq_length is not None:
                        restore_var = [v for v in tf.trainable_variables() if 'RNN' not in v.name] 
                        self.net.restore(sess, ckpt.model_checkpoint_path, restore_var)
                    else:
                        self.net.restore(sess, ckpt.model_checkpoint_path)
            
#            pred_shape = self.store_prediction(sess, test_x, test_y, "_init")
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)

            logging.info("Start optimization")
            
            avg_gradients = None
            total_loss_list = []
            valid_loss_list = []
            mb_total_loss_list = []
            dsc_loss_list = []
            
            seed_loss_list = []
            z_loss_list = []
            angle_loss_list = []
            class_loss_list = []
            
            n_sample = len(valid_provider._find_data_files())
            valid_iters = n_sample // self.batch_size
#            min_loss = 1e10
            
            import time

            for epoch in range(self.epochs):
                total_loss = 0
                valid_loss = 0
                dsc_total_loss = 0
                
                seed_total_loss = 0
                z_total_loss = 0
                angle_total_loss = 0
                class_total_loss = 0
                crf_total_loss = 0
                
                
                
                for v_step in range(valid_iters):
                    test_x, test_y, test_z, test_angle, test_class_gt = valid_provider(self.batch_size)
                    _feed_dict = {self.net.x: test_x, 
                                  self.net.y: test_y, 
                                  self.net.keep_prob: dropout, 
                                  self.net.is_training: is_training,
                                  }
                    if self.net.z_flag: _feed_dict[self.net.z_label] = test_z 
                    if self.net.angle_flag: _feed_dict[self.net.angle_label] = test_angle
                    valid_loss += sess.run(self.net.total_loss, feed_dict=_feed_dict)
 
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y, batch_z, batch_angle, batch_class_gt = data_provider(self.batch_size)
                    _feed_dict = {self.net.x: batch_x, 
                                  self.net.y: batch_y, 
                                  self.net.keep_prob: dropout, 
                                  self.net.is_training: is_training,
                                  }
                    if self.net.z_flag: _feed_dict[self.net.z_label] = batch_z
                    if self.net.angle_flag: _feed_dict[self.net.angle_label] = batch_angle
#                    if self.net.class_flag: _feed_dict[self.net.class_label] = batch_class_gt


                        
                    pred, _, dsc_loss, lr, loss, \
                    prior = sess.run((
                                                                            self.net.logits, 
                                                                            self.optimizer, 
                                                                             self.net.seg_loss, 
                                                                             self.learning_rate_node, 
#                                                                             self.net.gradients_node,
#                                                                             self.net.z_cost,
#                                                                             self.net.angle_loss,
                                                                             self.net.total_loss,
#                                                                             self.net.z_output,
#                                                                             self.net.angle_output,
                                                                             self.net.prior_transform,
#                                                                             self.net.label_exist,
#                                                                             self.net.pred_for_show,
#                                                                             self.net.g1,
#                                                                             self.net.g1,
#                                                                             self.net.g1,
                                                                             ), 
                                                                          feed_dict=_feed_dict,
                                                                          )
    
#                    if self.net.summaries:
#                        tf.summary.image('summary_raw_01', get_image_summary(self.net.x))
#                        tf.summary.image('summary_label_01', get_image_summary(tf.expand_dims(tf.argmax(self.net.y, 3),3)))
#                        tf.summary.image('summary_predict_01', get_image_summary(tf.expand_dims(tf.argmax(self.net.predicter, 3),3)))
                        
                    if step%40 == 0:
                        self.show_prediction(batch_x, batch_y, batch_z, pred, output_path, display=True)
#                        plt.imshow(pp[0])
                        plt.show()
                        
#                        plt.subplot(131)
#                        plt.imshow(g1[0,...,1])
#                        plt.subplot(132)
#                        plt.imshow(g2[0,...,1])
#                        plt.subplot(133)
#                        plt.imshow(g3[0,...,1])
#                        plt.show()
#                        plt.imshow(label_exist[0,0])
#                        plt.show()
                        

#                        plt.imshow(1/(1+np.exp(-class_output[0:1])))
#                        plt.show()
                        
#                        print('z_loss: {}   angle_loss: {}'.format(z_loss, angle_loss))
#                        print(classmap.shape)
#                        sc = plt.imshow(classmap[0,...,0], 'jet')
#                        plt.colorbar(sc)
#                        plt.show()

#                    if self.net.summaries and self.norm_grads:
#                        avg_gradients = _update_avg_gradients(avg_gradients, gradients, step)
#                        norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
#                        self.norm_gradients_node.assign(norm_gradients).eval()
                    
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, batch_y, batch_z, batch_angle, batch_class_gt)
                        
                    total_loss += loss
                    dsc_total_loss += dsc_loss
                    
#                    seed_total_loss += seed_loss
#                    z_total_loss += z_loss
#                    angle_total_loss += angle_loss
#                    class_total_loss += class_loss
                    
                    mb_total_loss_list.append(loss)
                    
                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(test_y, pred, loss)
                  
                save_path = self.net.save(sess, save_path)


#                if min_loss > total_loss:
#                    min_loss = total_loss
#                    saver = tf.train.Saver()
#                    saver.save(sess, save_path+'.best')
                        
#                total_loss_list.append(total_loss/training_iters)
#                valid_loss_list.append(valid_loss/valid_iters)
#                dsc_loss_list.append(dsc_total_loss/training_iters)
                
#                display_loss([total_loss_list, valid_loss_list], 'loss', ['train', 'validate'], output_path=output_path)
#                display_loss([mb_total_loss_list], 'mb_loss', ['mb_loss'], output_path=output_path)
#                display_loss([dsc_loss_list], 'dsc_loss', ['dsc_loss'], output_path=output_path)
                
#                seed_loss_list.append(seed_total_loss/training_iters)
#                display_loss([dsc_loss_list, seed_loss_list], 'dsc_loss & seed_loss', ['dsc_loss', 'reference_loss'], output_path=output_path)
                
#                if self.net.z_flag:
#                    z_loss_list.append(z_total_loss/training_iters)
#                    display_loss([z_loss_list], 'z_level classification loss', ['z_level_loss'], output_path=output_path)
                    
#                if self.net.angle_flag:
#                    angle_loss_list.append(angle_total_loss/training_iters)
#                    display_loss([angle_loss_list], 'angle regression loss', ['angle_loss'], output_path=output_path)
                    
#                if self.net.class_flag:
#                    class_loss_list.append(class_total_loss/training_iters)
#                    display_loss([class_loss_list], 'organ classification loss', ['class_loss'], output_path=output_path)

                total_loss_list.append(total_loss/training_iters)
                valid_loss_list.append(valid_loss/valid_iters)
#                dsc_loss_list.append(dsc_total_loss/training_iters)
##                seed_loss_list.append(seed_total_loss/training_iters)
#                z_loss_list.append(z_total_loss/training_iters)
#                angle_loss_list.append(angle_total_loss/training_iters)
##                class_loss_list.append(class_total_loss/training_iters)
#                
                display_loss([total_loss_list, valid_loss_list], 'loss', ['train', 'validate'], output_path=output_path)
#                display_loss([mb_total_loss_list], 'mb_loss', ['mb_loss'], output_path=output_path)
#                display_loss([dsc_loss_list], 'dsc_loss', ['dsc_loss'], output_path=output_path)
##                display_loss([dsc_loss_list, seed_loss_list], 'dsc_loss & seed_loss', ['dsc_loss', 'reference_loss'], output_path=output_path)
#                display_loss([z_loss_list], 'z_level classification loss', ['z_level_loss'], output_path=output_path)
#                display_loss([angle_loss_list], 'angle regression loss', ['angle_loss'], output_path=output_path)
##                display_loss([class_loss_list], 'organ classification loss', ['class_loss'], output_path=output_path)

            logging.info("Optimization Finished!")
            
            return output_path
#            return save_path




    def show_prediction(self, batch_x, batch_y, batch_z, pred, output_path, display=True):
#        print(batch_x[0,0,0,0],30*'-')
        fig = plt.figure()
        plt.ion()
        plt.subplot(131)
        if batch_x.shape[-1]==3:
            plt.imshow(batch_x[0])
        elif batch_x.shape[-1]==1:
            plt.imshow(batch_x[0,...,0], cmap='gray')
        plt.subplot(132)
        plt.imshow(np.argmax(batch_y[0], axis=-1), vmin=0, vmax=self.net.n_class-1)
        plt.subplot(133)
        plt.imshow(np.float32(np.argmax(pred[0], axis=-1)), vmin=0, vmax=self.net.n_class-1)
        if display:
            plt.show()
            
        plt.subplot(131)
        plt.imshow(pred[0,...,0])
        plt.subplot(132)
        plt.imshow(pred[0,...,1])
        plt.subplot(133)
        plt.imshow(pred[0,...,2])
    
        plt.savefig(output_path+'_pred'+'.png')
        plt.ioff()
        if display:
            plt.show()
        
        
    def store_prediction(self, batch_y, prediction, loss):
        logging.info("Verification error= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,                                                    
                                                                        batch_y),
                                                                        loss))

    
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, batch_z, batch_angle, batch_class_gt, is_training=False):
        # Calculate batch loss and accuracy
        _feed_dict = {self.net.x: batch_x, 
                      self.net.y: batch_y, 
                      self.net.keep_prob: 1, 
                      self.net.is_training: is_training}
        if self.net.z_flag: _feed_dict[self.net.z_label] = batch_z
        if self.net.angle_flag: _feed_dict[self.net.angle_label] = batch_angle
        summary_str, loss, acc, predictions = sess.run([self.summary_op, 
                                                            self.net.seg_loss, 
                                                            self.net.accuracy, 
                                                            self.net.logits], 
                                                           feed_dict=_feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                            loss,
                                                                                                            acc,
                                                                                                            error_rate(predictions, batch_y)))
    def acc_and_loss(self, _mb_loss, _loss, _v_loss, _a_loss, output_path, _z_mb_loss, _z_loss, display=True):
        print('-'*30)
        print('Plotting...')
        print('-'*30)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        
        # summarize history for accuracy
        fig = plt.figure()
        
        plt.plot(_mb_loss)    
        plt.title('mini-batch loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(output_path+'_mb_loss'+'.png')
        if display:
            plt.show()
        plt.close(fig)
        
        # summarize history for loss
        fig = plt.figure()
        
        plt.plot(_loss) 
        plt.hold(True)
        plt.plot(_v_loss, 'g')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper right')
        plt.savefig(output_path+'_loss'+'.png')
        if display:
            plt.show()
        plt.close(fig)
        
        # summarize history for accuracy
        fig = plt.figure()
        
        plt.plot(_z_mb_loss, 'darkorange')    
        plt.title('z_axis classification mini-batch loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
#        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(output_path+'_z_mb_loss'+'.png')
        if display:
            plt.show()
        plt.close(fig)
        
        # summarize history for loss
        fig = plt.figure()
        
        plt.plot(_z_loss, 'darkorange') 

        plt.title('z_axis classification loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
#        plt.legend(['train'], loc='upper left')
        plt.savefig(output_path+'_z_loss'+'.png')
        if display:
            plt.show()
        plt.close(fig)
           
        # summarize history for loss
        fig = plt.figure()
        plt.plot(_a_loss, 'r')
        plt.title('angle regression loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
#        plt.legend(['train'], loc='upper left')
        plt.savefig(output_path+'_a_loss'+'.png')
        if display:
            plt.show()
        plt.close(fig)
        
        
def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))
        
    return avg_gradients

def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V   

def display_loss(loss, loss_title, loss_name, output_path=None):
    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # summarize history for loss
    plt.ion()
    fig = plt.figure()
    
    assert len(loss) == len(loss_name)
    plt.plot(loss[0])
    if len(loss) > 1:
        plt.hold(True)
        for idx in range(1, len(loss)):
            plt.plot(loss[idx])
    
    plt.legend(loss_name, loc='upper right')    
    plt.title(loss_title)
    plt.ylabel('loss')
    plt.xlabel('epoch')

    if output_path is not None:
        plt.savefig(output_path+loss_title+'.png')
    plt.ioff()
    plt.show()
    plt.close(fig)
    