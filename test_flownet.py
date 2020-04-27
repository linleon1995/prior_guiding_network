
# from ..net import Net, Mode
# from ..utils import LeakyReLU, average_endpoint_error, pad, antipad
# from ..downsample import downsample
import abc
from enum import Enum
import numpy as np
import os
import math
import tensorflow as tf
slim = tf.contrib.slim



class Mode(Enum):
    TRAIN = 1
    TEST = 2
        
        
class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.global_step = slim.get_or_create_global_step()
        self.mode = mode
        self.debug = debug

    @abc.abstractmethod
    def model(self, inputs, training_schedule, trainable=True):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return

    # def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=False):
    #     input_a = imread(input_a_path)
    #     input_b = imread(input_b_path)

    #     # Convert from RGB -> BGR
    #     input_a = input_a[..., [2, 1, 0]]
    #     input_b = input_b[..., [2, 1, 0]]

    #     # Scale from [0, 255] -> [0.0, 1.0] if needed
    #     if input_a.max() > 1.0:
    #         input_a = input_a / 255.0
    #     if input_b.max() > 1.0:
    #         input_b = input_b / 255.0

    #     # TODO: This is a hack, we should get rid of this
    #     training_schedule = LONG_SCHEDULE

    #     inputs = {
    #         'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
    #         'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
    #     }
    #     predictions = self.model(inputs, training_schedule)
    #     pred_flow = predictions['flow']

    #     saver = tf.train.Saver()

    #     with tf.Session() as sess:
    #         saver.restore(sess, checkpoint)
    #         pred_flow = sess.run(pred_flow)[0, :, :, :]

    #         unique_name = 'flow-' + str(uuid.uuid4())
    #         if save_image:
    #             flow_img = flow_to_image(pred_flow)
    #             full_out_path = os.path.join(out_path, unique_name + '.png')
    #             imsave(full_out_path, flow_img)

    #         if save_flo:
    #             full_out_path = os.path.join(out_path, unique_name + '.flo')
    #             write_flow(pred_flow, full_out_path)

    def train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        predictions = self.model(inputs, training_schedule)
        total_loss = self.loss(flow, predictions)
        tf.summary.scalar('loss', total_loss)

        if checkpoints:
            for (checkpoint_path, (scope, new_scope)) in checkpoints.iteritems():
                variables_to_restore = slim.get_variables(scope=scope)
                renamed_variables = {
                    var.op.name.split(new_scope + '/')[1]: var
                    for var in variables_to_restore
                }
                restorer = tf.train.Saver(renamed_variables)
                with tf.Session() as sess:
                    restorer.restore(sess, checkpoint_path)

        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            slim.learning.train(
                train_op,
                log_dir,
                # session_config=tf.ConfigProto(allow_soft_placement=True),
                global_step=self.global_step,
                save_summaries_secs=60,
                number_of_steps=training_schedule['max_iter']
            )
            
            
class FlowNetS(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        super(FlowNetS, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, training_schedule, trainable=True):
        _, height, width, _ = inputs['input_a'].shape.as_list()
        stacked = False
        with tf.variable_scope('FlowNetS'):
            if 'warped' in inputs and 'flow' in inputs and 'brightness_error' in inputs:
                stacked = True
                concat_inputs = tf.concat([inputs['input_a'],
                                           inputs['input_b'],
                                           inputs['warped'],
                                           inputs['flow'],
                                           inputs['brightness_error']], axis=3)
            else:
                concat_inputs = tf.concat([inputs['input_a'], inputs['input_b']], axis=3)
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=slim.variance_scaling_initializer(),
                                activation_fn=LeakyReLU,
                                # We will do our own padding to match the original Caffe code
                                # padding='VALID'
                                ):

                weights_regularizer = slim.l2_regularizer(training_schedule['weight_decay'])
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    with slim.arg_scope([slim.conv2d], stride=2):
                        conv_1 = slim.conv2d(pad(concat_inputs, 3), 64, 7, scope='conv1')
                        conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope='conv2')
                        conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope='conv3')

                    conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope='conv3_1')
                    with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                        conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                        conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                        conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                        conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                    conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

                    """ START: Refinement Network """
                    with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None):
                        predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3,
                                                    scope='predict_flow6',
                                                    activation_fn=None)
                        deconv5 = antipad(slim.conv2d_transpose(conv6_1, 512, 4,
                                                                stride=2,
                                                                scope='deconv5'))
                        upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow6to5',
                                                                          activation_fn=None))
                        concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

                        predict_flow5 = slim.conv2d(pad(concat5), 2, 3,
                                                    scope='predict_flow5',
                                                    activation_fn=None)
                        deconv4 = antipad(slim.conv2d_transpose(concat5, 256, 4,
                                                                stride=2,
                                                                scope='deconv4'))
                        upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow5to4',
                                                                          activation_fn=None))
                        concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

                        predict_flow4 = slim.conv2d(pad(concat4), 2, 3,
                                                    scope='predict_flow4',
                                                    activation_fn=None)
                        deconv3 = antipad(slim.conv2d_transpose(concat4, 128, 4,
                                                                stride=2,
                                                                scope='deconv3'))
                        upsample_flow4to3 = antipad(slim.conv2d_transpose(predict_flow4, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow4to3',
                                                                          activation_fn=None))
                        concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

                        predict_flow3 = slim.conv2d(pad(concat3), 2, 3,
                                                    scope='predict_flow3',
                                                    activation_fn=None)
                        deconv2 = antipad(slim.conv2d_transpose(concat3, 64, 4,
                                                                stride=2,
                                                                scope='deconv2'))
                        upsample_flow3to2 = antipad(slim.conv2d_transpose(predict_flow3, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow3to2',
                                                                          activation_fn=None))
                        concat2 = tf.concat([conv_2, deconv2, upsample_flow3to2], axis=3)

                        predict_flow2 = slim.conv2d(pad(concat2), 2, 3,
                                                    scope='predict_flow2',
                                                    activation_fn=None)
                    """ END: Refinement Network """
                    
                    flow = predict_flow2
                    # flow = predict_flow2 * 20.0
                    # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
                    flow = tf.image.resize_bilinear(flow,
                                                    tf.stack([height, width]),
                                                    align_corners=True)

                    return {
                        'predict_flow6': predict_flow6,
                        'predict_flow5': predict_flow5,
                        'predict_flow4': predict_flow4,
                        'predict_flow3': predict_flow3,
                        'predict_flow2': predict_flow2,
                        'flow': flow,
                    }

    def loss(self, flow, predictions):
        flow = flow * 0.05

        losses = []
        INPUT_HEIGHT, INPUT_WIDTH = float(flow.shape[1].value), float(flow.shape[2].value)

        # L2 loss between predict_flow6, blob23 (weighted w/ 0.32)
        predict_flow6 = predictions['predict_flow6']
        size = [predict_flow6.shape[1], predict_flow6.shape[2]]
        downsampled_flow6 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow6, predict_flow6))

        # L2 loss between predict_flow5, blob28 (weighted w/ 0.08)
        predict_flow5 = predictions['predict_flow5']
        size = [predict_flow5.shape[1], predict_flow5.shape[2]]
        downsampled_flow5 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow5, predict_flow5))

        # L2 loss between predict_flow4, blob33 (weighted w/ 0.02)
        predict_flow4 = predictions['predict_flow4']
        size = [predict_flow4.shape[1], predict_flow4.shape[2]]
        downsampled_flow4 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow4, predict_flow4))

        # L2 loss between predict_flow3, blob38 (weighted w/ 0.01)
        predict_flow3 = predictions['predict_flow3']
        size = [predict_flow3.shape[1], predict_flow3.shape[2]]
        downsampled_flow3 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow3, predict_flow3))

        # L2 loss between predict_flow2, blob43 (weighted w/ 0.005)
        predict_flow2 = predictions['predict_flow2']
        size = [predict_flow2.shape[1], predict_flow2.shape[2]]
        downsampled_flow2 = downsample(flow, size)
        losses.append(average_endpoint_error(downsampled_flow2, predict_flow2))

        loss = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005])

        # Return the 'total' loss: loss fns + regularization terms defined in the model
        return tf.losses.get_total_loss()
    
    

            
            
            # Thanks, https://github.com/tensorflow/tensorflow/issues/4079
def LeakyReLU(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)


def average_endpoint_error(labels, predictions):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
        sqrt[sum_across_channels{(X - Y)^2}]
    """
    num_samples = predictions.shape.as_list()[0]
    with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        squared_difference = tf.square(tf.subtract(predictions, labels))
        # sum across channels: sum[(X - Y)^2] -> N, H, W, 1
        loss = tf.reduce_sum(squared_difference, 3, keep_dims=True)
        loss = tf.sqrt(loss)
        return tf.reduce_sum(loss) / num_samples


def pad(tensor, num=1):
    return tensor
    # """
    # Pads the given tensor along the height and width dimensions with `num` 0s on each side
    # """
    # return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")


def antipad(tensor, num=1):
    return tensor
    # """
    # Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
    # padding from the output rather than adding it to the input.
    # """
    # batch, h, w, c = tensor.shape.as_list()
    # return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])


# _downsample = tf.load_op_library(
#     tf.resource_loader.get_path_to_datafile("./ops/build/downsample.so"))


# def downsample(tensor, size):
#     return _downsample.downsample(tensor, size)


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


# Warping layer ---------------------------------
def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
    warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
    # clip value by height/width limitation
    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h-1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)
            
    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11

class WarpingLayer(object):
    def __init__(self, warp_type = 'nearest', name = 'warping'):
        self.warp = warp_type
        self.name = name

    def __call__(self, x, flow):
        # expect shape
        # x:(#batch, height, width, #channel)
        # flow:(#batch, height, width, 2)
        with tf.name_scope(self.name) as ns:
            assert self.warp in ['nearest', 'bilinear']
            if self.warp == 'nearest':
                x_warped = nearest_warp(x, flow)
            else:
                x_warped = bilinear_warp(x, flow)
            return x_warped