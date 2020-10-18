import tensorflow as tf
from core import features_extractor, utils, preprocess_utils
import common

slim = tf.contrib.slim
mlp = utils.mlp


def pgb_network(images,
                raw_height,
                raw_width,
                model_options,
                prior_segs=None,
                num_class=None,
                z_loss_name=None,
                mt_output_node=None,
                fusion_slice=None,
                drop_prob=None,
                reuse=None,
                is_training=None,
                scope=None,
                **kwargs,
                ):
    """
    Prior Guidiing Network
    """
    output_dict = {}
    weight_decay = kwargs.pop("weight_decay", None)
    fusions = kwargs.pop("fusions", None)
    out_node = kwargs.pop("out_node", None)
    guid_encoder = kwargs.pop("guid_encoder", None)
    z_model = kwargs.pop("z_model", "simple")
    guid_loss_name = kwargs.pop("guid_loss_name", None)
    stage_pred_loss_name = kwargs.pop("stage_pred_loss_name", None)
    guid_conv_nums = kwargs.pop("guid_conv_nums", None)
    guid_conv_type = kwargs.pop("guid_conv_type", None)
    seq_length = kwargs.pop("seq_length", None)
    cell_type = kwargs.pop("cell_type", None)

    # Flatten time dimension for slice-wise feature extracting
    if seq_length > 1:
        images = tf.split(images, num_or_size_splits=seq_length, axis=1)
        images = tf.concat(images, axis=0)
        images = tf.squeeze(images, axis=1)
        
    # Produce Prior
    if prior_segs is not None:
        prior_from_data = tf.tile(prior_segs[...,0], [seq_length,1,1,1])

    if guid_encoder in ("early"):
        in_node = tf.concat([images, prior_from_data], axis=3)
    elif guid_encoder in ("image_only"):
        in_node = images

    # Feature Extractor (Encoder)
    features, end_points = features_extractor.extract_features(images=in_node,
                                                               output_stride=model_options.output_stride,
                                                               multi_grid=model_options.multi_grid,
                                                               model_variant=model_options.model_variant,
                                                               reuse=reuse,
                                                               is_training=is_training,
                                                               fine_tune_batch_norm=model_options.fine_tune_batch_norm,
                                                               preprocessed_images_dtype=model_options.preprocessed_images_dtype)

    layers_dict = {"low_level5": features,
                   "low_level4": end_points["resnet_v1_50/block3"],
                   "low_level3": end_points["resnet_v1_50/block2"],
                   "low_level2": end_points["resnet_v1_50/block1"],
                   "low_level1": end_points["resnet_v1_50/conv1_3"]}

    # Multi-task
    if z_loss_name is not None:
        z_logits = predict_z_dimension(layers_dict["low_level5"], out_node=mt_output_node,
                                       extractor_type=z_model)
        output_dict[common.OUTPUT_Z] = z_logits

    # Guidance Generating
    with slim.arg_scope([slim.batch_norm],
                        is_training=is_training):
        with slim.arg_scope([slim.conv2d],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.initializers.he_normal(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          normalizer_fn=slim.batch_norm):
            if "guid" in fusions or "guid_class" in fusions or "guid_uni" in fusions or "context_att" in fusions or "self_att" in fusions:
                if guid_loss_name is not None:
                    # Apply PGN-v1 if z (longitudinal) prediction exist
                    if common.OUTPUT_Z in output_dict:
                        z_pred = tf.nn.softmax(z_logits, axis=1)
                        # TODO: get v1 prior and use fusion slices properly
                        prior_pred = prior_from_data * z_pred
                    else:
                        prior_pred = slim.conv2d(layers_dict["low_level5"], num_class, kernel_size=[1,1], stride=1, activation_fn=None, scope='prior_pred_pred_class%d' %num_class)
                        embed_latent = slim.conv2d(layers_dict["low_level5"], out_node, kernel_size=[1,1], scope="guidance_embedding")
                    output_dict[common.GUIDANCE] = prior_pred

                    if "softmax" in guid_loss_name:
                        prior_pred = tf.nn.softmax(prior_pred, axis=3)
                    elif "sigmoid" in guid_loss_name:
                        prior_pred = tf.nn.sigmoid(prior_pred)
                else:
                    prior_pred = None
            else:
                embed_latent = None
                prior_pred = None

    # Refining Model (Decoder)
    refine_model = utils.Refine(layers_dict, fusions, prior_seg=embed_latent, stage_pred_loss_name=stage_pred_loss_name,
                                prior_pred=prior_pred, guid_conv_nums=guid_conv_nums, guid_conv_type=guid_conv_type,
                                embed_node=out_node, weight_decay=weight_decay, is_training=is_training, num_class=num_class,
                                **kwargs)
    logits = refine_model.model()

    # Sequential Model for slice fusion
    if seq_length is not None:
        if seq_length > 1:
            # logits = tf.reshape(logits, [n, t, h, w, num_class])
            logits = tf.expand_dims(logits, axis=1)
            logits = tf.split(logits, num_or_size_splits=seq_length, axis=0)
            logits  = tf.concat(logits , axis=1)
            logits, _ = utils.seq_model(logits, raw_height, raw_width, num_class, weight_decay, is_training, cell_type)

    if drop_prob is not None:
        logits = tf.nn.dropout(logits, rate=drop_prob)
    output_dict[common.OUTPUT_TYPE] = logits

    trainable_vars = tf.trainable_variables()
    for v in trainable_vars:
      print(30*"-", v.name)
    return output_dict, layers_dict


def predict_z_dimension(feature, out_node, extractor_type):
    with tf.variable_scope('multi_task_branch'):
        if extractor_type == "simple":
            gap = tf.reduce_mean(feature, axis=[1,2], keep_dims=False)
            z_logits = mlp(gap, output_dims=out_node, num_layers=2,
                           decreasing_root=16, scope='z_info_extractor')
        # TODO: region based extractor, consider to add in spatial information
        elif extractor_type == "region":
            pass
        else:
            raise ValueError("Unknown Extractor Type")
    return z_logits

