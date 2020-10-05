import tensorflow as tf
from core import features_extractor, utils, resnet_v1_beta, preprocess_utils
import common

slim = tf.contrib.slim
mlp = utils.mlp


def pgb_network(images,
                raw_height,
                raw_width,
                model_options,
                prior_segs=None,
                num_class=None,
                prior_slice=None,
                batch_size=None,
                z_label_method=None,
                z_class=None,
                fusion_slice=None,
                drop_prob=None,
                reuse=None,
                is_training=None,
                scope=None,
                **kwargs,
                ):
    """
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    Args:
        images:
        prior_segs: [1,H,W,Class,Layer]
    Returns:
        segmentations:
    """
    output_dict = {}
    weight_decay = kwargs.pop("weight_decay", None)
    fusions = kwargs.pop("fusions", None)
    out_node = kwargs.pop("out_node", None)
    guid_encoder = kwargs.pop("guid_encoder", None)
    z_model = kwargs.pop("z_model", None)
    guidance_loss_name = kwargs.pop("guidance_loss_name", None)
    stage_pred_loss_name = kwargs.pop("stage_pred_loss_name", None)
    guid_conv_nums = kwargs.pop("guid_conv_nums", None)
    guid_conv_type = kwargs.pop("guid_conv_type", None)
    seq_length = kwargs.pop("seq_length", None)
    cell_type = kwargs.pop("cell_type", None)

    if seq_length > 1:
        n, t, h, w, c = preprocess_utils.resolve_shape(images, rank=5)
        images = tf.reshape(images, [-1, h, w, c])
    # Produce Prior
    if prior_segs is not None:
        prior_from_data = tf.tile(prior_segs[...,0], [seq_length,1,1,1])

    if guid_encoder in ("early", "p_embed_prior"):
        in_node = tf.concat([images, prior_from_data], axis=3)
    elif guid_encoder in ("late", "image_only", "p_embed"):
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
    if z_label_method is not None and z_model is not None:
        if z_label_method == "reg":
            multi_task_node = 1
        elif z_label_method == "cls":
            if z_class is None:
                raise ValueError("Unknown Z class")
            multi_task_node = z_class
        else:
            raise ValueError("Unknown Z label method")
        z_logits = predict_z_dimension(layers_dict["low_level5"], out_node=multi_task_node,
                                       extractor_type="simple")
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
                # Refined by Decoder
                if guid_encoder in ("early", "image_only"):
                    embed_latent = slim.conv2d(layers_dict["low_level5"], out_node, kernel_size=[1,1], scope="guidance_embedding")
                # elif guid_encoder in ("p_embed", "p_embed_prior"):
                #     if z_class is None:
                #         raise ValueError("Unknown Z class for prior embedding weighting")
                #     with tf.variable_scope('encoded_priors', scope):
                #         prior_list = []
                #         for i in range(z_class):
                #             prior_list.append(slim.conv2d(layers_dict["low_level5"], out_node, kernel_size=[3,3],
                #                                           activation_fn=None, scope="prior_encoder"+str(i)))
                #     tf.add_to_collection("prior_list", prior_list)
                #     p = tf.stack(prior_list, axis=4)
                #     z = tf.reshape(tf.nn.softmax(z_logits, axis=1), [-1,1,1,z_class,1])
                #     embed_latent = tf.squeeze(tf.matmul(p, z), axis=4)

                # tf.add_to_collection("guid_f", prior_seg)

                if guidance_loss_name is not None:
                    # TODO: PGN_v1
                    # if z_logits is not None and :
                    #     z_pred = tf.nn.softmax(z_logits, axis=1)

                    # else:
                    prior_pred = slim.conv2d(layers_dict["low_level5"], num_class, kernel_size=[1,1], stride=1, activation_fn=None, scope='prior_pred_pred_class%d' %num_class)
                    # prior_pred = slim.conv2d(embed_latent, num_class, kernel_size=[1,1], stride=1, activation_fn=None, scope='prior_pred_pred_class%d' %num_class)
                    # tf.add_to_collection("stage_pred", prior_pred)
                    output_dict[common.GUIDANCE] = prior_pred

                    if "softmax" in guidance_loss_name:
                        prior_pred = tf.nn.softmax(prior_pred, axis=3)
                    elif "sigmoid" in guidance_loss_name:
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
    # logits, preds = refine_model.model()
    # layers_dict.update(preds)

    # Sequential Model for slice fusion
    if seq_length is not None:
        if seq_length > 1:
            logits = tf.reshape(logits, [n, t, h, w, num_class])
            logits = utils.seq_model(logits, raw_height, raw_width, num_class, weight_decay, is_training, cell_type)

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
        # TODO: region based extractor
        elif extractor_type == "region":
            pass
        else:
            raise ValueError("Unknown Extractor Type")
    return z_logits

