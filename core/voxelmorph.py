import tensorflow as tf
from core import stn
slim = tf.contrib.slim
bilinear_sampler = stn.bilinear_sampler


def voxel_deformable_transform(moving_images, 
                               fixed_images, 
                               moving_segs=None,
                               unet_num_layers=4,
                               unet_features_depth=32,
                               unet_output_strides=16,
                               unet_output_dims=2,
                               is_training=None,
                               scope=None):
  """
  VoxelMorph: A Learning Framework for Deformable Medical Image Registration
  Args:
      moving_images: A 3D images
      fixed_images:
      moving_segs:
      fixed_segs:
      model_options:
  Returns:
      transformed_images: A 3D images which transformed by registration field
      transformed_segs: (optional) A 3D segmentations which transformed by registration field
  """
  with tf.variable_scope(scope, 'voxelmorph', [moving_images, fixed_images]) as sc:
    inputs = tf.concat([moving_images, fixed_images], axis=3)
    # TODO: parameter
    inputs.set_shape([None, 256, 256, 2])
    field, layer_dict = simplify_unet(inputs=inputs, 
                                      output_dims=unet_output_dims,
                                      num_layers=unet_num_layers, 
                                      features_depth=unet_features_depth,
                                      output_strides=unet_output_strides,
                                      is_training=is_training)                                                 
    transformed_images = deformable_transform(moving_images, field)

    if moving_segs is not None:
      transformed_segs = deformable_transform(moving_segs, field)
      return transformed_images, transformed_segs
    else:
      return transformed_images
    
    
def simplify_unet(inputs, 
                  output_dims, 
                  num_layers=4, 
                  features_depth=32, 
                  output_strides=16, 
                  is_training=None,
                  scope=None,):
  """
  VoxelMorph: A Learning Framework for Deformable Medical Image Registration
  Args:
    moving_images: A 3D images
    fixed_images:
    moving_segs:
    fixed_segs:
    model_options:
  Returns:
    transformed_images: A 3D images which transformed by registration field
    transformed_segs: (optional) A 3D segmentations which transformed by registration field
  """
  # TODO: variable activation_funce (Leaky_relu)
  # TODO: 3d conv
  # TODO: layer_dict (1. do we need 2. if we need, complete with all layers)
  # TODO: endpoints or layer_dict
  # TODO: bool var to decide existence of last three layers??
  
  with tf.variable_scope(scope, 'simplify_unet', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d],
                        outputs_collections=end_points_collection):
      if is_training is not None:
        arg_scope = slim.arg_scope([slim.batch_norm], is_training=is_training)
      else:
        arg_scope = slim.arg_scope([])
      with arg_scope:
        layer_dict = {}
        net = inputs
        for layer in range(num_layers):
          if layer == 0:
            depth = 16
          else:
            depth = features_depth
          layer_dict['downsampling'+str(layer)] = net
          net = slim.conv2d(net, features_depth, [1, 1], stride=1, scope='downconv'+str(layer))
          net = tf.nn.relu(net)
          if layer < num_layers-1:
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool'+str(layer))
          else:
            layer_dict['intermedia'] = net
        
        for layer in range(num_layers-1,-1,-1):
          layer_dict['upsampling'+str(layer)] = net
          net = tf.concat([net, layer_dict['upsampling'+str(layer)]], axis=3)
          net = slim.conv2d(net, features_depth, [1, 1], stride=1, scope='upconv'+str(layer))
          net = tf.nn.relu(net)
          new_h = 2 * net.get_shape().as_list()[1]
          new_w = 2 * net.get_shape().as_list()[2]
          net = tf.image.resize_bilinear(net, [new_h, new_w], name='bilinear'+str(layer))
          
        net = slim.conv2d(net, features_depth//2, [1, 1], stride=1, scope='conv_output1')
        net = tf.nn.relu(net)
        net = slim.conv2d(net, features_depth//2, [1, 1], stride=1, scope='conv_output2')
        net = tf.nn.relu(net)
        outputs = slim.conv2d(net, output_dims, [1, 1], stride=1, scope='outputs') 
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)  
  return outputs, layer_dict



def deformable_transform(input_fmap, transform_field):
  """
  VoxelMorph: A Learning Framework for Deformable Medical Image Registration
  Args:
    moving_images: A 3D images
    fixed_images:
    moving_segs:
    fixed_segs:
    model_options:
  Returns:
    transformed_images: A 3D images which transformed by registration field
    transformed_segs: (optional) A 3D segmentations which transformed by registration field
  """
  # TODO: general expression
  # def _2d_transform(input_fmp, transform_field):
  #   x_s = transform_field[:, 0, :, :]
  #   y_s = transform_field[:, 1, :, :]
  #   out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
  #   return out_fmap
  
  # def _3d_transform(input_fmp, transform_field):  
  #   x_s = transform_field[:, 0, :, :]
  #   y_s = transform_field[:, 1, :, :]
  #   z_s = transform_field[:, 2, :, :]
  #   out_fmap = bilinear_sampler_multi_dims(input_fmap, x=x_s, y=y_s, z=z_s)
  #   return out_fmap

  # c = tf.equal(tf.rank(input_fmap), 4)
  # out_fmap = tf.cond(c, lambda: _2d_transform(input_fmap, transform_field), lambda: _3d_transform(input_fmap, transform_field))
    
  x_s = transform_field[:, 0, :, :]
  y_s = transform_field[:, 1, :, :]
  out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
  return out_fmap




# def deformable_transformer_network(input_fmap, transform_field, out_dims=None, **kwargs):
#     """
#     Spatial Transformer Network layer implementation as described in [1].
#     The layer is composed of 3 elements:
#     - localization_net: takes the original image as input and outputs
#       the parameters of the affine transformation that should be applied
#       to the input image.
#     - affine_grid_generator: generates a grid of (x,y) coordinates that
#       correspond to a set of points where the input should be sampled
#       to produce the transformed output.
#     - bilinear_sampler: takes as input the original image and the grid
#       and produces the output image using bilinear interpolation.
#     Input
#     -----
#     - input_fmap: output of the previous layer. Can be input if spatial
#       transformer layer is at the beginning of architecture. Should be
#       a tensor of shape (B, H, W, C).
#     - theta: affine transform tensor of shape (B, 6). Permits cropping,
#       translation and isotropic scaling. Initialize to identity matrix.
#       It is the output of the localization network.
#     Returns
#     -------
#     - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
#     Notes
#     -----
#     [1]: 'Spatial Transformer Networks', Jaderberg et. al,
#          (https://arxiv.org/abs/1506.02025)
#     """
#     # TODO: input can be arbituary dimensions
#     # out_dims??
#     # grab input dimensions
#     # B = tf.shape(input_fmap)[0]
#     # H = tf.shape(input_fmap)[1]
#     # W = tf.shape(input_fmap)[2]
#     # if data_dims == 3:
#     #   D = tf.shape(input_fmap)[3]
    

#     # # generate grids of same size or upsample/downsample if specified
#     # if out_dims:
#     #     out_H = out_dims[0]
#     #     out_W = out_dims[1]
#     #     batch_grids = deformable_grid_generator(out_H, out_W, transform_field)
#     # else:
#     #     batch_grids = deformable_grid_generator(H, W, transform_field)

#     # x_s = batch_grids[:, 0, :, :]
#     # y_s = batch_grids[:, 1, :, :]
    

#     # sample input with grid to get output
#     x_s = transform_field[:, 0, :, :]
#     y_s = transform_field[:, 1, :, :]
#     z_s = transform_field[:, 2, :, :]
#     out_fmap = bilinear_sampler_multi_dims(input_fmap, x=x_s, y=y_s, z=z_s)

#     return out_fmap



def bilinear_sampler_multi_dims(inputs, **kwargs):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    # def processing_grid(x, length):
    #   max_x = tf.cast(length - 1, 'int32')
    #   zero = tf.zeros([], dtype='int32')
      
    #   x = tf.cast(x, 'float32')
    #   x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    #   x0 = tf.cast(tf.floor(x), 'int32')
    #   x1 = x0 + 1
    #   x0 = tf.clip_by_value(x0, zero, max_x)
    #   x1 = tf.clip_by_value(x1, zero, max_x)
    #   return [x0, x1]
    
    # def get_output(grid_list):
    #   for grid in grid_list:
    #     for value in grid:
    #       value
          
    # total_grid = []
    # for i, grid in enumerate(*args):
    #   v = processing_grid(grid, tf.shape(inputs)[i])
    #   total_grid.append(v)
    
    # get_output
    
  
    H = tf.shape(inputs)[1]
    W = tf.shape(inputs)[2]
    D = tf.shape(inputs)[3]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    max_z = tf.cast(D - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    x = kwargs['x']
    y = kwargs['y']
    z = kwargs['z']
    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    z = tf.cast(z, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))
    z = 0.5 * ((z + 1.0) * tf.cast(max_z-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    # get pixel value at corner coords
    Ia = get_pixel_value(inputs, x0, y0, z0)
    Ib = get_pixel_value(inputs, x0, y1, z0)
    Ic = get_pixel_value(inputs, x1, y0, z0)
    Id = get_pixel_value(inputs, x1, y1, z0)
    Ie = get_pixel_value(inputs, x0, y0, z1)
    If = get_pixel_value(inputs, x0, y1, z1)
    Ig = get_pixel_value(inputs, x1, y0, z1)
    Ih = get_pixel_value(inputs, x1, y1, z1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')
    z0 = tf.cast(z0, 'float32')
    z1 = tf.cast(z1, 'float32')
    
    # calculate deltas
    wa = (x1-x) * (y1-y) * (z1-z)
    wb = (x1-x) * (y-y0) * (z1-z)
    wc = (x-x0) * (y1-y) * (z1-z)
    wd = (x-x0) * (y-y0) * (z1-z)
    we = (x1-x) * (y1-y) * (z-z0)
    wf = (x1-x) * (y-y0) * (z-z0)
    wg = (x-x0) * (y1-y) * (z-z0)
    wh = (x-x0) * (y-y0) * (z-z0)
    
    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih])
    return out
  
  
def get_pixel_value(img, x, y, z=None):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    if z is not None:
      depth = shape[3]

    batch_idx = tf.range(0, batch_size)
    if z is not None:
      batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1))
      b = tf.tile(batch_idx, (1, height, width, depth))
      indices = tf.stack([b, y, x, z], 3)
    else:
      batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
      b = tf.tile(batch_idx, (1, height, width))
      indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

