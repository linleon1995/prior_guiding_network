#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 17:15:34 2019

@author: acm528_02
"""

import numpy as np

def get_pixel_value_np(img, x, y):
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
    shape = x.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    
    batch_idx = np.arange(batch_size)
    batch_idx = np.reshape(batch_idx, (batch_size, 1, 1))
    b = np.tile(batch_idx, (1, height, width))
    
    return img[b,y,x]
#    indices = np.stack([b, y, x], 3)
#    indices = np.int32(indices)

#    return np.take(img, indices)


def affine_grid_generator_np(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = np.shape(theta)[0]

    # create normalized 2D grid
    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    x_t, y_t = np.meshgrid(x, y)

    # flatten
    x_t_flat = np.reshape(x_t, [-1])
    y_t_flat = np.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = np.ones_like(x_t_flat)
    sampling_grid = np.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = np.expand_dims(sampling_grid, axis=0)
    sampling_grid = np.tile(sampling_grid, np.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = np.float32(theta)
    sampling_grid = np.float32(sampling_grid)

    # transform the sampling grid - batch multiply
    batch_grids = np.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = np.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def bilinear_sampler_np(img, x, y):
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
    import matplotlib.pyplot as plt
#    plt.imshow(img[0,...,0])
#    plt.show()
    H = np.shape(img)[1]
    W = np.shape(img)[2]
    max_y = np.int32(H-1)
    max_x = np.int32(W-1)
    zero = np.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = np.float32(x)
    y = np.float32(y)
    x = 0.5 * ((x + 1.0) * np.float32(max_x-1))
    y = 0.5 * ((y + 1.0) * np.float32(max_y-1))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.int32(np.floor(x))
    x1 = x0 + 1
    y0 = np.int32(np.floor(y))
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value_np(img, x0, y0)
    Ib = get_pixel_value_np(img, x0, y1)
    Ic = get_pixel_value_np(img, x1, y0)
    Id = get_pixel_value_np(img, x1, y1)

    # recast as float for delta calculation
    x0 = np.float32(x0)
    x1 = np.float32(x1)
    y0 = np.float32(y0)
    y1 = np.float32(y1)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = np.expand_dims(wa, axis=3)
    wb = np.expand_dims(wb, axis=3)
    wc = np.expand_dims(wc, axis=3)
    wd = np.expand_dims(wd, axis=3)

    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    
#    print('a')
#    plt.imshow(wa[0,...,0])
#    plt.show()
#    print('b')
#    plt.imshow(Ia[0,...,0])
#    plt.show()
#    plt.show()
#    print('c')
#    plt.imshow(out[0,...,0])
#    plt.show()
    
    return out


def nearlist_sampler_np(img, x, y):
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
    H = np.shape(img)[1]
    W = np.shape(img)[2]
    max_y = np.int32(H-1)
    max_x = np.int32(W-1)
    zero = np.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = np.float32(x)
    y = np.float32(y)
    x = 0.5 * ((x + 1.0) * np.float32(max_x-1))
    y = 0.5 * ((y + 1.0) * np.float32(max_y-1))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.int32(np.floor(x))
    x1 = x0 + 1
    y0 = np.int32(np.floor(y))
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value_np(img, x0, y0)
    Ib = get_pixel_value_np(img, x0, y1)
    Ic = get_pixel_value_np(img, x1, y0)
    Id = get_pixel_value_np(img, x1, y1)

    # recast as float for delta calculation
    x0 = np.float32(x0)
    x1 = np.float32(x1)
    y0 = np.float32(y0)
    y1 = np.float32(y1)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = np.expand_dims(wa, axis=3)
    wb = np.expand_dims(wb, axis=3)
    wc = np.expand_dims(wc, axis=3)
    wd = np.expand_dims(wd, axis=3)

    w_max = np.concatenate([wa, wb, wc, wd], axis=3)
    w_max = np.argmax(w_max, axis=3)
    id = np.eye(4)
    w_max = id[w_max]
    
    wa = w_max[...,0:1]
    wb = w_max[...,1:2]
    wc = w_max[...,2:3]
    wd = w_max[...,3:4]
    
    wa = np.float32(wa)
    wb = np.float32(wb)
    wc = np.float32(wc)
    wd = np.float32(wd)
    
    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return out

