import os
import re
import sys
import glob
import json
import time
import numpy as np 
# import skimage
# import skimage.io as imgio
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary
   
def crf_inference(feat,img,crf_config,categorys_num,gt_prob=0.7,use_log=False):
    '''
    feat: the feature map of cnn, shape [h,w,c] , shape [h,w], float32
    img: the origin img, shape [h,w,3], uint8
    crf_config: {"g_sxy":3,"g_compat":3,"bi_sxy":5,"bi_srgb":5,"bi_compat":10,"iterations":5}
    '''
    img = img.astype(np.uint8)
    pred_or_feat = feat.astype(np.float32)
    h,w = img.shape[0:2]
    crf = dcrf.DenseCRF2D(w,h,categorys_num)

    if use_log is True:
        feat = np.exp(feat -np.max(feat,axis=2,keepdims=True))
        feat /= np.sum(feat,axis=2,keepdims=True)
        unary = -np.log(feat)
    else:
        unary = -feat
    unary = np.reshape(unary,(-1,categorys_num))
    unary = np.swapaxes(unary,0,1)
    unary = np.copy(unary,order="C")
    crf.setUnaryEnergy(unary)

    # pairwise energy
    crf.addPairwiseGaussian( sxy=crf_config["g_sxy"], compat=crf_config["g_compat"] )
    crf.addPairwiseBilateral( sxy=crf_config["bi_sxy"], srgb=crf_config["bi_srgb"], rgbim=img, compat=crf_config["bi_compat"] )
    Q = crf.inference( crf_config["iterations"] )
    Q = np.array(Q)
    Q = np.reshape(Q,[categorys_num,h,w])
    Q = np.transpose(Q,axes=[1,2,0]) # new shape: [h,w,c]
    return Q


#def crf_inference(feat,img,crf_config,categorys_num,gt_prob=0.7,use_log=False, shape=(32,32)):
#    '''
#    feat: the feature map of cnn, shape [h,w,c] , shape [h,w], float32
#    img: the origin img, shape [h,w,3], uint8
#    crf_config: {"g_sxy":3,"g_compat":3,"bi_sxy":5,"bi_srgb":5,"bi_compat":10,"iterations":5}
#    '''
#    img = img.astype(np.float32)
#    pred_or_feat = feat.astype(np.float32)
#    h,w = img.shape[0:2]
#    crf = dcrf.DenseCRF2D(w,h,categorys_num)
#
#    if use_log is True:
#        feat = np.exp(feat -np.max(feat,axis=2,keepdims=True))
#        feat /= np.sum(feat,axis=2,keepdims=True)
#        unary = -np.log(feat)
#    else:
#        unary = -feat
#    unary = np.reshape(unary,(-1,categorys_num))
#    unary = np.swapaxes(unary,0,1)
#    unary = np.copy(unary,order="C")
#    crf.setUnaryEnergy(unary)
#
#    # pairwise energy
##    crf.addPairwiseGaussian( sxy=crf_config["g_sxy"], compat=crf_config["g_compat"] )
#    gaussian_f = create_pairwise_gaussian(sdims=(crf_config["g_sxy"],crf_config["g_sxy"]), shape=shape)
#    crf.addPairwiseEnergy(gaussian_f, compat=crf_config["g_compat"],
#                    kernel=dcrf.DIAG_KERNEL,
#                    normalization=dcrf.NORMALIZE_SYMMETRIC)
#    bilateral_f = create_pairwise_bilateral(sdims=(crf_config["bi_sxy"],crf_config["bi_sxy"]), 
#                                            schan=(crf_config["bi_srgb"]), 
#                                            img=img, 
#                                            chdim=-1)
#    crf.addPairwiseEnergy(bilateral_f, compat=crf_config["bi_compat"],
#                     kernel=dcrf.DIAG_KERNEL,
#                     normalization=dcrf.NORMALIZE_SYMMETRIC)
#
#    Q = crf.inference( crf_config["iterations"] )
#    Q = np.array(Q)
#    Q = np.reshape(Q,[categorys_num,h,w])
#    Q = np.transpose(Q,axes=[1,2,0]) # new shape: [h,w,c]
#    return Q
