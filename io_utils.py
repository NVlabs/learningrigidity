""" 
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license 
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Author: Zhaoyang Lv

Partially refer to:
=============================================================================

The I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark.

CHANGELOG:
v1.0 (2015/02/03): First release

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany
"""

# Requirements: Numpy as PIL/Pillow
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib.colors as color

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'

def flow_visualize(flow, max_range = 1e3):
    du = flow[0]
    dv = flow[1]
    [h,w] = du.shape
    max_flow = min(max_range, np.max(np.sqrt(du * du + dv * dv)))
    img = np.ones((h, w, 3), dtype=np.float64)
    # angle layer
    img[:, :, 0] = (np.arctan2(dv, du) / (2 * np.pi) + 1) % 1.0
    # magnitude layer, normalized to 1
    img[:, :, 1] = np.sqrt(du * du + dv * dv) / (max_flow + 1e-8)
    # phase layer
    #img[:, :, 2] = valid
    # convert to rgb
    img = color.hsv_to_rgb(img)
    # remove invalid point
    img[:, :, 0] = img[:, :, 0]
    img[:, :, 1] = img[:, :, 1]
    img[:, :, 2] = img[:, :, 2]
    return img

def flow_read_from_flo(filename):
    """ Read optical flow from file, return (U,V) tuple.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' flow_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' flow_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    tmp = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width*2))
    u = tmp[:,np.arange(width)*2]
    v = tmp[:,np.arange(width)*2 + 1]
    return u,v

def flow_write(filename,uv,v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def depth_write(filename, depth):
    """ Write depth to file. """
    height,width = depth.shape[:2]
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)

    depth.astype(np.float32).tofile(f)
    f.close()

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.

    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

def cam_write(filename, M, N):
    """ Write intrinsic matrix M and extrinsic matrix N to file. """
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    M.astype('float64').tofile(f)
    N.astype('float64').tofile(f)
    f.close()

def image_with_mask(image, mask):
    """ return the masked image visualization
    """
    H, W = mask.shape
    color_mask = image.copy()
    color_mask[mask>0] = [125, 0, 0]
    hsv_mask = color.rgb_to_hsv(color_mask)
    I_hsv = color.rgb_to_hsv(image)
    I_hsv[..., 0] = hsv_mask[..., 0]
    I_hsv[..., 1] = hsv_mask[..., 1] * 0.6
    return color.hsv_to_rgb(I_hsv)
