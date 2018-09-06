"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license 
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Author: Zhaoyang Lv
"""

import torch
import numpy as np

def generate_index_grid(h, w):
    """ Generate a meshgrid
    :param height of the image
    :param H of the image
    """
    u = torch.arange(0, w).cuda()
    v = torch.arange(0, h).cuda()

    return u.repeat(h, 1), v.repeat(w, 1).t()

def torch_rgbd2uvd(color, depth, fx, fy, cx, cy):
    """ Generate the u, v, inverse depth point cloud, 
    given color, depth and intrinsic parameters
    
    The input image dimension is as following:

    :param Color dim B * 3 * H * W
    :param Depth dim B * H * W
    :param fx dim B
    :param fy dim B
    :param cx dim B
    :param cy dim B
    """
    B, C, H, W = color.size()

    u_, v_ = generate_index_grid(H, W)

    x_ = (u_ - cx) / fx
    y_ = (v_ - cy) / fy

    inv_z_ = 1.0 / depth

    uvdrgb = torch.cat(( x_.view(B,1,H,W), y_.view(B,1,H,W), 
        inv_z_.view(B,1,H,W), color ), 1)

    return uvdrgb

def torch_depth2xyz(depth, fx, fy, cx, cy):
    """ Generate the xyz point cloud 
    :param Depth dim B * H * W
    :param fx dim B
    :param fy dim B
    :param cx dim B
    :param cy dim B
    """
    B, C, H, W = depth.size()

    u_, v_ = generate_index_grid(H, W)

    x_ = depth * (u_ - cx) / fx
    y_ = depth * (v_ - cy) / fy

    xyz = torch.cat((x_.view(B,1,H,W), y_.view(B,1,H,W), depth.view(B,1,H,W)), 1)

    return xyz

_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def torch_euler2mat(ai, aj, ak, axes='sxyz'):
    """ A gpu version euler2mat from transform3d:
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    :param ai : First rotation angle (according to `axes`).
    :param aj : Second rotation angle (according to `axes`).
    :param ak : Third rotation angle (according to `axes`).
    :param axes : Axis specification; one of 24 axis sequences as string or encoded tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    mat : array-like shape (B, 3, 3)

    Tested w.r.t. transforms3d.euler module
    """

    B = ai.size()[0]

    cos = torch.cos
    sin = torch.sin

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    order = [i, j, k]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    # M = torch.zeros(B, 3, 3).cuda()
    if repetition:
        c_i = [cj, sj*si, sj*ci]
        c_j = [sj*sk, -cj*ss+cc, -cj*cs-sc]
        c_k = [-sj*ck, cj*sc+cs, cj*cc-ss]
    else:
        c_i = [cj*ck, sj*sc-cs, sj*cc+ss]
        c_j = [cj*sk, sj*ss+cc, sj*cs-sc]
        c_k = [-sj, cj*si, cj*ci]

    def permute(X): # sort X w.r.t. the axis indices
        return [ x for (y, x) in sorted(zip(order, X)) ]

    c_i = permute(c_i)
    c_j = permute(c_j)
    c_k = permute(c_k)

    r =[torch.stack(c_i, 1),
        torch.stack(c_j, 1),
        torch.stack(c_k, 1)]
    r = permute(r)

    return torch.stack(r, 1)


def np_depth2flow(depth, K0, T0, K1, T1):
    """ Numpy implementation.
    Estimate the ego-motion flow given two frames depths and transformation matrices. 
    The output is an ego-motion flow in 2D (2*H*W).

    :param the depth map of the reference frame
    :param the intrinsics of the reference frame
    :param the camera coordinate of the reference frame
    :param the intrinsics of the target frame
    :param the camera coordinate of the target frame
    """
    rows, cols = depth.shape
    u_mat = np.repeat(np.array(range(0, cols)).reshape(1, cols), rows, axis=0)
    v_mat = np.repeat(np.array(range(0, rows)).reshape(rows, 1), cols, axis=1)

    # inv_k = [ 1/f_x,  0, -c_x/f_x, 0;
    #           0,  1/f_y, -c_y/f_y, 0;
    #           0,      0,  1,       0;
    #           0,      0,  0,       1]
    inv_K = np.eye(4)
    inv_K[0,0], inv_K[1,1] = 1.0 / K0[0], 1.0 / K0[1]
    inv_K[0,2], inv_K[1,2] = -K0[2] / K0[0], -K0[3] / K0[1]

    # the point cloud move w.r.t. the inverse of camera transform
    K = np.eye(4)
    K[0,0], K[1,1], K[0,2], K[1,2] = K1[0], K1[1], K1[2], K1[3]

    if T0.shape != (4,4):
        T0, T1 = to_homogenous(T0), to_homogenous(T1)
    T = reduce(np.dot, [K, T1, np.linalg.inv(T0), inv_K])

    # blender's coordinate is different from sintel
    ones = np.ones((rows, cols))
    z = depth
    x = depth * u_mat
    y = depth * v_mat
    p4d = np.dstack((x, y, z, ones)).transpose((2,0,1))
    p4d_t = np.tensordot(T, p4d, axes=1)

    x_t, y_t, z_t, w_t = np.split(p4d_t, 4)
    # homogenous to cartsian
    x_t, y_t, z_t = x_t[0] / w_t[0], y_t[0] / w_t[0], z_t[0] / w_t[0]

    u_t_mat = x_t / z_t
    v_t_mat = y_t / z_t

    # this is the final ego-motion flow
    d_u = u_t_mat - u_mat
    d_v = v_t_mat - v_mat

    return  np.stack((d_u, d_v), axis=0), u_t_mat, v_t_mat, z_t
