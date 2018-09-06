"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license 
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Author: Zhaoyang Lv
"""

import os, sys, time
import numpy as np

sys.path.append('external_packages/flow2pose/build')
from pyFlow2Pose import pyFlow2Pose

from scipy.misc import imsave
from scipy import ndimage
from skimage.morphology import dilation, square, erosion

def forward_backward_consistency(F_f, F_b, threshold):
    """ get the mask that is foreward-backward consistent
    """
    u_b = F_b[0]
    v_b = F_b[1]
    u_f = F_f[0]
    v_f = F_f[1]
    [H,W] = np.shape(u_b)
    [x,y] = np.meshgrid(np.arange(0,W),np.arange(0,H))
    x2 = x + u_b
    y2 = y + v_b
    # Out of boundary
    B = (x2 > W-1) | (y2 > H-1) | (x2 < 0) | (y2 <0)
    u = ndimage.map_coordinates(u_f, [y2, x2])
    v = ndimage.map_coordinates(v_f, [y2, x2])
    u_inv = u
    v_inv = v
    dif = np.zeros((H, W), dtype=np.float64)
    dif = ((u_b + u_inv)**2 + (v_b + v_inv)**2)**0.5
    mask = (dif < threshold)
    mask = mask | B

    return mask

def depth2pointcloud_batch(K0, depth0):
    """ Transfer a pair of depth images into point cloud
    """
    B, _, H, W = depth0.shape

    inv_K0 = np.linalg.inv(K0)

    u_mat = np.repeat(np.array(range(0, W)).reshape(1, W), H, axis=0)
    v_mat = np.repeat(np.array(range(0, H)).reshape(H, 1), W, axis=1)

    uvd_map_0 = np.concatenate((depth0 * u_mat, depth0 * v_mat, depth0), axis=1)

    vertex_map_0 = uvd_map_0.copy()

    for idx in range(B):
        vertex_map_0[idx] = np.tensordot(inv_K0[idx], uvd_map_0[idx], axes=1)

    return vertex_map_0

class PoseRefine():

    def __init__(self):

        self.f2p = pyFlow2Pose()

    def run(self, f_flow, b_flow, V0, V1, bg0, bg1, 
        pose_init=None, max_depth=None):
        """
        :param f_flow: forward flow vector
        :param b_flow: backward flow vector
        :param V0: vertex map for frame0
        :param V1: vertex map for frame1
        :param bg0: background for frame0
        :param bg1: background for frame1
        :param pose_init: initial pose
        """
        # the threshold is hard-coded
        m = forward_backward_consistency(f_flow, b_flow, 0.75)

        flow = f_flow.transpose((1,2,0))

        bg0 = dilation(bg0, square(5))
        bg1 = dilation(bg1, square(5))
        occlusion = erosion(m.astype(int), square(10))

        if max_depth is not None:
            invalid = (V0[:,:,2] < max_depth) & (V0[:,:,2] > 1e-4)
            occlusion *= invalid.astype(int)

        if pose_init is None:
            pose_init = np.zeros((3,4)).astype(np.float32)
            pose_init[:, :3] = np.eye(3)

        ## Becareful about the type of input:
        # V0: double
        # V1: double
        # flow: float32
        # bg0: int
        # bg1: int
        # occlusion: int
        # pose_init: float32
        pose_refined = self.f2p.calculate_transform(
            V0.astype(np.float64), V1.astype(np.float64), flow.astype(np.float32), 
            bg0.astype(int), bg1.astype(int), occlusion.astype(int), 
            pose_init.astype(np.float32))

        return pose_refined

    def run_batch(self, vertices0, vertices1, rigidity0, rigidity1, 
        forward_flow, backward_flow, max_depth=None):
        """ Run the pose refine algorithm in batches
        :param the first image point cloud (x0, y0, z0)
        :param the second image point cloud (x1, y1, z1)
        :param the first frame rigidity mask
        :param the second frame rigidity mask
        :param forward optical flow
        :param backward optical flow
        :param maximum depth range
        """
        # V0_batch = depth2pointcloud_batch(K0, D0)
        # V1_batch = depth2pointcloud_batch(K1, D1)

        B = vertices0.shape[0]
        est_Rt44 = np.zeros((B, 4, 4))
        for idx in range(B):
            V0 = vertices0[idx].transpose((1,2,0))
            V1 = vertices1[idx].transpose((1,2,0))

            est_Rt44[idx] = self.run(forward_flow[idx], 
                backward_flow[idx], V0, V1, 
                rigidity0[idx], rigidity1[idx], max_depth=None)

        return est_Rt44
