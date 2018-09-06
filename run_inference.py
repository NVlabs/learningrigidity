# Run the inference

import os, sys, time, pickle
import geometry, io_utils
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave

from pose_refine import PoseRefine
from math import ceil
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

from models.PWCNet import pwc_dc_net
from models.RigidityNet import rigidity_transform_net
from SimpleLoader import SimpleLoader

def check_directory(filename):
    target_dir = os.path.dirname(filename)
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

def check_cuda(data):
    if torch.cuda.is_available(): return data.cuda()
    else: return data

def color_normalize(color):
    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=rgb_mean, std=rgb_std)
    
    B = color.shape[0] 
    for idx in range(B):
        color[idx] = normalize(color[idx])
    return color

def batch_resize_input(img0, img1):
    """ Resize the input for flow-network
    """
    B, C, H, W = img0.shape
    resize_H = int(ceil(H / 64.0)) * 64
    resize_W = int(ceil(W / 64.0)) * 64
    if H != resize_H or W != resize_W:
        resize_img = nn.Upsample(size=(resize_H, resize_W),mode='bilinear')
        img0 = resize_img(img0)
        img1 = resize_img(img1)

    return img0, img1

def batch_resize_output(flow, target_size):
    """ Resize the output of flow image to match the given resolution
    """
    H_in, W_in = flow.shape[-2:]
    H, W = target_size

    if H_in != H or W_in != W:
        resize_flow = nn.Upsample(size=(H, W),mode='bilinear')
        scale_H, scale_W = float(H) / H_in, float(W) / W_in
        output = resize_flow(flow)
        output_flow = output.clone()

        output_flow[:,0,:,:] = output[:,0,:,:] * scale_W
        output_flow[:,1,:,:] = output[:,1,:,:] * scale_H

        return output_flow
    else:
        return flow

def rigidity_net_forward(net, uvd0, uvd1):
    """ Run an forward estimation of the rigidity network
    """
    input0 = uvd0.clone()
    input1 = uvd1.clone()

    # clamp the inverse depth
    input0[:,2,:,:] = torch.clamp(input0[:,2,:,:], min=1e-4, max=10)
    input1[:,2,:,:] = torch.clamp(input1[:,2,:,:], min=1e-4, max=10)

    data = torch.cat((input0, input1), 1)
    data = check_cuda(data)
    data = Variable(data, volatile=True)

    pose_feat_map, seg_feat_map = net(data)

    est_pose = nn.AdaptiveAvgPool2d((1,1))(pose_feat_map)
    B, C, H, W = data.size() 
    est_pose = est_pose.view([B, -1])

    R = geometry.torch_euler2mat(est_pose[:, 3], est_pose[:, 4], est_pose[:, 5])
    t = est_pose[:, :3]

    seg_feat_map = nn.functional.upsample(seg_feat_map, (H,W), mode='bilinear')
    seg_map = nn.functional.softmax(seg_feat_map, dim=1)
    _, rigidity = torch.max(seg_map, 1)

    return rigidity, [R, t]

def visualize_flow(flow_tensor, output_path, file_idx, matplot_viz=False):
    """ Visualize the pytorch flow results using matplotlib 
    only visualize the first batch (assume batch-size is 1)
    """
    np_flow = flow_tensor[0].cpu().numpy()
    flow_viz = io_utils.flow_visualize(np_flow)

    filename = "{:}/optical_flow_{:04d}.png".format(output_path, file_idx)
    check_directory(filename)
    imsave(filename, flow_viz)

    if matplot_viz:
        plt.title("optical flow visualization")
        plt.imshow(flow_viz)
        plt.show()

def visualize_rigidity(rigidity_tensor, color_tensor, 
    output_path, file_idx, matplot_viz=False):
    """ Visualize the pytorch rigidity results using matplotlib
    only visualize the first batch (assume batch-size is 1)
    """
    image = color_tensor[0].cpu().numpy().transpose((1,2,0))
    rigidity = rigidity_tensor[0].cpu().numpy()

    rigidity_viz = io_utils.image_with_mask(image, rigidity)

    filename = "{:}/rigidity_{:04d}.png".format(output_path, file_idx)
    check_directory(filename)
    imsave(filename, rigidity_viz)

    if matplot_viz:
        plt.title("rigidity visualization")
        plt.imshow(rigidity_viz)
        plt.show()

def visualize_projected_flow(depth_tensor, flow_tensor, K, T, 
    output_path, file_idx, matplot_viz=False): 
    """ Visualize the ego-motion flow and projected scene flow using matplotlib
    only visualize the first batch (assume batch-size is 1)
    """
    origin = np.eye(4)
    ego_flow = geometry.np_depth2flow(depth_tensor.cpu().numpy()[0,0], K, origin, K, T[0])[0]
    ego_flow_viz = io_utils.flow_visualize(ego_flow)

    proj_scene_flow = flow_tensor[0].cpu().numpy() - ego_flow
    proj_scene_flow_viz = io_utils.flow_visualize(proj_scene_flow)

    filename_ego = "{:}/ego_motion_flow_{:04d}.png".format(output_path, file_idx)
    filename_proj= "{:}/proj_scene_flow_{:04d}.png".format(output_path, file_idx)
    check_directory(filename_ego)
    check_directory(filename_proj)
    imsave(filename_ego, ego_flow_viz)
    imsave(filename_proj, proj_scene_flow_viz)

    if matplot_viz:
        f, ax = plt.subplots(2)
        ax[0].set_title("ego-motion flow visualization")
        ax[0].imshow(ego_flow_viz)
        ax[1].set_title("projected scene flow visualization")
        ax[1].imshow(proj_scene_flow_viz)

        plt.subplots_adjust(hspace=0.5)
        plt.show()
    
def run_inference(simple_loader, pwc_net, rigidity_net, 
    post_refine=True, visualize_output=True, output_path='results'):

    pwc_net     = check_cuda(pwc_net)
    rigidity_net= check_cuda(rigidity_net)

    pwc_net.eval()
    rigidity_net.eval()

    if post_refine: 
        pose_refine = PoseRefine()

    for batch_idx, batch in enumerate(simple_loader):
        color0, color1, depth0, depth1 = [check_cuda(x) for x in batch]

        # run the rigidity network to estimate the forward backward rigidity
        uvd0 = geometry.torch_rgbd2uvd(color_normalize(color0.clone()), depth0, K[0], K[1], K[2], K[3])
        uvd1 = geometry.torch_rgbd2uvd(color_normalize(color1.clone()), depth1, K[0], K[1], K[2], K[3])

        rigidity_backward,Rt_backward= rigidity_net_forward(rigidity_net, uvd1, uvd0)
        rigidity_forward, Rt_forward = rigidity_net_forward(rigidity_net, uvd0, uvd1)

        visualize_rigidity(rigidity_forward.data, color0, output_path, batch_idx, matplot_viz=visualize_output)
        # visualize_rigidity(rigidity_backward.data, color1,output_path, batch_idx, matplot_viz=visualize_output)

        # run the pwc-net to estimate the forward backward flow 
        B, _, H, W = color0.shape
        img0, img1 = batch_resize_input(color0, color1)
        forward_flow = pwc_net(img0, img1) * 20
        backward_flow= pwc_net(img1, img0) * 20
        forward_flow = batch_resize_output(forward_flow, target_size=(H, W))
        backward_flow= batch_resize_output(backward_flow,target_size=(H, W))

        # visualize the output, finally move it outside of the loop
        visualize_flow(forward_flow.data, output_path, batch_idx, matplot_viz=visualize_output)
        # visualize_flow(backward_flow.data,output_path, batch_idx, matplot_viz=visualize_output)

        # run the inference refinement 
        if post_refine: 
            xyz0 = geometry.torch_depth2xyz(depth0, K[0], K[1], K[2], K[3])
            xyz1 = geometry.torch_depth2xyz(depth1, K[0], K[1], K[2], K[3])
            # if use indoor data, e.g. TUM, set the maximum depth to 7.5
            est_Rt44 = pose_refine.run_batch(
                xyz0.cpu().numpy(), xyz1.cpu().numpy(), 
                rigidity_forward.data.cpu().numpy(), 
                rigidity_backward.data.cpu().numpy(), 
                forward_flow.data.cpu().numpy(), 
                backward_flow.data.cpu().numpy(), 
                max_depth=None)

            print("Estimated two-view transform")
            print(est_Rt44)

            visualize_projected_flow(depth0, forward_flow.data, K, est_Rt44, output_path, batch_idx, matplot_viz=visualize_output)

if __name__ == '__main__': 

    import argparse

    parser = argparse.ArgumentParser(description='Run the network inference')

    parser.add_argument('--pwc_weights', default='weights/pwc_net_chairs.pth.tar', 
        type=str, help='the path to pytorch weights of PWC Net')
    parser.add_argument('--rigidity_weights', default='weights/rigidity_net.pth.tar', 
        type=str, help='the path to pytorch weights of Rigidity Transform Net')
    parser.add_argument('--visualize', default=False, action='store_true', 
        help='plot the output')
    parser.add_argument('--post_refine', default=False, action='store_true', 
        help='run the refinement process')
    parser.add_argument('--color_dir', default='data/market_5/clean',
        help='the directory of color images')
    parser.add_argument('--depth_dir', default='data/market_5/depth', 
        help='the directory of depth images (default SINTEl dpt)')
    parser.add_argument('--intrinsic', default='1120,1120,511.5,217.5', 
        help='Simple pin-hole camera intrinsics, input in the format (fx, fy, cx, cy)')
    parser.add_argument('--output_path', default='results',
        help='The path to save all the outputs.')

    config = parser.parse_args()

    pwc_net = pwc_dc_net(config.pwc_weights)
    rigidity_net = rigidity_transform_net(config.rigidity_weights)

    color_dir = config.color_dir
    depth_dir = config.depth_dir
    K = [float(x) for x in config.intrinsic.split(',')]

    simple_loader = SimpleLoader(color_dir, depth_dir)    
    simple_loader = DataLoader(simple_loader, batch_size=1, shuffle=False)

    run_inference(simple_loader, pwc_net, rigidity_net, 
        post_refine = config.post_refine, 
        visualize_output = config.visualize, 
        output_path = config.output_path
    )


