""" A simple data loader
"""

import os, io_utils
import torch.utils.data as data
import os.path as osp
import numpy as np

from scipy.misc import imread

class SimpleLoader(data.Dataset):

    def __init__(self, color_dir, depth_dir):
        """
        :param the directory of color images
        :param the directory of depth images
        """

        color_files = sorted(os.listdir(color_dir))
        depth_files = sorted(os.listdir(depth_dir))

        # please ensure the two folders use the same number of synchronized files
        assert(len(color_files) == len(depth_files))

        self.color_pairs = []
        self.depth_pairs = []
        self.ids = len(color_files) - 1
        for idx in range(self.ids):        
            self.color_pairs.append([
                osp.join(color_dir, color_files[idx]), 
                osp.join(color_dir, color_files[idx+1])
                ] )
            self.depth_pairs.append([
                osp.join(depth_dir, depth_files[idx]), 
                osp.join(depth_dir, depth_files[idx+1])
                ] )

    def __getitem__(self, index):

        image0_path, image1_path = self.color_pairs[index]
        depth0_path, depth1_path = self.depth_pairs[index]

        image0 = self.__load_rgb_tensor(image0_path)
        image1 = self.__load_rgb_tensor(image1_path)

        depth0 = self.__load_depth_tensor(depth0_path)
        depth1 = self.__load_depth_tensor(depth1_path)

        return image0, image1, depth0, depth1

    def __len__(self):
        return self.ids

    def __load_rgb_tensor(self, path):
        image = imread(path)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2,0,1))
        return image

    def __load_depth_tensor(self, path):
        if path.endswith('.dpt'):
            depth = io_utils.depth_read(path)
        elif path.endswith('.png'):
            depth = imread(path) / 1000.0
        else: 
            raise NotImplementedError
        return depth[np.newaxis, :]