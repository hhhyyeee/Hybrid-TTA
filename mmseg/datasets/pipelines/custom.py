import torch

import cv2
from skimage import feature

from ..builder import PIPELINES


@PIPELINES.register_module()
class GetHOG(object):
    def __init__(self, resize_ratio=0.25):
        self.resize_ratio = resize_ratio
        pass

    def __call__(self, results):
        img = results['img']
        img = cv2.resize(img, (0, 0), fx=self.resize_ratio, fy=self.resize_ratio)

        _, hog_image = feature.hog(img, orientations=9, channel_axis=2,
                                   pixels_per_cell=(4, 4), cells_per_block=(4, 4),
                                   block_norm='L2-Hys', visualize=True, transform_sqrt=True)
        results['hog'] = hog_image

        return results


@PIPELINES.register_module()
class GetStatistics(object):
    def __init__(self, n_channels=3):
        self.n_channels = n_channels
        pass

    def __call__(self, results):
        mean = torch.zeros(self.n_channels)
        std = torch.zeros(self.n_channels)

        img = results['img']
        for i_c in range(self.n_channels):
            mean[i_c] = img[:,:,i_c].mean()
            std[i_c] = img[:,:,i_c].std()
        results['ood_statistics'] = dict(mean=mean, std=std)
        return results
