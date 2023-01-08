import torch.utils.data as data
import os
from glob import glob
import torch
from torchvision import transforms
import random
import numpy as np
import cv2
import json


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w

def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    # fast create discrete map
    # points_np = np.array(points).round().astype(int)
    # p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    # p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    # p_index = torch.from_numpy(p_h* im_width + p_w)
    # discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).long().numpy()

    ''' 
    slow method
    '''
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size=256,
                 downsample_ratio=4,
                 method='train'):
        print("init_Crowd dataset")
        self.root_path = root_path
        self.gt_list = sorted(glob(os.path.join(self.root_path, '*.npy')))  # change to npy for gt_list
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        self.RGB_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.407, 0.389, 0.396],
                std=[0.241, 0.246, 0.242]),
        ])
        self.T_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.492, 0.168, 0.430],
                std=[0.317, 0.174, 0.191]),
        ])

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, item):
        gt_path = self.gt_list[item]
        rgb_path = gt_path.replace('GT', 'RGB').replace('npy', 'jpg')
        t_path = gt_path.replace('GT', 'T').replace('npy', 'jpg')

        RGB = cv2.imread(rgb_path)[..., ::-1].copy()
        T = cv2.imread(t_path)[..., ::-1].copy()

        if self.method == 'train':
            keypoints = np.load(gt_path)
            return self.train_transform(RGB, T, keypoints)

        elif self.method == 'val' or self.method == 'test':
            keypoints = np.load(gt_path)
            gt = keypoints
            k = np.zeros((T.shape[0], T.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < T.shape[0] and int(gt[i][0]) < T.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            target = k
            RGB = self.RGB_transform(RGB)
            T = self.T_transform(T)
            width, height = RGB.shape[2], RGB.shape[1]
            m = int(width / 224)
            n = int(height / 224)
            for i in range(0, m):
                for j in range(0, n):
                    if i == 0 and j == 0:
                        img_return = RGB[:, j * 224: 224 * (j + 1), i * 224:(i + 1) * 224].cuda().unsqueeze(0)
                        t_return = T[:, j * 224: 224 * (j + 1), i * 224:(i + 1) * 224].cuda().unsqueeze(0)
                    else:
                        crop_img = RGB[:, j * 224: 224 * (j + 1), i * 224:(i + 1) * 224].cuda().unsqueeze(0)
                        crop_t = T[:, j * 224: 224 * (j + 1), i * 224:(i + 1) * 224].cuda().unsqueeze(0)
                        img_return = torch.cat([img_return, crop_img], 0).cuda()
                        t_return = torch.cat([t_return, crop_t], 0).cuda()

            name = os.path.basename(gt_path).split('.')[0]
            input = [img_return, t_return]
            return input, target, name

        else:
            raise Exception("Not implement")

    def train_transform(self, RGB, T, keypoints):
        ht, wd, _ = RGB.shape
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        RGB = RGB[i:i+h, j:j+w, :]
        T = T[i:i+h, j:j+w, :]
        keypoints = keypoints - [j, i]
        idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                   (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
        keypoints = keypoints[idx_mask]

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        gt_discrete = np.expand_dims(gt_discrete, 0)

        RGB = self.RGB_transform(RGB)
        T = self.T_transform(T)
        input = [RGB, T]

        return input, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float(), st_size

