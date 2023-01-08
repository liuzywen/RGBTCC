# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: generate_point.py
@time: 2022/2/24 20:20
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from glob import glob
import os


def generate_data(label_path):
    with open(label_path, 'r') as f:
        label_file = json.load(f)
    points = np.asarray(label_file['points'])
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= 640) * (points[:, 1] >= 0) * (points[:, 1] <= 480)
    points = points[idx_mask]
    return points


if __name__ == '__main__':
    sub_dir = ""
    sub_save_dir = ""
    gt_list = glob(os.path.join(sub_dir, '*json'))
    for gt_path in gt_list:
        print(gt_path)
        name = os.path.basename(gt_path)
        points = generate_data(gt_path)
        im_save_path = os.path.join(sub_save_dir, name)
        gd_save_path = im_save_path.replace('json', 'npy')
        np.save(gd_save_path, points)