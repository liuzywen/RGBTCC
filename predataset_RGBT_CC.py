import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import random

# 先224*224
'''set your data path'''
root = 'F:/DataSets/'

rgbt_cc_train = os.path.join(root, 'RGBT-CC-use/train_depth')
rgbt_cc_test = os.path.join(root, 'RGBT-CC-use/test_depth')
rgbt_cc_val = os.path.join(root, 'RGBT-CC-use/val_depth')

# 记得与之修改后面的路径
# path_sets = [rgbt_cc_train]
path_sets = [rgbt_cc_test]
# path_sets = [rgbt_cc_val]
'''for part A'''
# if not os.path.exists(rgbt_cc_train.replace('train_depth', 'new_train_depth_384')):
    # os.makedirs(rgbt_cc_train.replace('train_depth', 'new_trian_depth_384'))

if not os.path.exists(rgbt_cc_test.replace('test_depth', 'new_test_depth_384')):
    os.makedirs(rgbt_cc_test.replace('test_depth', 'new_test_depth_384'))
# 
# if not os.path.exists(rgbt_cc_val.replace('val_depth', 'new_val_depth_384')):
    # os.makedirs(rgbt_cc_val.replace('val_depth', 'new_val_depth_384'))

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*RGB.png')):
        img_paths.append(img_path)

img_paths.sort()

np.random.seed(0)
random.seed(0)
for img_path in img_paths:
    # print(img_path)
    Img_data = cv2.imread(img_path)

    # T_data = cv2.imread(img_path.replace('_RGB', '_T'))

    # Gt_data = np.load(img_path.replace('_RGB.jpg', '_GT.npy'))
    # 448和672
    rate = 1
    rate_1 = 1
    rate_2 = 1
    flag = 0
    if Img_data.shape[1] >= Img_data.shape[0]:  # 后面的大
        rate_1 = 1152.0 / Img_data.shape[1]
        rate_2 = 768.0 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
        # T_data = cv2.resize(T_data, (0, 0), fx=rate_1, fy=rate_2)
        # Gt_data[:, 0] = Gt_data[:, 0] * rate_1
        # Gt_data[:, 1] = Gt_data[:, 1] * rate_2
        print("1111111")

    elif Img_data.shape[0] > Img_data.shape[1]:  # 前面的大
        rate_1 = 1152.0 / Img_data.shape[0]
        rate_2 = 768.0 / Img_data.shape[1]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
        # T_data = cv2.resize(T_data, (0, 0), fx=rate_2, fy=rate_1)
        # Gt_data[:, 0] = Gt_data[:, 0] * rate_2 # 对应的坐标进行扩大映射
        # Gt_data[:, 1] = Gt_data[:, 1] * rate_1 # 对应的坐标进行扩大映射
        print("22222")

    # kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    #
    # for i in range(0, len(Gt_data)):
    #     if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
    #         kpoint[int(Gt_data[i][1]), int(Gt_data[i][0])] = 1

    # height, width = Img_data.shape[0], Img_data.shape[1]

    # m = int(width / 224)
    # n = int(height / 224)
    # fname = img_path.split('/')[-1]
    # root_path = img_path.split('IMG_')[0].replace('images', 'images_crop')

    # kpoint = kpoint.copy()
    # if root_path.split('/')[-3] == 'train_data':
    #
    #     for i in range(0, m):
    #         for j in range(0, n):
    #             crop_img = Img_data[j * 224: 224 * (j + 1), i * 224:(i + 1) * 224, ]
    #             crop_kpoint = kpoint[j * 224: 224 * (j + 1), i * 224:(i + 1) * 224]
    #             gt_count = np.sum(crop_kpoint)
    #
    #             save_fname = str(i) + str(j) + str('_') + fname
    #             save_path = root_path + save_fname

                # h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
                # if gt_count == 0:
                #     print(save_path, h5_path)
                # with h5py.File(h5_path, 'w') as hf:
                #     hf['gt_count'] = gt_count

                # cv2.imwrite(save_path, crop_img)
    # else: # 训练才需要位置，验证，测试不需要
    img_path = img_path.replace('test_depth', 'new_test_depth_384')
    print(img_path)
    # T_path = img_path.replace('_RGB','_T')
    # gt_save_path = img_path.replace('_RGB.jpg', '_GT.npy')
    cv2.imwrite(img_path, Img_data)
    # cv2.imwrite(T_path, T_data)
    # np.save(gt_save_path, Gt_data)
    # gt_count = np.sum(kpoint)
    # with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
    #     hf['gt_count'] = gt_count
