# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: raw_evaluation.py
@time: 2022/4/19 8:41
"""

import cv2


def eval_game(output, target, L=0):
    output = output[0].cpu().numpy()
    # output = output.squeeze(0)
    target = target[0]
    H, W = target.shape
    ratio = H / output.shape[0]
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio*ratio)
    # print(output.shape, target.shape)
    assert output.shape == target.shape

    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

            abs_error += abs(output_block.sum()-target_block.sum().float())
            square_error += (output_block.sum()-target_block.sum().float()).pow(2)

    return abs_error, square_error


def eval_relative(output, target):
    output_num = output.cpu().data.sum()
    target_num = target.sum().float()
    relative_error = abs(output_num-target_num)/target_num
    return relative_error