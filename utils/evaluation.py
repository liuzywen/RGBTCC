import cv2
import math

def eval_game(output, target, L=0):
    B,_,_,_ = output.shape

    abs_error = 0
    square_error = 0
    for c in range(6):
        output1 = output[c][0].cpu().numpy()
        target1 = target[c][0]
        print(L, output1.sum()-target1.sum())
        H, W = target1.shape

        ratio = H / output1.shape[0]
        output1 = cv2.resize(output1, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio*ratio)
        assert output1.shape == target1.shape

        # eg: L=3, p=8 p^2=64
        p = pow(2, L)

        for i in range(p):
            for j in range(p):
                output_block = output1[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
                target_block = target1[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

                abs_error += abs(output_block.sum()-target_block.sum().float())
                square_error += (output_block.sum()-target_block.sum().float()).pow(2)
    # print(L, out_abs_error)
    return abs(abs_error), square_error


def eval_relative(output, target):
    output_num = output.cpu().data.sum()
    target_num = target.sum().float()
    relative_error = abs(output_num-target_num)/target_num
    return relative_error