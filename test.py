import torch
import os
import argparse
from datasets.crowd import Crowd
from nets.RGBTCCNet import ThermalRGBNet
from utils.raw_evaluation import eval_game
import numpy as np
import math

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--data-dir', default='F:/DataSets/RGBT_CC',
                        help='training data directory')
parser.add_argument('--save-dir', default='./ckpts_PVTV2_r224/0727-104422',
                        help='model directory')
parser.add_argument('--model', default='best_model_10.762619034647942.pth'
                    , help='model name')
parser.add_argument('--img_size', default=224, type=int, help='network input size')
parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':

    datasets = Crowd(os.path.join(args.data_dir, "new_test_224"), method='test')
    dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                             num_workers=0, pin_memory=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = ThermalRGBNet()
    model.to(device)
    model_path = os.path.join(args.save_dir, args.model)
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0
    epoch_res = []
    for idx, (inputs, target, name) in enumerate(dataloader):
        print(idx)
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)
        if len(inputs[0].shape) == 5:
            inputs[0] = inputs[0].squeeze(0)
            inputs[1] = inputs[1].squeeze(0)
        if len(inputs[0].shape) == 3:
            inputs[0] = inputs[0].unsqueeze(0)
            inputs[1] = inputs[1].unsqueeze(0)
        with torch.set_grad_enabled(False):
            count, outputs, _ = model(inputs)  # outputs batch_size为4
            outputs1 = torch.cat((outputs[0], outputs[1]), dim=1)
            outputs2 = torch.cat((outputs[2], outputs[3]), dim=1)
            outputs3 = torch.cat((outputs[4], outputs[5]), dim=1)
            outputs = torch.cat((outputs1, outputs2, outputs3), dim=2)

            res = torch.sum(target).item() - torch.sum(outputs).item()
            epoch_res.append(res)

            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
    N = len(dataloader)
    epoch_res = np.array(epoch_res)
    mse1 = np.sqrt(np.mean(np.square(epoch_res)))
    mae1 = np.mean(np.abs(epoch_res))
    print(mae1)
    game = [m / N for m in game]
    mse = [math.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N
    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'MSE {mse:.2f}  '. \
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0])
    print(log_str)

