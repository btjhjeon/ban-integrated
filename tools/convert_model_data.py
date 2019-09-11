import sys
import argparse

import torch
import torch.nn as nn

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('models/ban_vqa')

from vqa_dataset import build_dataset
import models.ban_vqa.base_model as base_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='saved_models/ban/model_epoch12.pth')
    parser.add_argument('--output', type=str, default='saved_models/ban/model_epoch12_module.pth')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()

    eval_dset = build_dataset('val')
    ban = base_model.build_ban(eval_dset, 768, '', 6, 'vqa')
    ban = nn.DataParallel(ban).cuda()

    ban.load_state_dict(torch.load(args.input)['model_state'])
    torch.save(ban.module.state_dict(), args.output)
