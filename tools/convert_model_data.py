import os
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

from models.ban import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ban_input', type=str, default='saved_models/ban/model_epoch12.pth')
    parser.add_argument('--buatt_cfg', type=str, help='config file',
                        default='models/bottom_up_features/cfgs/faster_rcnn_resnet101.yml')
    parser.add_argument('--buatt_input', type=str, help='path to pretrained model',
                        default='saved_models/bottom-up/bottomup_pretrained_10_100.pth')
    parser.add_argument('--output', type=str, default='saved_models/ban_compact/model_epoch12.pth')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    eval_dset = build_dataset('val')
    ban = base_model.build_ban(eval_dset, 768, '', 6, 'vqa')
    ban = nn.DataParallel(ban).cuda()

    ban.load_state_dict(torch.load(args.ban_input)['model_state'])
    ban_data = ban.module.state_dict()

    model = build_model(args.buatt_cfg, eval_dset)
    model = nn.DataParallel(model).cuda()

    model.module.load_submodels(core_data=ban_data, detector_path=args.buatt_input)
    torch.save(model.state_dict(), args.output)
