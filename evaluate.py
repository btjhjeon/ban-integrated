"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.ban import build_model
from vqa_dataset import build_dataset
import models.ban_vqa.train as train
import models.ban_vqa.utils as ban_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='saved_models/ban-integrated/model_epoch12.pth')
    parser.add_argument('--buatt_cfg', type=str, help='config file',
                        default='models/bottom_up_features/cfgs/faster_rcnn_resnet101.yml')
    args = parser.parse_args()
    return args


@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = None
    if hasattr(model.module, 'glimpse'):
        entropy = torch.Tensor(model.module.glimpse).zero_().cuda()
    for i, (im, im_info, q, a) in enumerate(dataloader):
        im = im.cuda()
        im_info = im_info.cuda()
        q = q.cuda()

        pred, att = model(im, im_info, q)
        batch_score = train.compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)
        if att is not None and 0 < model.module.glimpse:
            entropy += train.calc_entropy(att.data)[:model.module.glimpse]

    score = score / len(dataloader)
    upper_bound = upper_bound / len(dataloader)

    if entropy is not None:
        entropy = entropy / len(dataloader)

    return score, upper_bound, entropy


if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    n_device = torch.cuda.device_count()
    batch_size = 1

    eval_dset = build_dataset('val')
    eval_loader = DataLoader(eval_dset, batch_size=batch_size)

    model = build_model(args, eval_dset)
    model = nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load(args.input))
    model.eval()

    eval_score, bound, entropy = evaluate(model, eval_loader)
    print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
