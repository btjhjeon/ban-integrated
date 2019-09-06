"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.ban import build_model
from vqa_dataset import VQAFeatureDataset
from models.ban_vqa.dataset import Dictionary
import models.ban_vqa.train as train
import models.ban_vqa.utils as ban_utils
import models.bottom_up_features.utils as buatt_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ban_input', type=str, default='saved_models/ban/model_epoch12_cpu.pth')
    parser.add_argument('--buatt_cfg', type=str, help='config file',
                        default='models/bottom_up_features/cfgs/faster_rcnn_resnet101.yml')
    parser.add_argument('--buatt_input', type=str, help='path to pretrained model',
                        default='saved_models/bottom-up/bottomup_pretrained_10_100.pth')
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
    for i, (im_path, q, a) in enumerate(dataloader):
        im, im_info = load_im_tensor(im_path)

        im = im.cuda()
        im_info = im_info.cuda()
        q = q.cuda()
        a = a.unsqueeze(0)

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


def load_im_tensor(im_path):
    im = cv2.imread(im_path)
    blobs, im_scales = buatt_utils.get_image_blob(im)

    img = np.array(blobs)
    img = torch.from_numpy(img).permute(0, 3, 1, 2)
    img_info = torch.tensor([[blobs.shape[1], blobs.shape[2], im_scales[0]]])
    return img, img_info


if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    dict_path = 'data/dictionary.pkl'
    dictionary = Dictionary.load_from_file(dict_path)
    eval_dset = VQAFeatureDataset('val', dictionary)
    # eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)

    n_device = torch.cuda.device_count()
    batch_size = 1

    model = build_model(args, eval_dset)
    model = nn.DataParallel(model).cuda()
    model.eval()

    eval_score, bound, entropy = evaluate(model, eval_dset)
    print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
