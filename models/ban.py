import os, sys
import numpy as np

import torch
import torch.nn as nn

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('models/ban_vqa')
import base_model

import models.bottom_up_features._init_paths
import models.bottom_up_features.utils as buatt_utils
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet


class IntegratedBAN(nn.Module):
    def __init__(self, detector, vqa_model):
        super(IntegratedBAN, self).__init__()

        self.detector = detector
        self.vqa_model = vqa_model
        self.glimpse = vqa_model.glimpse

    def forward(self, im, im_info, q):
        # dummy
        gt_boxes = torch.zeros(1, 1, 5).to(im.device)
        num_boxes = torch.zeros(1).to(im.device)

        rois, cls_prob, _, _, _, _, _, _, pooled_feat = self.detector(im, im_info, gt_boxes, num_boxes)

        boxes = rois.data[:, :, 1:5].squeeze()
        boxes /= im_info[0,2]
        cls_prob = cls_prob.squeeze()

        v, b = buatt_utils.threshold_results(cls_prob, pooled_feat, boxes, 0.2)

        return self.vqa_model(v.unsqueeze(0), b.unsqueeze(0), q.unsqueeze(0), None)


def build_model(args, dataset):

    ban = base_model.build_ban(dataset, 1280, 'c', 8, 'vqa')
    ban_data = torch.load(args.ban_input)
    ban.load_state_dict(ban_data)

    # Load arguments.
    N_CLASSES = 1601

    if args.buatt_cfg is not None:
        cfg_from_file(args.buatt_cfg)

    cfg.CUDA = True
    np.random.seed(cfg.RNG_SEED)

    # Load the model.
    fasterRCNN = resnet(N_CLASSES, 101, pretrained=False)
    fasterRCNN.create_architecture()
    fasterRCNN.load_state_dict(torch.load(args.buatt_input))

    model = IntegratedBAN(fasterRCNN, ban)
    return model
