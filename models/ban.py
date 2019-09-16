import os, sys
import numpy as np

import torch
import torch.nn as nn

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path('models/ban_vqa')
import base_model
import utils

import models.bottom_up_features._init_paths
import models.bottom_up_features.utils as buatt_utils
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet


class IntegratedBAN(nn.Module):
    def __init__(self, detector, core):
        super(IntegratedBAN, self).__init__()

        self.detector = detector
        self.core = core
        self.glimpse = core.glimpse

    def forward(self, im, im_info, q):
        batch_size = im.size(0)

        # dummy
        gt_boxes = torch.zeros(batch_size, 1, 1, 5).to(im.device)
        num_boxes = torch.zeros(batch_size, 1).to(im.device)

        rois, cls_prob, _, _, _, _, _, _, pooled_feat = self.detector(im, im_info, gt_boxes, num_boxes)

        boxes = rois.data[:, :, 1:5]
        boxes /= im_info[:, 2]

        batch = []
        for i in range(batch_size):
            v, b = buatt_utils.threshold_results(cls_prob[i], pooled_feat[i], boxes[i], 0.2)
            batch.append([v, b])
        v, b = utils.trim_collate(batch)

        return self.core(v, b, q, None)

    def load_submodels(self, core_path=None, detector_path=None, core_data=None, detector_data=None):
        assert core_data is not None or core_path is not None
        assert detector_data is not None or detector_path is not None

        if core_data is None:
            core_data = torch.load(core_path)
        self.core.load_state_dict(core_data)

        if detector_data is None:
            detector_data = torch.load(detector_path)
        self.detector.load_state_dict(detector_data)


def build_model(args, dataset):

    ban = base_model.build_ban(dataset, 768, '', 6, 'vqa')

    # Load arguments.
    N_CLASSES = 1601

    if args.buatt_cfg is not None:
        cfg_from_file(args.buatt_cfg)

    cfg.CUDA = True
    np.random.seed(cfg.RNG_SEED)

    # Load the model.
    fasterRCNN = resnet(N_CLASSES, 101, pretrained=False)
    fasterRCNN.create_architecture()

    model = IntegratedBAN(fasterRCNN, ban)
    return model
