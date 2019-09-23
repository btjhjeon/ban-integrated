import os
import numpy as np

import torch
import torch.nn as nn

import models.ban as ban
import models.bottom_up_features.utils as buatt_utils
from vqa_dataset import build_dataset


def build_model(dataroot='data/', model_path='saved_models/ban/model_epoch12.pth'):
    dict_path = os.path.join(dataroot, 'dictionary.pkl')
    dataset = build_dataset('val', dataroot, dict_path)
    dictionary = dataset.dictionary
    dictionary.label2ans = dataset.label2ans

    this_dir = os.path.dirname(__file__)
    buatt_cfg = os.path.join(this_dir, 'models/bottom_up_features/cfgs/faster_rcnn_resnet101.yml')
    model = ban.build_model(dataset, buatt_cfg)

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, dictionary


@torch.no_grad()
def ask(model, dictionary, image, question):

    blobs, im_scales = buatt_utils.get_image_blob(image)
    img = np.array(blobs)
    img = torch.from_numpy(img).permute(0, 3, 1, 2).cuda()
    img_info = torch.tensor([[blobs.shape[1], blobs.shape[2], im_scales[0]]]).cuda()

    max_length = 14
    q_token = dictionary.tokenize(question, False)
    q_token = q_token[:max_length]
    if len(q_token) < max_length:
        # Note here we pad in front of the sentence
        padding = [dictionary.padding_idx] * (max_length - len(q_token))
        q_token = q_token + padding
    assert len(q_token) == max_length
    q = torch.from_numpy(np.array(q_token)).unsqueeze(0).cuda()

    pred, att = model(img, img_info, q)

    return dictionary.label2ans[pred[0].argmax(0).item()]


if __name__ == '__main__':
    import cv2
    image = cv2.imread('sample.jpg')    # image_id: 393225
    question = 'what is to the right of the soup?'  # answer: chopsticks or spoon

    model, dictionary = build_model(dataroot='data/', model_path='saved_models/ban-integrated/model_epoch12.pth')

    answer = ask(model, dictionary, image, question)

    print('Q: {}, A: {}'.format(question, answer))
