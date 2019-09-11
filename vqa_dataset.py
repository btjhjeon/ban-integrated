"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import _pickle as cPickle
import numpy as np
import json
import cv2

import torch
from torch.utils.data import Dataset

import models.ban_vqa.dataset as ban_dataset
import models.ban_vqa.utils as ban_utils
import models.bottom_up_features.utils as buatt_utils


def build_dataset(mode):
    dict_path = 'data/dictionary.pkl'
    dictionary = ban_dataset.Dictionary.load_from_file(dict_path)
    return VQAFeatureDataset(mode, dictionary)


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.entries = self.load_dataset(dataroot, name, self.label2ans)
        self.tokenize()
        self.tensorize()
        self.v_dim = 2048

    def load_dataset(self, dataroot, name, label2ans):
        """Load entries

        img_id2val: dict {img_id -> val} val can be used to retrieve image or features
        dataroot: root path of dataset
        name: 'train', 'val', 'test-dev2015', test2015'
        """
        def _make_path(name, img_id):
            if name in ['train', 'val']:
                split = '{}2014'.format(name)
            else:
                split = '{}2015'.format(name)
            img_dir = os.path.join(dataroot, split)
            img_path = os.path.join(img_dir, 'COCO_{}_{:012d}.jpg'.format(split, img_id))
            return img_path

        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' % \
                      (name + '2014' if 'test' != name[:4] else name))
        questions = sorted(json.load(open(question_path))['questions'],
                           key=lambda x: x['question_id'])
        if 'test' != name[:4]:  # train, val
            answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
            answers = cPickle.load(open(answer_path, 'rb'))
            answers = sorted(answers, key=lambda x: x['question_id'])

            ban_utils.assert_eq(len(questions), len(answers))
            entries = []
            for question, answer in zip(questions, answers):
                ban_utils.assert_eq(question['question_id'], answer['question_id'])
                ban_utils.assert_eq(question['image_id'], answer['image_id'])
                img_id = question['image_id']
                img_path = _make_path(name, img_id)
                if not ban_dataset.COUNTING_ONLY or ban_dataset.is_howmany(question['question'], answer, label2ans):
                    entries.append(ban_dataset._create_entry(img_path, question, answer))
        else:  # test2015
            entries = []
            for question in questions:
                img_id = question['image_id']
                img_path = _make_path(name, img_id)
                if not ban_dataset.COUNTING_ONLY or ban_dataset.is_howmany(question['question'], None, None):
                    entries.append(ban_dataset._create_entry(img_path, question, None))

        return entries

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            ban_utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def load_im_tensor(self, im_path):
        im = cv2.imread(im_path)
        blobs, im_scales = buatt_utils.get_image_blob(im)

        img = np.array(blobs)
        img = torch.from_numpy(img).permute(0, 3, 1, 2).squeeze(0)
        img_info = torch.tensor([blobs.shape[1], blobs.shape[2], im_scales[0]])
        return img, img_info

    def __getitem__(self, index):
        entry = self.entries[index]

        img_path = entry['image']
        img, img_info = self.load_im_tensor(img_path)

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return img, img_info, question, target
        else:
            return img, img_info, question, question_id

    def __len__(self):
        return len(self.entries)