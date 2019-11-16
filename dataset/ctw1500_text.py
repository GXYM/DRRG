#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset
from util.io import read_lines
import cv2


class Ctw1500Text(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training

        self.image_root = os.path.join(data_root, 'train' if is_training else 'test', "text_image")
        self.annotation_root = os.path.join(data_root, 'train' if is_training else 'test', "text_label_circum")
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = ['{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)
        try:
            h, w, c = image.shape
            assert(c == 3)
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

        return self.get_test_data(image, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

