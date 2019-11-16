#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset


class TD500Text(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training

        self.image_root = os.path.join(data_root, 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root,'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)

        p = re.compile('.rar|.txt')
        self.image_list = [x for x in self.image_list if not p.findall(x)]
        p = re.compile('(.jpg|.JPG|.PNG|.JPEG)')
        self.annotation_list = ['{}'.format(p.sub("", img_name)) for img_name in self.image_list]

    def __getitem__(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        return self.get_test_data(image, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

