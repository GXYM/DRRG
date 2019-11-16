import copy
import cv2
import torch
import numpy as np
from PIL import Image

def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image

class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training

    def get_test_data(self, image, image_id, image_path):

        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        image = image.transpose(2, 0, 1)
        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()
