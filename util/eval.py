import os
import cv2
import numpy as np
import subprocess
from util.config import config as cfg
from util.misc import mkdirs


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def data_transfer_ICDAR(contours):
    cnts = list()
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        cnts.append(points)
    return cnts


def data_transfer_TD500(contours, res_file, img=None):
    with open(res_file, 'w') as f:
        for cont in contours:
            rect = cv2.minAreaRect(cont)
            points = cv2.boxPoints(rect)
            box = np.int0(points)

            cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
            # cv2.imshow("lllll", img)
            # cv2.waitKey(0)

            cx, cy = rect[0]
            w_, h_ = rect[1]
            angle = rect[2]
            mid_ = 0
            if angle > 45:
                angle = 90 - angle
                mid_ = w_;
                w_ = h_;
                h_ = mid_
            elif angle < -45:
                angle = 90 + angle
                mid_ = w_;
                w_ = h_;
                h_ = mid_
            angle = angle / 180 * 3.141592653589

            x_min = int(cx - w_ / 2)
            x_max = int(cx + w_ / 2)
            y_min = int(cy - h_ / 2)
            y_max = int(cy + h_ / 2)
            f.write('{},{},{},{},{}\r\n'.format(x_min, y_min, x_max, y_max, angle))

    return img


def data_transfer_MLT2017(contours, res_file):
    with open(res_file, 'w') as f:
        for cont in contours:
            rect = cv2.minAreaRect(cont)
            points = cv2.boxPoints(rect)
            points = np.int0(points)
            p = np.reshape(points, -1)
            f.write('{},{},{},{},{},{},{},{},{}\r\n'
                    .format(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 1))



