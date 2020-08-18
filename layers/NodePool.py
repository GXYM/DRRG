import torch
import torch.nn as nn
from torch.autograd import Variable, variable
import numpy as np
import time
import cv2


class NodeRoIPool(nn.Module):
    def __init__(self, spatial_scale, pool="p1"):
        super(NodeRoIPool, self).__init__()
        self.spatial_scale = float(spatial_scale)
        self.pooling = pool

    def Bilinear(self, center, data):
        if all(np.floor(center) == np.ceil(center)):
            yc, xc = np.round(center)
            result = data[:, xc, yc]
        else:
            yc, xc = center
            y1, x1 = np.floor(center).astype(int)
            y2, x2 = np.ceil(center).astype(int)
            result = data[:, x1, y1] * (x2 - xc) * (y2 - yc) \
                     + data[:, x2, y1] * (xc - x1) * (y2 - yc) \
                     + data[:, x1, y2] * (x2 - xc) * (yc - y1) \
                     + data[:, x2, y2] * (xc - x1) * (yc - y1)

        return result

    def maskconv(self, p, limit, size=4):
        stride = size // 2

        x, y = np.ceil(p)
        x = np.clip(x, stride, limit[1] - stride)
        y = np.clip(y, stride, limit[0] - stride)
        shifts_x = torch.arange(x - stride, x + stride, step=1, dtype=torch.int)
        shifts_y = torch.arange(y - stride, y + stride, step=1, dtype=torch.int)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y), dim=1).type(torch.long)

        return shifts

    def pooling1(self, roi, feat):
        (fh, fw) = feat.size(1), feat.size(2)
        fs = list()
        for p in roi:
            index = self.maskconv(p, (fh, fw))
            f = torch.mean(feat[:,  index[:, 1], index[:, 0]], dim=1)
            fs.append(f)

        return torch.cat(fs, dim=0)
    
    def pooling2(self, roi, feat):
        box = np.array(roi * self.spatial_scale).reshape((4, 2))
        p1 = self.Bilinear([box[0][0], box[0][1]], feat).view(-1)
        p2 = self.Bilinear([box[1][0], box[1][1]], feat).view(-1)
        p3 = self.Bilinear([box[2][0], box[2][1]], feat).view(-1)
        p4 = self.Bilinear([box[3][0], box[3][1]], feat).view(-1)
        c0 = self.Bilinear(np.mean(box, axis=0), feat)

        return torch.cat([p1, p2, p3, p4, c0], dim=0)

    def forward(self, feat, rois):
        num_channels, data_height, data_width = feat.size()
        rrois = (rois * self.spatial_scale).reshape((-1, 4, 2))
        roib = np.zeros((rrois.shape[0], 5, 2))
        roib[:, 0:3] = rrois[:, 1:4]
        roib[:, 3] = rrois[:, 0]
        roib[:, 0:4, :] = (rrois + roib[:, 0:4, :]) / 2
        roib[:, 4, :] = np.mean(rrois, axis=1)

        outputs = Variable(torch.zeros(rois.shape[0], num_channels*5)).cuda()

        for roi_ind, roi in enumerate(roib):
            if self.pooling == "p1":
                outputs[roi_ind, :] = self.pooling1(roi, feat)
            elif self.pooling == "p2":
                outputs[roi_ind, :] = self.pooling2(roi, feat)
            else:
                print("pooling type is not support")

        return outputs

