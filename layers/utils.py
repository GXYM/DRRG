#-*- coding:utf-8 -*-
#####
# Created on 19-9-20 上午10:05
#
# @Author: Greg Gao(laygin)
#####
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def normalize_adj(A, type="AD"):
    if type == "DAD":
        # d is  Degree of nodes A=A+I
        # L = D^-1/2 A D^-1/2
        A = A + np.eye(A.shape[0])  # A=A+I
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)
        G = torch.from_numpy(G)
    elif type == "AD":
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)
    else:
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        D = np.diag(D)
        G = D - A
    return G


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA = A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED


def bbox_transfor_inv(radius_map, sin_map, cos_map, score_map, wclip=(4, 12), expend=1.0):
    xy_text = np.argwhere(score_map > 0)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    origin = xy_text
    radius = radius_map[xy_text[:, 0], xy_text[:, 1], :]
    sin = sin_map[xy_text[:, 0], xy_text[:, 1]]
    cos = cos_map[xy_text[:, 0], xy_text[:, 1]]
    dtx = radius[:, 0] * cos * expend
    dty = radius[:, 0] * sin * expend
    ddx = radius[:, 1] * cos * expend
    ddy = radius[:, 1] * sin * expend
    topp = origin + np.stack([dty, dtx], axis=-1)
    botp = origin - np.stack([ddy, ddx], axis=-1)
    width = (radius[:, 0] + radius[:, 1]) // 3
    width = np.clip(width, wclip[0], wclip[1])

    top1 = topp - np.stack([width * cos, -width * sin], axis=-1)
    top2 = topp + np.stack([width * cos, -width * sin], axis=-1)
    bot1 = botp - np.stack([width * cos, -width * sin], axis=-1)
    bot2 = botp + np.stack([width * cos, -width * sin], axis=-1)

    bbox = np.stack([top1, top2, bot2, bot1], axis=1)[:, :, ::-1]
    bboxs = np.zeros((bbox.shape[0], 9), dtype=np.float32)
    bboxs[:, :8] = bbox.reshape((-1, 8))
    bboxs[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]

    return bboxs


def clip_box(bbox, im_shape):
    # x >= 0 and x<=w-1
    bbox[:, 0:8:2] = np.clip(bbox[:, 0:8:2], 1,  im_shape[1] - 2)
    # y >= 0 and  y <= h-1
    bbox[:, 1:8:2] = np.clip(bbox[:, 1:8:2], 1,  im_shape[0] - 2)

    return bbox


def filter_bbox(bbox, minsize=16):
    ws = np.sqrt((bbox[:, 0] - bbox[:, 2])**2 + (bbox[:, 1] - bbox[:, 3])**2)
    hs = np.sqrt((bbox[:, 0] - bbox[:, 6])**2 + (bbox[:, 1] - bbox[:, 7])**2)
    keep = np.where(ws*hs >= minsize)[0]
    bbox = bbox[keep]
    return bbox


def random_bbox(origin, limit=(8, 32)):
    rad0 = np.random.randint(limit[0], limit[1])
    rad1 = np.random.randint(limit[0], limit[1])
    cos = 2*np.random.random()-1
    sin = 2*np.random.random()-1

    dtx = rad0 * cos
    dty = rad0 * sin
    ddx = rad1 * cos
    ddy = rad1 * sin
    topp = origin + np.stack([dty, dtx], axis=-1)
    botp = origin - np.stack([ddy, ddx], axis=-1)
    width = (rad0 + rad1) // 3
    width = np.clip(width, 4, 12)

    top1 = topp - np.stack([width * cos, -width * sin], axis=-1)
    top2 = topp + np.stack([width * cos, -width * sin], axis=-1)
    bot1 = botp - np.stack([width * cos, -width * sin], axis=-1)
    bot2 = botp + np.stack([width * cos, -width * sin], axis=-1)

    bbox = np.stack([[top1], [top2], [bot2], [bot1]], axis=1)[:, :, ::-1]
    bboxs = np.zeros((bbox.shape[0], 9), dtype=np.float32)
    bboxs[:, :8] = bbox.reshape((-1, 8))
    bboxs[:, 8] = 0

    ctrp = np.mean(bboxs[0, :8].reshape((4, 2)), axis=0).astype(np.int32)

    return np.array([ctrp[0], ctrp[1], rad0+rad1, 2*width, cos, sin, 0])


def jitter_gt_boxes(gt_boxes, jitter=0.25):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    """
    jit_boxs = gt_boxes.copy()

    ws = np.sqrt((jit_boxs[:, 0] - jit_boxs[:, 2])**2
                 + (jit_boxs[:, 1] - jit_boxs[:, 2])**2) + 1.0
    hs = np.sqrt((jit_boxs[:, 0] - jit_boxs[:, 6])**2
                 + (jit_boxs[:, 1] - jit_boxs[:, 7])**2) + 1.0
    ws = np.clip(ws, 10, 30)
    hs = np.clip(hs, 10, 120)
    width_offset = (np.random.rand(jit_boxs.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jit_boxs.shape[0]) - 0.5) * jitter * hs
    width_offset = np.repeat(np.expand_dims(width_offset, axis=1), 4, axis=1)
    height_offset = np.repeat(np.expand_dims(height_offset, axis=1), 4, axis=1)
    jit_boxs[:, 0:8:2] += width_offset
    jit_boxs[:, 1:8:2] += height_offset

    return jit_boxs


def jitter_gt_map(gt_map, jitter=0.25):
    """
    jitter the gtboxes, before adding them into rois,
    to be more robust for cls and rgs
    gt_map: (G, 7) [xc ,yc ,h, w, cos, sin, class] int
    """

    hs = gt_map[:, 2]
    ws = gt_map[:, 3]
    dim = gt_map.shape[0]

    x_offset = (np.random.rand(dim) - 0.5) * 2
    y_offset = (np.random.rand(dim) - 0.5) * 2

    h_offset = (np.random.rand(dim) - 0.5) * jitter * hs
    w_offset = (np.random.rand(dim) - 0.5) * jitter * ws

    cos_offset = (np.random.rand(dim) - 0.5) * 0.2
    sin_offset = (np.random.rand(dim) - 0.5) * 0.2

    gt_map[:, 0] += x_offset
    gt_map[:, 1] += y_offset

    gt_map[:, 2] += h_offset
    gt_map[:, 3] += w_offset

    sin = gt_map[:, 4] + cos_offset
    cos = gt_map[:, 5] + sin_offset

    scale = np.sqrt(1.0 / (sin ** 2 + cos ** 2))

    gt_map[:, 4] = scale * sin
    gt_map[:, 5] = scale * cos

    return gt_map

