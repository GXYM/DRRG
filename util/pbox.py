#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'

import numpy as np
from typing import List


def functools_reduce(a):
    # 使用functools內建模块
    import functools
    import operator
    return functools.reduce(operator.concat, a)


def minConnectPath(list_all: List[list]):
    list_nodo = list_all.copy()
    res = []
    ept = [0, 0]

    def norm2(a, b):
        """计算两点之间的距离"""
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    dict00 = {}  # 格式   {距离,(起点坐标,终点坐标)}
    dict11 = {}  # 格式   {距离,(起点坐标,终点坐标)}
    # 放入一个初始值
    ept[0] = list_nodo[0]  # left end point
    ept[1] = list_nodo[0]  # right end point
    list_nodo.remove(list_nodo[0])
    while list_nodo:
        for i in list_nodo:  # i 待处理的
            length0 = norm2(i, ept[0])  # 端点0终点距离
            dict00[length0] = [i, ept[0]]
            length1 = norm2(ept[1], i)  # 端点0终点距离
            dict11[length1] = [ept[1], i]
        key0 = min(dict00.keys())
        key1 = min(dict11.keys())

        if key0 <= key1:
            ss = dict00[key0][0]
            ee = dict00[key0][1]
            res.insert(0, [list_all.index(ss), list_all.index(ee)])
            list_nodo.remove(ss)
            ept[0] = ss
        else:
            ss = dict11[key1][0]
            ee = dict11[key1][1]
            res.append([list_all.index(ss), list_all.index(ee)])
            list_nodo.remove(ee)
            ept[1] = ee

        dict00 = {}
        dict11 = {}

    path = functools_reduce(res)
    path = sorted(set(path), key=path.index)  # 去重

    return res, path


def bbox_transfor_inv(radius_map, sin_map, cos_map, score_map, wclip=(2, 8), expend=1.0):
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


