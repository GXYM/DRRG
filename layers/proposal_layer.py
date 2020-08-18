#-*- coding:utf-8 -*-
#####
# Created on 19-9-20 上午10:05
#
# @Author: GXYM
#####
import numpy as np
import cv2
import random
from layers.utils import bbox_transfor_inv,\
    clip_box, filter_bbox, random_bbox, jitter_gt_map
from nmslib import lanms


class ProposalTarget(object):
    def __init__(self, k_at_hop1=15, clip=(4, 12), nms_threshold=0.25):
        self.nms_threshold = nms_threshold
        self.clip = clip
        self.k_at_hop1 = k_at_hop1

    def add_proposal(self, pk, sel_mask):
        if pk < self.k_at_hop1 + 1:
            h, w = sel_mask.shape
            b_num = self.k_at_hop1 + 1 - pk
            xy = np.argwhere(sel_mask[50:h - 50, 50:w - 50] > 0)
            xy = random.sample(xy.tolist(), b_num)
            gep = list()
            for idx, xy in enumerate(xy):
                g = random_bbox(np.array(xy))
                gep.append(g)
            gep = np.stack(gep, axis=0)

            return gep

        else:
            return None

    def __call__(self, tcl_mask, radius_map, sin_map, cos_map):

        imgsize = cos_map.shape
        # ## 1. Reverse generation of box
        proposals = bbox_transfor_inv(radius_map, sin_map, cos_map, tcl_mask[:, :, 1], wclip=self.clip)

        # ## 2. remove predicted boxes with either height or width < threshold
        proposals = filter_bbox(proposals, minsize=16)

        # ## 3. local nms
        proposals = lanms.merge_quadrangle_n9(proposals.astype('float32'), self.nms_threshold)

        # ## 4. clip bbox
        if proposals.shape[0] > 0:
            proposals = clip_box(proposals, imgsize)

        if proposals.shape[0] > 0:
            # ## 5. generate cluster label
            _, label_mask = cv2.connectedComponents(tcl_mask[:, :, 0].astype(np.uint8), connectivity=8)
            cxy = np.mean(proposals[:, :8].reshape((-1, 4, 2)), axis=1).astype(np.int32)

            # ## 6. Geometric features
            x_map = cxy[:, 0]
            y_map = cxy[:, 1]
            gh = (radius_map[:, :, 0] + radius_map[:, :, 1])
            h_map = gh[cxy[:, 1], cxy[:, 0]]
            w_map = np.clip(h_map // 4, self.clip[0], self.clip[1])*2
            c_map = cos_map[cxy[:, 1], cxy[:, 0]]
            s_map = sin_map[cxy[:, 1], cxy[:, 0]]
            label_map = label_mask[cxy[:, 1], cxy[:, 0]]
            geo_map = np.stack([x_map, y_map, h_map, w_map, c_map, s_map, label_map], axis=1)
        else:
            geo_map = None

        # ## 7. add bbox, sure proposals >self.k_at_hop[0]
        gps = self.add_proposal(proposals.shape[0], 1 - tcl_mask[:, :, 0])
        if gps is not None:
            if geo_map is not None:
                geo_map = np.concatenate([geo_map, gps], axis=0)
            else:
                geo_map = gps

        # ## 8. adding Random Disturbance
        geo_map = jitter_gt_map(geo_map, jitter=0.20)

        # ## 9. [roi_num, (xc ,yc ,h, w, cos, sin, class), img_size]-->Bx9
        roi_num = np.ones((geo_map.shape[0], 1), dtype=np.float16)*geo_map.shape[0]
        img_size = np.ones((geo_map.shape[0], 1), dtype=np.float16) * imgsize[0]
        gt_roi = np.hstack([roi_num, geo_map, img_size])

        return gt_roi

