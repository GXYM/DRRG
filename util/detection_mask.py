import numpy as np
import cv2
from util.config import config as cfg
from util.misc import fill_hole, regularize_sin_cos
from util.misc import norm2
from nmslib import lanms
from util.pbox import minConnectPath


class TextDetector(object):

    def __init__(self, model):
        self.model = model
        self.tr_thresh = cfg.tr_thresh
        self.tcl_thresh = cfg.tcl_thresh
        self.expend = cfg.expend + 1.0

        # evaluation mode
        model.eval()


    @staticmethod
    def in_contour(cont, point):
        """
        utility function for judging whether `point` is in the `contour`
        :param cont: cv2.findCountour result
        :param point: 2d coordinate (x, y)
        :return:
        """
        x, y = point
        return cv2.pointPolygonTest(cont, (x, y), False) > 0

    def select_edge(self, cont, box):
        cont = np.array(cont)
        box = box.astype(np.int32)
        c1 = np.array(0.5 * (box[0, :] + box[3, :]), dtype=np.int)
        c2 = np.array(0.5 * (box[1, :] + box[2, :]), dtype=np.int)

        if not self.in_contour(cont, c1):
            return [box[0, :].tolist(), box[3, :].tolist()]
        elif not self.in_contour(cont, c2):
            return [box[1, :].tolist(), box[2, :].tolist()]
        else:
            return None

    def bbox_transfor_inv(self, radius_map, sin_map, cos_map, score_map, wclip=(2, 8)):
        xy_text = np.argwhere(score_map > 0)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        origin = xy_text
        radius = radius_map[xy_text[:, 0], xy_text[:, 1], :]
        sin = sin_map[xy_text[:, 0], xy_text[:, 1]]
        cos = cos_map[xy_text[:, 0], xy_text[:, 1]]
        dtx = radius[:, 0] * cos * self.expend
        dty = radius[:, 0] * sin * self.expend
        ddx = radius[:, 1] * cos * self.expend
        ddy = radius[:, 1] * sin * self.expend
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

    def detect_contours(self, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred):

        # thresholding
        tr_pred_mask = tr_pred > self.tr_thresh
        tcl_pred_mask = tcl_pred > self.tcl_thresh

        # multiply TR and TCL
        tcl_mask = tcl_pred_mask * tr_pred_mask

        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        # find disjoint regions
        tcl_mask = fill_hole(tcl_mask)
        tcl_contours, _ = cv2.findContours(tcl_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(tcl_mask)
        bbox_contours = list()
        for cont in tcl_contours:
            deal_map = mask.copy()
            cv2.drawContours(deal_map, [cont], -1, 1, -1)
            if deal_map.sum() <= 100:
                continue
            text_map = tr_pred * deal_map
            bboxs = self.bbox_transfor_inv(radii_pred, sin_pred, cos_pred, text_map, wclip=(4, 12))
            # nms
            boxes = lanms.merge_quadrangle_n9(bboxs.astype('float32'), 0.25)
            boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
            boundary_point = None
            if boxes.shape[0] > 1:
                center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                paths, routes_path = minConnectPath(center)
                boxes = boxes[routes_path]
                top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
                edge0 = self.select_edge(top + bot[::-1], boxes[0])
                edge1 = self.select_edge(top + bot[::-1], boxes[-1])
                if edge0 is not None:
                    top.insert(0, edge0[0])
                    bot.insert(0, edge0[1])
                if edge1 is not None:
                    top.append(edge1[0])
                    bot.append(edge1[1])
                boundary_point = np.array(top + bot[::-1])

            elif boxes.shape[0] == 1:
                top = boxes[0, 0:2, :].astype(np.int32).tolist()
                bot = boxes[0, 2:4:-1, :].astype(np.int32).tolist()
                boundary_point = np.array(top + bot)

            if boundary_point is None:
                continue
            reconstruct_mask = mask.copy()
            cv2.drawContours(reconstruct_mask, [boundary_point], -1, 1, -1)
            if (reconstruct_mask * tr_pred_mask).sum() < reconstruct_mask.sum() * 0.5:
                continue
            # if reconstruct_mask.sum() < 200:
            #     continue

            rect = cv2.minAreaRect(boundary_point)
            if min(rect[1][0], rect[1][1]) < 10 or rect[1][0] * rect[1][1] < 300:
                continue

            bbox_contours.append([boundary_point, np.array(np.stack([top, bot], axis=1))])

        return bbox_contours

    def detect(self, image, img_show):
        # get model output
        output = self.model.forward_test(image)
        image = image[0].data.cpu().numpy()
        tr_pred = output[0, 0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = output[0, 2:4].softmax(dim=0).data.cpu().numpy()
        sin_pred = output[0, 4].data.cpu().numpy()
        cos_pred = output[0, 5].data.cpu().numpy()
        radii_pred = output[0, 6:8].permute(1, 2, 0).contiguous().data.cpu().numpy()

        # find text contours
        contours = self.detect_contours(tr_pred[1], tcl_pred[1], sin_pred, cos_pred, radii_pred)  # (n_tcl, 3)
        # contours = self.adjust_contours(img_show, contours)

        output = {
            'image': image,
            'tr': tr_pred,
            'tcl': tcl_pred,
            'sin': sin_pred,
            'cos': cos_pred,
            'radii': radii_pred
        }
        return contours, output

    def adjust_contours(self, image, all_contours):
        mask = np.zeros(image.shape[0:2])
        # image_show = image.copy()

        bbox_contours = list()
        for idx, (boundary_point, line) in enumerate(all_contours):
            deal_mask = mask.copy()
            cv2.drawContours(deal_mask, [boundary_point], -1, 1, -1)
            deal_contours, _ = cv2.findContours(deal_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            new_line = list()
            for ip, _ in enumerate(line):
                if not (self.in_contour(deal_contours[0], line[ip, 0, :])\
                        or self.in_contour(deal_contours[0], line[ip, 1, :])):
                    new_line.append(line[ip])
            try:
                new_line = np.array(new_line)
                vet10 = np.array(new_line[0:-1, 0, :]) - np.array(new_line[1:, 0, :])
                vet20 = np.array(new_line[0:-1, 1, :]) - np.array(new_line[1:, 1, :])
            except:
                continue
            cosin0 = np.sum(vet10 * vet20, axis=1) / (norm2(vet10, axis=1) * norm2(vet20, axis=1))

            vet11 = np.array(new_line[0:-1, 0, :]) - np.array(new_line[0:-1, 1, :])
            vet21 = np.array(new_line[1:, 0, :]) - np.array(new_line[1:, 1, :])
            cosin1 = np.sum(vet11 * vet21, axis=1) / (norm2(vet11, axis=1) * norm2(vet21, axis=1))
            defect_point = (np.where((cosin0 < 0.6) & (cosin1 < 0.6))[0]).tolist()
            defect_size = (np.where((cosin0 < 0.6) & (cosin1 < 1))[0])

            if len(defect_point):
                defect_point = sorted(defect_point)
                dps = list()
                for index in defect_point:
                    iip = defect_size[np.where(np.abs(defect_size - index) <= 5)].tolist()
                    min_iip = min(iip)-1
                    max_iip = max(iip)+1
                    dps += list(range(min_iip, max_iip+1))

                defect_point.insert(0, 0)
                defect_point.append(len(new_line))
                defect_point = sorted(list(set(defect_point)))
                segline = np.stack([defect_point[0:-1], defect_point[1:]], axis=1)
                for seg in segline[1::2]:
                    new_line[seg[0]:seg[1]] = new_line[seg[0]:seg[1], ::-1, :]
                new_line = np.delete(new_line, dps, axis=0)

            if new_line.shape[0] < 4:
                continue
            boundary_point = np.concatenate([new_line[:, 0, :], new_line[::-1, 1, :]], axis=0)
            bbox_contours.append([boundary_point, new_line])

        return bbox_contours

    def merge_detect_contours(self, tr_pred, tcl_pred, sin_pred, cos_pred, radii_pred):

        # thresholding
        tr_pred_mask = tr_pred > self.tr_thresh
        tcl_pred_mask = tcl_pred > self.tcl_thresh
        mask = np.zeros_like(cos_pred)

        # regularize
        sin_pred, cos_pred = regularize_sin_cos(sin_pred, cos_pred)

        ret, labels = cv2.connectedComponents(tcl_pred_mask.astype(np.uint8), connectivity=8)
        bbox_contours = list()
        for idx in range(1, ret+1):
            bbox_mask = labels == idx

            tr_mask = bbox_mask * tr_pred_mask
            if np.sum(tr_mask) < 100:
                continue

            bbox_mask = fill_hole(bbox_mask)
            tcl_contours, _ = cv2.findContours(bbox_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # rect = cv2.minAreaRect(tcl_contours[0])
            # rect_area = int(rect[1][0]*rect[1][1])
            # ori_area = np.sum(bbox_mask)
            # flag = rect_area/ori_area > 1.5
            #
            # if flag:
            #     # find disjoint regions
            #     tr_mask = fill_hole(tr_mask)
            #     tcl_contours, _ = cv2.findContours(tr_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cont in tcl_contours:
                deal_map = mask.copy()
                cv2.drawContours(deal_map, [cont], -1, 1, -1)
                if deal_map.sum() <= 10:
                    continue
                text_map = tcl_pred * deal_map
                bboxs = self.bbox_transfor_inv(radii_pred, sin_pred, cos_pred, text_map, wclip=(4, 12))
                # nms
                boxes = lanms.merge_quadrangle_n9(bboxs.astype('float32'), 0.25)
                boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
                boundary_point = None
                if boxes.shape[0] > 1:
                    center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                    paths, routes_path = minConnectPath(center)
                    boxes = boxes[routes_path]
                    top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                    bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
                    edge0 = self.select_edge(top + bot[::-1], boxes[0])
                    edge1 = self.select_edge(top + bot[::-1], boxes[-1])
                    if edge0 is not None:
                        top.insert(0, edge0[0])
                        bot.insert(0, edge0[1])
                    if edge1 is not None:
                        top.append(edge1[0])
                        bot.append(edge1[1])
                    boundary_point = np.array(top + bot[::-1])

                elif boxes.shape[0] == 1:
                    top = boxes[0, 0:2, :].astype(np.int32).tolist()
                    bot = boxes[0, 2:4:-1, :].astype(np.int32).tolist()
                    boundary_point = np.array(top + bot)

                if boundary_point is None:
                    continue
                reconstruct_mask = mask.copy()
                cv2.drawContours(reconstruct_mask, [boundary_point], -1, 1, -1)
                if (reconstruct_mask * tr_pred_mask).sum() < reconstruct_mask.sum() * 0.5:
                    continue
                if reconstruct_mask.sum() < 100:
                    continue

                bbox_contours.append([boundary_point, np.array(np.stack([top, bot], axis=1))])

        return bbox_contours























