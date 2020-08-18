import cv2
import numpy as np
from util.pbox import minConnectPath
from util.misc import norm2
from util.config import config as cfg
from util.graph import graph_propagation, graph_propagation_soft, \
    graph_propagation_naive, single_remove, clusters2labels

from util import canvas as cav


class TextDetector(object):

    def __init__(self, model):
        self.model = model
        self.tr_thresh = cfg.tr_thresh
        self.tcl_thresh = cfg.tcl_thresh
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

    def detect_contours(self, bboxs, final_pred):

        bbox_contours = list()
        for idx in range(0, int(np.max(final_pred)) + 1):
            fg = np.where(final_pred == idx)
            boxes = bboxs[fg, :8].reshape((-1, 4, 2)).astype(np.int32)

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

            bbox_contours.append([boundary_point, np.array(np.stack([top, bot], axis=1))])

        return bbox_contours

    def detect(self, image, img_show):
        # get model output
        edges, scores, bboxs, output = self.model.forward_test_graph(image)

        contours = list()
        if edges is not None:
            clusters = graph_propagation_naive(edges, scores, cfg.link_thresh)
            final_pred = clusters2labels(clusters, bboxs.shape[0])
            bboxs, final_pred = single_remove(bboxs, final_pred)

            # find text contours
            contours = self.detect_contours(bboxs, final_pred)
            # contours = self.adjust_contours(img_show, contours)

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
                if not (self.in_contour(deal_contours[0], line[ip, 0, :])
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
                    min_iip = min(iip) - 1
                    max_iip = max(iip) + 1
                    dps += list(range(min_iip, max_iip + 1))

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




