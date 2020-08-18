import copy
import cv2
import torch
import numpy as np
from PIL import Image
from util.config import config as cfg
from layers.proposal_layer import ProposalTarget
from util.misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin, split_edge_seqence_by_step, point_dist_to_line


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def Equal_width_bbox_cover(self, step=16.0):

        inner_points1, inner_points2 = split_edge_seqence_by_step(self.points, self.e1, self.e2, step=step)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center

        return inner_points1, inner_points2, center_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.proposal = ProposalTarget(k_at_hop1=cfg.k_at_hop1)

    @staticmethod
    def make_text_region(img, polygons):

        tr_mask = np.zeros(img.shape[:2], np.uint8)
        train_mask = np.ones(img.shape[:2], np.uint8)
		# tr_weight = np.ones(img.shape[:2], np.float)
        if polygons is None:
            return tr_mask, train_mask
		
		# region_masks = list()
        # num_positive_bboxes = 0
        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
                   # else:
            #     num_positive_bboxes +=1
            #     deal_mask = np.zeros_like(tr_mask)
            #     cv2.fillPoly(deal_mask, [polygon.points.astype(np.int32)], color=(1,))
            #     region_masks.append(deal_mask)

        # if cfg.weight_method == "BBOX_BALANCED":
        #     pos_region_mask = tr_mask*train_mask
        #     num_region_pixels = np.sum(pos_region_mask)
        #     for idx in range(len(region_masks)):
        #         bbox_region_mask = region_masks[idx] * pos_region_mask
        #         num_bbox_region_pixels = np.sum(bbox_region_mask)
        #         if num_bbox_region_pixels > 0:
        #             per_bbox_region_weight = num_region_pixels * 1.0 / num_positive_bboxes
        #             per_region_pixel_weight = per_bbox_region_weight / num_bbox_region_pixels
        #             tr_weight += bbox_region_mask * per_region_pixel_weight

        return tr_mask, train_mask

    @staticmethod
    def fill_polygon(mask, pts, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param pts: polygon to draw
        :param value: fill value
        """
        # cv2.drawContours(mask, [polygon.astype(np.int32)], -1, value, -1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], color=(value,))
        # rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(mask.shape[0],mask.shape[1]))
        # mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2,
                              center_line, tcl_msk1, tcl_msk2,
                              radius_map, sin_map, cos_map,
                              expand=0.3, shrink=1, width=1):

        mask = np.zeros_like(tcl_msk1)
        # TODO: shrink 1/2 * radius at two line end
        p1 = np.mean(sideline1, axis=0)
        p2 = np.mean(sideline2, axis=0)
        vpp = vector_sin(p1 - p2)
        if vpp >= 0:
            top_line = sideline2
            bot_line = sideline1
        else:
            top_line = sideline1
            bot_line = sideline2

        if len(center_line) < 5:
            shrink = 0

        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = top_line[i]
            top2 = top_line[i + 1]
            bottom1 = bot_line[i]
            bottom2 = bot_line[i + 1]
            top = (top1 + top2) / 2
            bottom = (bottom1 + bottom1) / 2

            sin_theta = vector_sin(top - bottom)
            cos_theta = vector_cos(top - bottom)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            ploy1 = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_msk1, ploy1, value=1)
            self.fill_polygon(sin_map, ploy1, value=sin_theta)
            self.fill_polygon(cos_map, ploy1, value=cos_theta)

            deal_mask = mask.copy()
            self.fill_polygon(deal_mask, ploy1, value=1)
            bbox_point_cords = np.argwhere(deal_mask == 1)
            for y, x in bbox_point_cords:
                point = np.array([x, y], dtype=np.float32)
                # top   h1
                radius_map[y, x, 0] = point_dist_to_line((top1, top2), point)  # 计算point到直线的距离
                # down  h2
                radius_map[y, x, 1] = point_dist_to_line((bottom1, bottom2), point)

            pp1 = c1 + (top1 - c1) * width/norm2(top1 - c1)
            pp2 = c1 + (bottom1 - c1) * width/norm2(bottom1 - c1)
            pp3 = c2 + (bottom2 - c2) * width/norm2(top1 - c1)
            pp4 = c2 + (top2 - c2) * width/norm2(bottom2 - c2)
            poly2 = np.stack([pp1, pp2, pp3, pp4])
            self.fill_polygon(tcl_msk2, poly2, value=1)

    def get_training_data(self, image, polygons, image_id, image_path):

        H, W, _ = image.shape
        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        tcl_mask = np.zeros((image.shape[0], image.shape[1], 2), np.uint8)
        radius_map = np.zeros((image.shape[0], image.shape[1], 2), np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        tcl_msk1 = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        tcl_msk2 = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        if polygons is not None:
            for i, polygon in enumerate(polygons):
                if polygon.text == '#':
                    continue
                polygon.find_bottom_and_sideline()
                sideline1, sideline2, center_points = polygon.Equal_width_bbox_cover(step=4.0)
                self.make_text_center_line(sideline1, sideline2, center_points,
                                           tcl_msk1, tcl_msk2, radius_map, sin_map, cos_map)

        tcl_mask[:, :, 0] = tcl_msk1
        tcl_mask[:, :, 1] = tcl_msk2
        tr_mask, train_mask = self.make_text_region(image, polygons)
        # clip value (0, 1)
        tcl_mask = np.clip(tcl_mask, 0, 1)
        tr_mask = np.clip(tr_mask, 0, 1)
        train_mask = np.clip(train_mask, 0, 1)

        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        if not self.is_training:
            points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
            length = np.zeros(cfg.max_annotation, dtype=int)
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    pts = polygon.points
                    points[i, :pts.shape[0]] = polygon.points
                    length[i] = pts.shape[0]

            meta = {
                'image_id': image_id,
                'image_path': image_path,
                'annotation': points,
                'n_annotation': length,
                'Height': H,
                'Width': W
            }

            return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta

        rpn_roi = self.proposal(tcl_mask, radius_map, sin_map, cos_map)

        gt_roi = np.zeros((cfg.max_roi, rpn_roi.shape[1]))
        gt_roi[:rpn_roi.shape[0], :] = rpn_roi[:cfg.max_roi]

        # gt_roi = np.zeros((cfg.max_roi, 9))

        image = torch.from_numpy(image).float()
        train_mask = torch.from_numpy(train_mask).byte()
        tr_mask = torch.from_numpy(tr_mask).byte()
        tcl_mask = torch.from_numpy(tcl_mask).byte()
        radius_map = torch.from_numpy(radius_map).float()
        sin_map = torch.from_numpy(sin_map).float()
        cos_map = torch.from_numpy(cos_map).float()
        gt_roi = torch.from_numpy(gt_roi).float()

        return image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
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
