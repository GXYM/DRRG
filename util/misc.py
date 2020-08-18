import numpy as np
import errno
import os
import cv2
import math
from shapely.geometry import Polygon
from util.config import config as cfg


def to_device(*tensors):
    if len(tensors) < 2:
        return tensors[0].to(cfg.device)
    return (t.to(cfg.device) for t in tensors)


def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


def rescale_result(image, bbox_contours, H, W):
    ori_H, ori_W = image.shape[:2]
    image = cv2.resize(image, (W, H))
    contours = list()
    for (cont, lines) in bbox_contours:
        cont[:, 0] = (cont[:, 0] * W / ori_W).astype(int)
        cont[:, 1] = (cont[:, 1] * H / ori_H).astype(int)
        contours.append(cont)
    return image, contours


def fill_hole(input_mask):
    h, w = input_mask.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    mask = np.zeros((h + 4, w + 4), np.uint8)

    cv2.floodFill(canvas, mask, (0, 0), 1)
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool)

    return (~canvas | input_mask.astype(np.uint8))


def regularize_sin_cos(sin, cos):
    # regularization
    scale = np.sqrt(1.0 / (sin ** 2 + cos ** 2))
    return sin * scale, cos * scale


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1, delte=6):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / delte)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def point_dist_to_line(line, p3):
    # 计算点到直线的距离
    # line = (p1, p2)
    # compute the distance from p3 to p1-p2 #cross(x,y)矩阵的叉积，norm()求范数
    # np.linalg.norm(np.cross(p2 - p1, p1 - p3)) * 1.0 / np.linalg.norm(p2 - p1)
    # compute the distance from p3 to p1-p2
    p1, p2 = line
    d = p2 - p1

    def l2(p):
        return math.sqrt(p[0] * p[0]+ p[1]*p[1])

    if l2(d) > 0:
        distance = abs(d[1] * p3[0] - d[0] * p3[1] + p2[0] * p1[1] - p2[1] * p1[0]) / l2(d)
    else:
        distance = math.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)

    return distance


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))


def vector_sin(v):
    assert len(v) == 2
    # sin = y / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-5
    return v[1] / l


def vector_cos(v):
    assert len(v) == 2
    # cos = x / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2) + 1e-5
    return v[0] / l


def find_bottom(pts):

    if len(pts) > 4:
        e = np.concatenate([pts, pts[:3]])
        candidate = []
        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i + 2] - e[i + 1]
            if cos(v_prev, v_next) < -0.875:
                candidate.append((i % len(pts), (i + 1) % len(pts), norm2(e[i] - e[i + 1])))

        if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            dist_list = []
            if len(candidate) > 2:

                bottom_idx = np.argsort([angle for s1, s2, angle in candidate])[0:2]
                bottoms = [candidate[bottom_idx[0]][:2], candidate[bottom_idx[1]][0:2]]
                long_edge1, long_edge2 = find_long_edges(pts, bottoms)
                edge_length1 = [norm2(pts[e1] - pts[e2]) for e1, e2 in long_edge1]
                edge_length2 = [norm2(pts[e1] - pts[e2]) for e1, e2 in long_edge2]
                l1 = sum(edge_length1)
                l2 = sum(edge_length2)
                len1 = len(edge_length1)
                len2 = len(edge_length2)

                if l1 > 2*l2 or l2 > 2*l1 or len1 == 0 or len2 == 0:
                    for i in range(len(pts)):
                        mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                        mid_list.append((i, (i + 1) % len(pts), mid_point))

                    for i in range(len(pts)):
                        for j in range(len(pts)):
                            s1, e1, mid1 = mid_list[i]
                            s2, e2, mid2 = mid_list[j]
                            dist = norm2(mid1 - mid2)
                            dist_list.append((s1, e1, s2, e2, dist))
                    bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-1]
                    bottoms = [dist_list[bottom_idx][:2], dist_list[bottom_idx][2:4]]
            else:
                mid_list = []
                for i in range(len(pts)):
                    mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                    mid_list.append((i, (i + 1) % len(pts), mid_point))

                dist_list = []
                for i in range(len(pts)):
                    for j in range(len(pts)):
                        s1, e1, mid1 = mid_list[i]
                        s2, e2, mid2 = mid_list[j]
                        dist = norm2(mid1 - mid2)
                        dist_list.append((s1, e1, s2, e2, dist))
                bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
                bottoms = [dist_list[bottom_idx[0]][:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]
    else:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if 2*d1 < d2 else [(1, 2), (3, 0)]
        # bottoms = [(0, 1), (2, 3)] if 2 * d1 < d2 and d1 > 32 else [(1, 2), (3, 0)]
    assert len(bottoms) == 2, 'fewer than 2 bottoms'
    return bottoms


def split_long_edges(points, bottoms):
    """
    Find two long edge sequence of and polygon
    """
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)

    i = b1_end + 1
    long_edge_1 = []
    while i % n_pts != b2_end:
        long_edge_1.append((i - 1, i))
        i = (i + 1) % n_pts

    i = b2_end + 1
    long_edge_2 = []
    while i % n_pts != b1_end:
        long_edge_2.append((i - 1, i))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def find_long_edges(points, bottoms):
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)
    i = (b1_end + 1) % n_pts
    long_edge_1 = []

    while i % n_pts != b2_end:
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_1.append((start, end))
        i = (i + 1) % n_pts

    i = (b2_end + 1) % n_pts
    long_edge_2 = []
    while i % n_pts != b1_end:
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_2.append((start, end))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def split_edge_seqence(points, long_edge, n_parts):

    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while cur_end > point_cumsum[cur_node + 1]:
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


def split_edge_seqence_by_step(points, long_edge1, long_edge2, step=16.0):

    edge_length1 = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge1]
    edge_length2 = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge2]
    # 取长边 计算bbox个数
    total_length = (sum(edge_length1)+sum(edge_length2))/2
    n_parts = math.ceil(float(total_length) / step)
    try:
        inner1 = split_edge_seqence(points, long_edge1, n_parts=n_parts)
        inner2 = split_edge_seqence(points, long_edge2, n_parts=n_parts)
    except:
        print(edge_length1)
        print(edge_length2)

    return inner1, inner2


def disjoint_find(x, F):
    if F[x] == x:
        return x
    F[x] = disjoint_find(F[x], F)
    return F[x]


def disjoint_merge(x, y, F):
    x = disjoint_find(x, F)
    y = disjoint_find(y, F)
    if x == y:
        return False
    F[y] = x
    return True


def merge_polygons(polygons, merge_map):

    def merge_two_polygon(p1, p2):
        p2 = Polygon(p2)
        merged = p1.union(p2)
        return merged

    merge_map = [disjoint_find(x, merge_map) for x in range(len(merge_map))]
    merge_map = np.array(merge_map)
    final_polygons = []

    for i in np.unique(merge_map):
        merge_idx = np.where(merge_map == i)[0]
        if len(merge_idx) > 0:
            merged = Polygon(polygons[merge_idx[0]])
            for j in range(1, len(merge_idx)):
                merged = merge_two_polygon(merged, polygons[merge_idx[j]])
            x, y = merged.exterior.coords.xy
            final_polygons.append(np.stack([x, y], axis=1).astype(int))

    return final_polygons


