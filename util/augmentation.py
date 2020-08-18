import numpy as np
import math
import cv2
import copy
import numpy.random as random

from shapely.geometry import Polygon


###<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<###
###<<<<<<<<<  Function  >>>>>>>>>>>>###
###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>###
def crop_first(image, polygons, scale =10):
    polygons_new = copy.deepcopy(polygons)
    h, w, _ = image.shape
    pad_h = h // scale
    pad_w = w // scale
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

    text_polys = []
    pos_polys = []
    for polygon in polygons_new:
        rect = cv2.minAreaRect(polygon.points.astype(np.int32))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        text_polys.append([box[0], box[1], box[2], box[3]])
        if polygon.label != -1:
            pos_polys.append([box[0], box[1], box[2], box[3]])

    polys = np.array(text_polys, dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)  # 四舍五入
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text 保证截取区域不会横穿文字
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    pp_polys = np.array(pos_polys, dtype=np.int32)

    return h_axis, w_axis, pp_polys

####<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<####
####<<<<<<<<<<<  Class  >>>>>>>>>>>>>####
####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>####
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons


class MinusMean(object):
    def __init__(self, mean):
        self.mean = np.array(mean)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image, polygons


class RandomMirror(object):
    # 镜像
    def __init__(self):
        pass

    def __call__(self, image, polygons=None):
        if polygons is None:
            return image, polygons
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
            for polygon in polygons:
                polygon.points[:, 0] = width - polygon.points[:, 0]
        return image, polygons


class AugmentColor(object):
    # 颜色增强（添加噪声）
    def __init__(self):
        self.U = np.array([[-0.56543481, 0.71983482, 0.40240142],
                      [-0.5989477, -0.02304967, -0.80036049],
                      [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32)
        self.EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)
        self.sigma = 0.1
        self.color_vec = None

    def __call__(self, img, polygons=None):
        color_vec = self.color_vec
        if self.color_vec is None:
            if not self.sigma > 0.0:
                color_vec = np.zeros(3, dtype=np.float32)
            else:
                color_vec = np.random.normal(0.0, self.sigma, 3)

        alpha = color_vec.astype(np.float32) * self.EV
        noise = np.dot(self.U, alpha.T) * 255
        return np.clip(img + noise[np.newaxis, np.newaxis, :], 0, 255), polygons


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, polygons=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return np.clip(image, 0, 255), polygons


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return np.clip(image, 0, 255), polygons


class RandomErasing(object):
    def __init__(self, sr=(0.0004, 0.01), scale=(0.5, 3), ratio=0.2, Type ="Erasing"):
        """

        :param area:
        :param type: Erasing or Cutout
        """
        self.sr = sr
        self.scale= scale
        self.ratio=ratio
        self.type=Type

    def __call__(self, img, polygons=None):

        if random.random()< self.ratio:
            return img, polygons
        area=img.shape[0]*img.shape[1]
        target_area=random.randint(*self.sr)*area
        aspect_ratio=random.uniform(*self.scale)
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        w = int(round(math.sqrt(target_area * aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[1] - w)
            y1 = random.randint(0, img.shape[0] - h)
            if self.type == "Erasing":
                color=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
                img[y1:y1+h, x1:x1+h,:]=color
            else:
                Gray_value=random.randint(0, 255)
                color = (Gray_value, Gray_value ,Gray_value)
                img[y1:y1 + h, x1:x1 + h, :] = color

        return img, polygons


class RandomMixUp(object):
    def __init__(self, mixup_alpha=2):
        self.mixup_alpha = mixup_alpha

    def __call__(self, img1, img2, label1=[], label2=[]):
        beta=np.random.beta(self.mixup_alpha,self.mixup_alpha)

        #image = img1 * Gama + (1 - Gama) * img2
        image=cv2.addWeighted(img1, beta, img2, (1-beta), 0)

        if label1 is None or label2  is None:
            return  img1, label1
        if isinstance(label1, list) and isinstance(label2, list):
            label=[]
            for id in range(len(label1)):
                lab = beta*label1[id]+ (1-beta)*label2[id]
                label.append(lab)
            return image, label
        else:
            print("Error: label is not a list type")

        return img1, label1


class Rotate(object):
    def __init__(self, up=30):
        self.up = up

    @staticmethod
    def rotate(center, pt, theta):  # 二维图形学的旋转
        xr, yr = center
        yr = -yr
        x, y = pt[:, 0], pt[:, 1]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        _x = xr + (x - xr) * cos - (y - yr) * sin
        _y = yr + (x - xr) * sin + (y - yr) * cos

        return _x, -_y

    def __call__(self, img, polygons=None):
        if np.random.randint(2):
            return img, polygons
        angle = np.random.normal(loc=0.0, scale=0.5) * self.up  # angle 按照高斯分布
        rows, cols = img.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])
        center = cols / 2.0, rows / 2.0
        if polygons is not None:
            for polygon in polygons:
                x, y = self.rotate(center, polygon.points, angle)
                pts = np.vstack([x, y]).T
                polygon.points = pts
        return img, polygons


class RotatePadding(object):
    def __init__(self, up=60,colors=True):
        self.up = up
        self.colors = colors
        self.ratio = 0.5

    @staticmethod
    def rotate(center, pt, theta, movSize=[0, 0], scale=1):  # 二维图形学的旋转
        (xr, yr) = center
        yr = -yr
        x, y = pt[:, 0], pt[:, 1]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - xr) * scale
        y = (y - yr) * scale

        _x = xr + x * cos - y * sin + movSize[0]
        _y = -(yr + x * sin + y * cos) + movSize[1]

        return _x, _y

    @staticmethod
    def shift(size, degree):
        angle = degree * math.pi / 180.0
        width = size[0]
        height = size[1]

        alpha = math.cos(angle)
        beta = math.sin(angle)
        new_width = int(width * math.fabs(alpha) + height * math.fabs(beta))
        new_height = int(width * math.fabs(beta) + height * math.fabs(alpha))

        size = [new_width, new_height]
        return size

    def __call__(self, image, polygons=None, scale=1.0):
        if np.random.random() <= self.ratio:
            return image, polygons
        angle = np.random.normal(loc=0.0, scale=0.5) * self.up  # angle 按照高斯分布
        rows, cols = image.shape[0:2]
        center = (cols / 2.0, rows / 2.0)
        newSize = self.shift([cols * scale, rows * scale], angle)
        movSize = [int((newSize[0] - cols) / 2), int((newSize[1] - rows) / 2)]

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += int((newSize[0] - cols) / 2)
        M[1, 2] += int((newSize[1] - rows) / 2)

        if self.colors:
            H, W, _ = image.shape
            mask = np.zeros_like(image)
            (h_index, w_index) = (np.random.randint(0, H * 7 // 8), np.random.randint(0, W * 7 // 8))
            img_cut = image[h_index:(h_index + H // 9), w_index:(w_index + W // 9)]
            img_cut = cv2.resize(img_cut, (newSize[0], newSize[1]))
            mask = cv2.warpAffine(mask, M, (newSize[0], newSize[1]), borderValue=[1, 1, 1])
            image = cv2.warpAffine(image, M, (newSize[0], newSize[1]), borderValue=[0,0,0])
            image=image+img_cut*mask
        else:
            color = [0, 0, 0]
            image = cv2.warpAffine(image, M, (newSize[0], newSize[1]), borderValue=color)

        if polygons is not None:
            for polygon in polygons:
                x, y = self.rotate(center, polygon.points, angle,movSize,scale)
                pts = np.vstack([x, y]).T
                polygon.points = pts
        return image, polygons


class SquarePadding(object):

    def __call__(self, image, polygons=None):

        H, W, _ = image.shape

        if H == W:
            return image, polygons

        padding_size = max(H, W)
        (h_index, w_index) = (np.random.randint(0, H*7//8),np.random.randint(0, W*7//8))
        img_cut = image[h_index:(h_index+H//9),w_index:(w_index+W//9)]
        expand_image = cv2.resize(img_cut,(padding_size, padding_size))
        #expand_image = np.zeros((padding_size, padding_size, 3), dtype=image.dtype)
        #expand_image=img_cut[:,:,:]
        if H > W:
            y0, x0 = 0, (H - W) // 2
        else:
            y0, x0 = (W - H) // 2, 0
        if polygons is not None:
            for polygon in polygons:
                polygon.points += np.array([x0, y0])
        expand_image[y0:y0+H, x0:x0+W] = image
        image = expand_image

        return image, polygons


class RandomImgCropPatch(object):
    def __init__(self, up=30, beta=0.3):
        self.up = up
        self.beta=0.3
        self.scale = 10

    @staticmethod
    def get_contour_min_area_box(contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    def CropWH(self, image, cut_w, cut_h, polygons=None):
        h_axis, w_axis, polys = crop_first(image, polygons, scale=self.scale)
        h, w, _ = image.shape
        pad_h = h // self.scale
        pad_w = w // self.scale
        # TODO try Flip
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = xmin + cut_w
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = ymin + cut_h
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        cropped = image[ymin:ymax + 1, xmin:xmax + 1, :]
        polygons_new = []
        for idx in selected_polys:
            polygon = polygons[idx]
            polygon.points -= np.array([xmin, ymin])
            polygons_new.append(polygon)
        image = cropped
        polygon = polygons_new

        return image, polygon

    def __call__(self, images, polygons_list=None):
        I_x, I_y = 1024,1024

        w = int(round(I_x * random.beta(self.beta, self.beta)))
        h = int(round(I_y * random.beta(self.beta, self.beta)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]
        new_img = np.zeros((I_x, I_y, 3), dtype=images[0].dtype)
        imgs=[]
        new_polygons=[]
        for i, im in enumerate(images):
           img, polygons = self.CropWH(im,  w_[i],  h_[i], polygons=polygons_list[i])
           imgs.append(img)
           new_polygons.append(polygons)
        new_img[0:w, 0:h, :] = imgs[0]
        new_img[w:I_x, 0:h, :] = imgs[1]
        new_img[0:w, h:I_y, :] = imgs[2]
        new_img[w:I_x, h:I_y, :] = imgs[3]
        for polygon in new_polygons[1]:
            polygon.points += np.array([w, 0])
        for polygon in new_polygons[2]:
            polygon.points += np.array([0, h])
        for polygon in new_polygons[3]:
            polygon.points += np.array([w, h])

        polygons=new_polygons[0]+new_polygons[1]+new_polygons[2]+new_polygons[3]

        return new_img, polygons


class RandomCropFlip(object):

    def __init__(self, min_crop_side_ratio=0.2):
        self.scale=10
        self.ratio =0.5
        self.epsilon =1e-2
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, image, polygons=None):

        if polygons is None:
            return image, polygons

        if np.random.random() <= self.ratio:
            return image, polygons

        # 计算 有效的Crop区域, 方便选取有效的种子点
        h_axis, w_axis, pp_polys = crop_first(image, polygons, scale =self.scale)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return image, polygons

        # TODO try crop
        attempt = 0
        h, w, _ = image.shape
        area = h * w
        pad_h = h // self.scale
        pad_w = w // self.scale
        while attempt < 10:
            attempt += 1
            polygons_new = []
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin) * (ymax - ymin) < area *  self.min_crop_side_ratio:
                # area too small
                continue

            pts = np.stack([[xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
            pp = Polygon(pts).buffer(0)
            Fail_flag = False
            for polygon in polygons:
                ppi = Polygon(polygon.points).buffer(0)
                ppiou = float(ppi.intersection(pp).area)
                if np.abs(ppiou - float(ppi.area)) >self.epsilon  and np.abs(ppiou)> self.epsilon:
                    Fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area)) <self.epsilon:
                    polygons_new.append(polygon)
                else:
                    pass

            if Fail_flag:
                continue
            else:
                break

        if len(polygons_new) == 0:
            cropped = image[ymin:ymax, xmin:xmax, :]
            select_type = random.randint(3)
            if select_type == 0:
                img = np.ascontiguousarray(cropped[:, ::-1])
            elif select_type == 1:
                img = np.ascontiguousarray(cropped[::-1, :])
            else:
                img = np.ascontiguousarray(cropped[::-1, ::-1])
            image[ymin:ymax, xmin:xmax, :] = img
            return image, polygons

        else:

            cropped = image[ymin:ymax, xmin:xmax, :]
            height, width, _ = cropped.shape
            select_type = random.randint(3)
            if select_type == 0:
                img = np.ascontiguousarray(cropped[:, ::-1])
                for polygon in polygons_new:
                    polygon.points[:, 0] = width - polygon.points[:, 0] + 2 * xmin
            elif select_type == 1:
                img = np.ascontiguousarray(cropped[::-1, :])
                for polygon in polygons_new:
                    polygon.points[:, 1] = height - polygon.points[:, 1] + 2 * ymin
            else:
                img = np.ascontiguousarray(cropped[::-1, ::-1])
                for polygon in polygons_new:
                    polygon.points[:, 0] = width - polygon.points[:, 0] + 2 * xmin
                    polygon.points[:, 1] = height - polygon.points[:, 1] + 2 * ymin
            image[ymin:ymax, xmin:xmax, :] = img

        return image, polygons


class RandomResizedCrop(object):
    def __init__(self, min_crop_side_ratio = 0.2):
        self.scale = 10
        self.epsilon = 1e-2
        self.min_crop_side_ratio = min_crop_side_ratio

    def __call__(self, image, polygons):

        if polygons is None:
            return image, polygons

        # 计算 有效的Crop区域, 方便选取有效的种子点
        h_axis, w_axis, pp_polys = crop_first(image, polygons, scale =self.scale)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return image, polygons

        # TODO try crop
        attempt = 0
        h, w, _ = image.shape
        area = h * w
        pad_h = h // self.scale
        pad_w = w // self.scale
        while attempt < 10:
            attempt += 1
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin)*(ymax - ymin) <area*self.min_crop_side_ratio:
                # area too small
                continue
            if pp_polys.shape[0] != 0:
                poly_axis_in_area = (pp_polys[:, :, 0] >= xmin) & (pp_polys[:, :, 0] <= xmax) \
                                    & (pp_polys[:, :, 1] >= ymin) & (pp_polys[:, :, 1] <= ymax)
                selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polys = []

            if len(selected_polys) == 0:
                continue
            else:
                pts = np.stack([[xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
                pp = Polygon(pts).buffer(0)
                polygons_new = []
                Fail_flag = False
                for polygon in copy.deepcopy(polygons):
                    ppi = Polygon(polygon.points).buffer(0)
                    ppiou = float(ppi.intersection(pp).area)
                    if np.abs(ppiou - float(ppi.area)) > self.epsilon and np.abs(ppiou) > self.epsilon:
                        Fail_flag = True
                        break
                    elif np.abs(ppiou - float(ppi.area)) < self.epsilon:
                        # polygon.points -= np.array([xmin, ymin])
                        polygons_new.append(polygon)

                if Fail_flag:
                    continue
                else:
                    cropped = image[ymin:ymax + 1, xmin:xmax + 1, :]
                    for polygon in polygons_new:
                        polygon.points -= np.array([xmin, ymin])

                    return cropped, polygons_new

        return image, polygons


class RandomResizeScale(object):
    def __init__(self, size=512, ratio=(3./4, 5./2)):
        self.size = size
        self.ratio = ratio

    def __call__(self, image, polygons=None):

        aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])
        h, w, _ = image.shape
        scales = self.size*1.0/max(h, w)
        aspect_ratio = scales * aspect_ratio
        aspect_ratio = int(w * aspect_ratio)*1.0/w
        image = cv2.resize(image, (int(w * aspect_ratio), int(h*aspect_ratio)))
        scales = np.array([aspect_ratio, aspect_ratio])
        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


class Resize(object):
    def __init__(self, size=(480, 1024)):
        self.size = size
        self.SP = SquarePadding()

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size,
                                   self.size))
        scales = np.array([self.size / w, self.size / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


class ResizeSquare(object):
    def __init__(self, size=(480, 1280)):
        self.size = size

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        img_size_min = min(h, w)
        img_size_max = max(h, w)

        if img_size_min < self.size[0]:
            im_scale = float(self.size[0]) / float(img_size_min)  # expand min to size[0]
            if np.round(im_scale * img_size_max) > self.size[1]:  # expand max can't > size[1]
                im_scale = float(self.size[1]) / float(img_size_max)
        elif img_size_max > self.size[1]:
            im_scale = float(self.size[1]) / float(img_size_max)
        else:
            im_scale = 1.0

        new_h = int(int(h * im_scale/32)*32)
        new_w = int(int(w * im_scale/32)*32)
        image = cv2.resize(image, (new_w, new_h))
        scales = np.array([new_w / w, new_h / h])
        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


class ResizeLimitSquare(object):
    def __init__(self, size=512, ratio=0.6):
        self.size = size
        self.ratio = ratio
        self.SP = SquarePadding()

    def __call__(self, image, polygons=None):
        if np.random.random() <= self.ratio:
            image, polygons = self.SP(image, polygons)
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size,self.size))
        scales = np.array([self.size*1.0/ w, self.size*1.0 / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


class RandomResizePadding(object):
    def __init__(self, size=512, random_scale=np.array([0.75, 1.0, 1.25,1.5,2.0]),stride=32, ratio=0.6667):
        self.random_scale = random_scale
        self.size = size
        self.ratio=ratio
        self.stride=stride
        self.SP=SquarePadding()

        ###########Random size for different eproches ########################
        rd_scale = np.random.choice(self.random_scale)
        step_num = round(np.random.normal(loc=0.0, scale=0.35) * 8)  # step 按照高斯分布
        self.input_size = np.clip(int(self.size * rd_scale + step_num * self.stride),
                                  (int(self.size * self.random_scale[0] - self.stride)),
                                  int(self.size * self.random_scale[-1] + self.stride))
        ############################ end ########################

    def __call__(self, image, polygons=None):

        if np.random.random() <= self.ratio:
            image, polygons = self.SP(image, polygons)
        h, w, _ = image.shape
        image = cv2.resize(image, (self.input_size,self.input_size))
        scales = np.array([self.input_size*1.0/ w, self.input_size*1.0 / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


class Augmentation(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            RandomResizeScale(size=self.size, ratio=(3. / 4, 5. / 2)),
            RandomCropFlip(),
            RandomResizedCrop(),
            RotatePadding(up=60, colors=True),
            # RandomResizePadding(size=self.size, random_scale=self.input_scale),
            ResizeLimitSquare(size=self.size),
            # RandomBrightness(),
            # RandomContrast(),
            RandomMirror(),
            Normalize(mean=self.mean, std=self.std),
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            # Resize(size),
            ResizeSquare(size=self.size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class BaseTransformNresize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)
