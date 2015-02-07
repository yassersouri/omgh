import cv2
import numpy as np
import copy
from itertools import ifilter


def gen_dense_points(xdim=227, ydim=227):
    dense_points = Parts()
    for i in range(xdim):
        for j in range(ydim):
            dense_points.append(Part(-1, '?', -1, i, j, 1))
    return dense_points


class Part(object):

    def __init__(self, img_id, part_name, part_id, x, y, is_visible):
        self.img_id = img_id
        self.part_name = part_name
        self.part_id = part_id
        self.x = x
        self.y = y
        self.is_visible = is_visible

    def __str__(self):
        return "img_id:%d \tpart_name:%s \tpart_id:%d x:%d \ty:%d \tis_visible:%d" % (self.img_id, self.part_name.ljust(20), self.part_id, self.x, self.y, self.is_visible)

    def __repr__(self):
        return self.__str__()

    def is_part(self, part_name):
        return part_name == self.part_name

    def is_part_id(self, part_id):
        return part_id == self.part_id


class Parts(object):

    HEAD_PART_NAMES = ['beak', 'crown', 'forehead', 'nape', 'right eye', 'throat', 'left eye']
    BODY_PART_NAMES = ['back', 'belly', 'breast', 'left leg', 'left wing', 'right leg', 'right wing', 'tail']

    def __init__(self, parts=None):
        if hasattr(parts, '__iter__'):
            self.parts = parts
        elif parts is None:
            self.parts = []
        else:
            self.parts = [parts]

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return self.parts.__iter__()

    def __str__(self):
        string = '\n'.join([str(p) for p in self])
        return '[' + string + ']'

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        return self.parts[key]

    def filter_by_name(self, names):
        return Parts(list(ifilter(lambda part: part.part_name in names, self.parts)))

    def center(self):
        mean_x, mean_y = 0., 0.

        for part in self.parts:
            mean_x += part.x
            mean_y += part.y

        if len(self) == 0:
            return 0, 0
        return mean_x/len(self), mean_y/len(self)

    def bounding_width_height(self):
        """
        FIXME: I only now noticed that this is such and ugly code. Why not just use the min and max instead of writing them verbosly!
        """
        min_x, max_x, min_y, max_y = 100000, 0, 100000, 0

        for part in self.parts:
            if min_x > part.x:
                min_x = part.x
            if min_y > part.y:
                min_y = part.y

            if max_x < part.x:
                max_x = part.x
            if max_y < part.y:
                max_y = part.y

        return max_x - min_x, max_y - min_y

    def draw_part(self, ax, color=None):
        for part in self:
            if color is None:
                ax.plot(part.x, part.y, 'o')
            else:
                ax.plot(part.x, part.y, 'o', color=color)

    def get_rect_info(self, img_shape, alpha=0.6666, add_noise=False, noise_std_c=5.0, noise_std_d=10.0):
        c_x, c_y = self.center()
        # if we actually have no choice
        if c_x == 0 and c_y == 0:
            c_x = img_shape[1]/2.0
            c_y = img_shape[0]/2.0
        if add_noise:
            # add noise here
            if noise_std_c > 0:
                c_x += np.random.normal(0, noise_std_c)
                c_y += np.random.normal(0, noise_std_c)
        c_x, c_y = int(c_x), int(c_y)

        w, h = self.bounding_width_height()
        if add_noise:
            # add noise here
            if noise_std_d > 0:
                w += np.random.normal(0, noise_std_d)
                h += np.random.normal(0, noise_std_d)
        # just to prevent things from breaking
        if w < 10:
            w = 10
        if h < 10:
            h = 10
        xmin = int(max(0, (c_y - h * alpha)))
        xmax = int(min(img_shape[0] - 1, (c_y + h * alpha)))
        ymin = int(max(0, (c_x - w * alpha)))
        ymax = int(min(img_shape[1] - 1, (c_x + w * alpha)))

        return xmin, xmax, ymin, ymax

    def draw_rect(self, img, alpha=0.6666, color=100, width=2, rect_info=None):
        if rect_info is None:
            xmin, xmax, ymin, ymax = self.get_rect_info(img.shape, alpha)
        else:
            xmin, xmax, ymin, ymax = rect_info
        new_img = img.copy()
        # because opencv doesn't use the sane convention
        cv2.rectangle(new_img, (ymin, xmin), (ymax, xmax), color, width)
        return new_img

    def get_rect(self, img, alpha=0.6666, add_noise=False, noise_std_c=5.0, noise_std_d=10.0, rect_info=None):
        if rect_info is None:
            xmin, xmax, ymin, ymax = self.get_rect_info(img.shape, alpha, add_noise, noise_std_c, noise_std_d)
        else:
            xmin, xmax, ymin, ymax = rect_info

        return img[xmin:xmax, ymin:ymax]

    def get_gray_out_rect(self, img):
        c_x, c_y = self.center()
        c_x, c_y = int(c_x), int(c_y)

        w, h = self.bounding_width_height()

        mul = 2
        div = 3

        xmin = max(0, (c_y - h*mul/div))
        xmax = min(img.shape[0]-1, (c_y + h*mul/div))
        ymin = max(0, (c_x - w*mul/div))
        ymax = min(img.shape[1]-1, (c_x + w*mul/div))

        new_img = np.ones_like(img) * 0.5
        new_img[xmin:xmax, ymin:ymax] = img[xmin:xmax, ymin:ymax]

        return new_img

    def norm_for_bbox(self, bx, by):
        for part in self.parts:
            part.x = part.x - bx
            part.y = part.y - by
        return self

    def denorm_for_bbox(self, bx, by):
        for part in self.parts:
            part.x = part.x + bx
            part.y = part.y + by
        return self

    def norm_for_size(self, bw, bh, size=256):
        size = float(size)
        for part in self.parts:
            part.x = int(round(part.x * size / bw))
            part.y = int(round(part.y * size / bh))
        return self

    def denorm_for_size(self, bw, bh, size=256):
        size = float(size)
        for part in self.parts:
            part.x = int(round(part.x * bw / size))
            part.y = int(round(part.y * bh / size))
        return self

    def transfer(self, s_bbox, d_bbox, size=256):
        new_parts = copy.deepcopy(self)
        sbx, sby, sbw, sbh = int(s_bbox[0]), int(s_bbox[1]), int(s_bbox[2]), int(s_bbox[3])
        dbx, dby, dbw, dbh = int(d_bbox[0]), int(d_bbox[1]), int(d_bbox[2]), int(d_bbox[3])
        new_parts.norm_for_bbox(sbx, sby)
        new_parts.norm_for_size(sbw, sbh)
        new_parts.denorm_for_size(dbw, dbh)
        new_parts.denorm_for_bbox(dbx, dby)
        return new_parts

    def append(self, part):
        self.parts.append(part)

    def appends(self, parts):
        for part in parts:
            self.append(part)

    def for_image(self, img_id):
        return Parts(list(ifilter(lambda part: part.img_id == img_id, self.parts)))

    def set_for(self, img_id):
        for part in self.parts:
            part.img_id = img_id


class CUBParts(object):

    PART_NUMBERS = {'back': 1, 'beak': 2, 'belly': 3, 'breast': 4, 'crown': 5, 'forehead': 6, 'left eye': 7, 'left leg': 8, 'left wing': 9, 'nape': 10, 'right eye': 11, 'right leg': 12, 'right wing': 13, 'tail': 14, 'throat': 15}

    PART_NAMES = {v: k for k, v in PART_NUMBERS.items()}

    DIM_IMG_ID = 0
    DIM_PART_ID = 1
    DIM_PART_X = 2
    DIM_PART_Y = 3
    DIM_VISIBLE = 4

    IS_VISIBLE = 1

    def __init__(self, info, bbox=None):
        self.info = info
        if bbox is not None:
            self.bbox = bbox
        else:
            self.bbox = None

    def for_image(self, img_id):
        """
        returns a list of Part objects
        """
        img_id = int(img_id)
        related_info = self.info[(self.info[:, self.DIM_IMG_ID] == img_id) & (self.info[:, self.DIM_VISIBLE] == self.IS_VISIBLE), :]
        parts = []
        for r_i in related_info:
            img_id = r_i[self.DIM_IMG_ID]
            part_id = r_i[self.DIM_PART_ID]
            part_name = self.PART_NAMES[part_id]
            part_x = r_i[self.DIM_PART_X]
            part_y = r_i[self.DIM_PART_Y]
            is_visible = r_i[self.DIM_VISIBLE]
            parts.append(Part(img_id, part_name, part_id, part_x, part_y, is_visible))

        return Parts(parts)
