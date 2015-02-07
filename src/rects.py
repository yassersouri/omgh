import numpy as np
import cv2
import cub_utils


class Rect(object):
    def __init__(self, xmin, xmax, ymin, ymax, info=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.info = info

    def __str__(self):
        return "Rect: \t xmin:%s \t xmax:%s \t ymin:%s \t ymax:%s \t\t info:%s" % (self.xmin, self.xmax, self.ymin, self.ymax, self.info)

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return Rect(self.xmin, self.xmax, self.ymin, self.ymax, self.info)

    @staticmethod
    def _expand_cendim(cendim, alpha):
        centerx, centery, dimx, dimy = cendim

        dimx = (2 * alpha) * dimx
        dimy = (2 * alpha) * dimy

        cendim = (centerx, centery, dimx, dimy)
        return cendim

    @staticmethod
    def _add_noise_to_cendim(cendim, center_std, dimension_std):
        centerx, centery, dimx, dimy = cendim

        if center_std > 0:
            centerx += np.random.normal(0, center_std)
            centery += np.random.normal(0, center_std)
        if dimension_std > 0:
            dimx += np.random.normal(0, dimension_std)
            dimy += np.random.normal(0, dimension_std)

        cendim = (centerx, centery, dimx, dimy)
        return cendim

    @staticmethod
    def _parse_bbox(bbox):
        by, bx, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        return bx, by, bw, bh

    def _get_cendim(self):
        centerx = float(self.xmin + self.xmax) / 2
        centery = float(self.ymin + self.ymax) / 2
        dimx = float(self.xmax - self.xmin)
        dimy = float(self.ymax - self.ymin)

        cendim = (centerx, centery, dimx, dimy)
        return cendim

    def _set_rect_from_cendim(self, cendim):
        centerx, centery, dimx, dimy = cendim
        self.xmin = int(round(centerx - dimx/2))
        self.xmax = int(round(centerx + dimx/2))
        self.ymin = int(round(centery - dimy/2))
        self.ymax = int(round(centery + dimy/2))

    def _trim_to_borders(self, img_shape):
        img_h, img_w = img_shape[:2]
        self.xmin = max(0, self.xmin)
        self.xmax = min(img_h - 1, self.xmax)
        self.ymin = max(0, self.ymin)
        self.ymax = min(img_w - 1, self.ymax)

    def draw_rect(self, img, color=100, width=2):
        """
        Annotate the `img` with this rect.
        """
        new_img = img.copy()

        cv2.rectangle(new_img, (self.ymin, self.xmin), (self.ymax, self.xmax), color, width)
        return new_img

    def get_rect(self, img):
        """
        Return a sub-image only containing information inside the rect.
        """
        self._trim_to_borders(img.shape)

        return img[self.xmin:self.xmax, self.ymin:self.ymax]

    def expand(self, alpha=0.666):
        cendim = self._get_cendim()
        cendim = Rect._expand_cendim(cendim, alpha)
        self._set_rect_from_cendim(cendim)

    def add_noise(self, center_std=1, dimension_std=1):
        cendim = self._get_cendim()
        cendim = Rect._add_noise_to_cendim(cendim, center_std, dimension_std)
        self._set_rect_from_cendim(cendim)

    def norm_for_size(self, source_shape, size=227):
        height, width = source_shape[:2]
        self.xmin = int(round(self.xmin * size / float(height)))
        self.xmax = int(round(self.xmax * size / float(height)))
        self.ymin = int(round(self.ymin * size / float(width)))
        self.ymax = int(round(self.ymax * size / float(width)))

    def denorm_for_size(self, source_shape, size=227):
        height, width = source_shape[:2]
        self.xmin = int(round(self.xmin * height / float(size)))
        self.xmax = int(round(self.xmax * height / float(size)))
        self.ymin = int(round(self.ymin * width / float(size)))
        self.ymax = int(round(self.ymax * width / float(size)))

    def norm_for_bbox(self, bbox_info):
        (bx, by, bw, bh) = Rect._parse_bbox(bbox_info)
        self.xmin -= bx
        self.xmax -= bx
        self.ymin -= by
        self.ymax -= by

    def denorm_for_bbox(self, bbox_info):
        (bx, by, bw, bh) = Rect._parse_bbox(bbox_info)
        self.xmin += bx
        self.xmax += bx
        self.ymin += by
        self.ymax += by

    def evalIOU(self, gt_rect, source_shape):
        # making sure not to generate errors futther down the line
        self._trim_to_borders(source_shape)
        gt_rect._trim_to_borders(source_shape)

        height, width = source_shape[:2]

        gt_part = np.zeros((height, width), np.uint8)
        gt_part[gt_rect.xmin:gt_rect.xmax, gt_rect.ymin:gt_rect.ymax] = 1

        sl_part = np.zeros((height, width), np.uint8)
        sl_part[self.xmin:self.xmax, self.ymin:self.ymax] = 1

        intersection = (gt_part & sl_part).sum()
        union = (gt_part | sl_part).sum()

        return intersection / float(union)

    def evalPCP(self, gt_rect, source_shape, thresh=0.5):
        iou = self.evalIOU(gt_rect, source_shape)
        if iou >= thresh:
            return 1
        else:
            return 0


class RectGenerator(object):
    def __init__(self):
        pass

    def setup(self):
        pass

    def generate(img_id):
        pass

    def generate_addr(img_path):
        pass


class BerkeleyRG(RectGenerator):
    def __init__(self, base_path, IDtrain, IDtest, rect_name):
        self.base_path = base_path
        self.IDtrain = IDtrain
        self.IDtest = IDtest
        self.rect_name = rect_name

    def setup(self):
        self.bah = cub_utils.BerkeleyAnnotationsHelper(self.base_path, self.IDtrain, self.IDtest)

    def generate(self, img_id):
        rect_info_raw = self.bah.get_berkeley_annotation(img_id, self.rect_name)
        xmin, xmax, ymin, ymax = rect_info_raw
        return Rect(xmin, xmax, ymin, ymax, info='GT - Berkley - imgid: %s' % img_id)


class SharifRG(RectGenerator):
    pass


class RandomForestRG(RectGenerator):
    pass


class LocalRandomForestRG(RectGenerator):
    pass


class NonparametricRG(RectGenerator):
    pass
