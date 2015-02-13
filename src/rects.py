import numpy as np
import cv2
import cub_utils
import parts
import utils


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

    def denorm_for_size(self, dest_shape, size=227):
        height, width = dest_shape[:2]
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
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def generate(img_id):
        raise NotImplementedError

    def generate_addr(img_path):
        raise NotImplementedError


class BerkeleyRG(RectGenerator):
    def __init__(self, base_path, cub, rect_name):
        self.base_path = base_path
        self.cub = cub
        self.rect_name = rect_name

    def setup(self):
        self.IDtrain, self.IDtest = self.cub.get_train_test_id()
        self.bah = cub_utils.BerkeleyAnnotationsHelper(self.base_path, self.IDtrain, self.IDtest)

    def generate(self, img_id):
        rect_info_raw = self.bah.get_berkeley_annotation(img_id, self.rect_name)
        xmin, xmax, ymin, ymax = rect_info_raw
        return Rect(xmin, xmax, ymin, ymax, info='GT - Berkley - imgid: %s' % img_id)

    def generate_addr(self, img_path):
        raise NotImplementedError("For a ground truth generator this is impossible.")


class SharifRG(RectGenerator):
    def __init__(self, cub, part_name, alpha=0.6666):
        self.cub = cub
        if part_name == 'body':
            self.part_filter_name = parts.Parts.BODY_PART_NAMES
        elif part_name == 'head':
            self.part_filter_name = parts.Parts.HEAD_PART_NAMES
        self.alpha = alpha

    def setup(self):
        self.cub_parts = self.cub.get_parts()
        self.all_image_infos = self.cub.get_all_image_infos()

    def generate(self, img_id, img_shape=None):
        if img_shape is None:
            img = cv2.imread(self.all_image_infos[img_id])
            img_shape = img.shape
        parts_for_img = self.cub_parts.for_image(img_id).filter_by_name(self.part_filter_name)
        rect_info_raw = parts_for_img.get_rect_info(img_shape, alpha=self.alpha)
        xmin, xmax, ymin, ymax = rect_info_raw
        return Rect(xmin, xmax, ymin, ymax, info='GT - Sharif - imgid: %s' % img_id)


class RandomForestRG(RectGenerator):
    pass


class LocalRandomForestRG(RectGenerator):
    pass


class NonparametricRG(RectGenerator):
    def __inti__(self, nn_finder, neighbor_gen, cub):
        self.nn_finder = nn_finder
        self.neighbor_gen = neighbor_gen
        self.cub = cub

    def setup(self):
        self.nn_finder.setup()
        self.neighbor_gen.setup()

        self.IDtrain, self.IDtest = self.cub.get_train_test_id()
        self.all_image_infos = self.cub.get_all_image_infos()
        self.bboxes = self.cub.get_bbox()

    def generate(self, img_id):
        query_img = cv2.imread(self.all_image_infos[img_id])
        query_bbox = self.bboxes[img_id - 1]
        query_subimg = utils.get_rect_from_bbox(query_img, query_bbox)
        query_img_shape = query_subimg.shape

        # find the neighbor
        nn_in_train_id = self.nn_finder.find_in_train(img_id)

        # generate the rect and other information from rect generator
        nn_in_train_rect = self.neighbor_gen.generate(nn_in_train_id)
        nn_in_train_img = cv2.imread(self.all_image_infos[nn_in_train_id])
        nn_in_train_bbox = self.bboxes[nn_in_train_id - 1]
        nn_in_train_subimg = utils.get_rect_from_bbox(nn_in_train_img, nn_in_train_bbox)
        nn_in_train_shape = nn_in_train_subimg.shape

        # transfer the rect from neighbor to query
        nn_in_train_rect.norm_for_bbox(nn_in_train_bbox)
        nn_in_train_rect.norm_for_size(nn_in_train_shape)
        nn_in_train_rect.denorm_for_size(query_img_shape)
        nn_in_train_rect.denorm_for_bbox(query_bbox)

        # set some debugging information
        nn_in_train_rect.info = "Transfered using nonparametricRG from imgid:%d" % nn_in_train_id

        return nn_in_train_rect
