import numpy as np
import scipy.io
import cv2
from parts import *
import sys
import os
import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe


def thresh_segment(seg, thresh):
    return seg >= thresh


def thresh_segment_max(seg):
    return thresh_segment(seg, np.max(seg))


def thresh_segment_mean(seg):
    return thresh_segment(seg, np.mean(seg))


def gen_part_points(part_rect, seg, parts, N=10):
    xmin, xmax, ymin, ymax = part_rect

    xs = np.random.uniform(low=xmin+1, high=xmax-1, size=N)
    ys = np.random.uniform(low=ymin+1, high=ymax-1, size=N)

    parts = Parts()

    for x, y in zip(xs, ys):
        if seg[x, y, 0]:
            parts.append(Part(-1, '?', -1, int(round(y)), int(round(x)), 1))
    return parts


def gen_bg_points(part_rect, seg, parts, N=100):
    h, w = seg.shape[0], seg.shape[1]
    xmin, xmax, ymin, ymax = part_rect

    xs = np.random.uniform(low=1, high=h, size=N)
    ys = np.random.uniform(low=1, high=w, size=N)

    parts = Parts()

    for x, y in zip(xs, ys):
        if (xmin <= x <= xmax and ymin <= y <= ymax):
            if not seg[x-2:x+3, y-2:y+3, 0].sum() and False:  # is this really necessary!?
                parts.append(Part(-1, '?', -1, int(round(y)), int(round(x)), 1))
        else:
            parts.append(Part(-1, '?', -1, int((round(y))), int(round(x)), 1))
    return parts


class DeepHelper(object):

    @classmethod
    def get_bvlc_net(test_phase=True, gpu_mode=True):
        net = caffe.Classifier(settings.DEFAULT_MODEL_FILE, settings.DEFAULT_PRETRAINED_FILE, mean=np.load(settings.ILSVRC_MEAN), channel_swap=(2, 1, 0), raw_scale=255)
        if test_phase:
            net.set_phase_test()
        if gpu_mode:
            net.set_mode_gpu()

        return net

    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    feats = {}
    num_feats = {}
    crop_dim = 0
    input_dim = 227

    def __init__(self, net=None):
        if net is None:
            self.net = self.get_bvlc_net()
        else:
            self.net = net

    def init_with_image(self, img):
        self.net.predict([img], oversample=False)
        self._make_features_ready()

    def _make_features_ready(self):
        for layer in self.layers:
            data = self.net.blobs[layer].data[self.crop_dim]

            data = data.swapaxes(0, 2)
            data = data.swapaxes(0, 1)
            data = cv2.resize(data, (self.input_dim, self.input_dim), interpolation=cv2.INTER_LINEAR)

            _, _, num_feat = data.shape

            self.num_feats[layer] = num_feat
            self.feats[layer] = data

    def features(self, points, layers=None):
        n_points = len(points)
        if layers is None:
            layers = self.layers
        n_features = sum(self.num_feats[l] for l in layers)
        features = np.zeros((n_points, n_features))

        for i, point in enumerate(points):
            x, y = point.y - 1, point.x - 1  # not because I'm idoit, but because of other things!
            feat_layers = [self.feats[l][x, y, :] for l in layers]
            features[i, :] = np.concatenate(feat_layers)

        return features

    def part_for_image(self, all_image_infos, all_segmentaion_infos, cub_parts, img_id, part_filter_names, N_part=10, N_bg=100):
        img = caffe.io.load_image(all_image_infos[img_id])
        seg = thresh_segment_mean(caffe.io.load_image(all_segmentaion_infos[img_id]))

        self.init_with_image(img)

        parts = cub_parts.for_image(img_id)
        part_parts = parts.filter_by_name(part_filter_names)
        part_positive = gen_part_points(part_parts.get_rect_info(img), seg, part_parts, N_part)
        part_negative = gen_bg_points(part_parts.get_rect_info(img), seg, part_parts, N_bg)

        part_positive.norm_for_size(img.shape[1], img.shape[0], self.input_dim)
        part_negative.norm_for_size(img.shape[1], img.shape[0], self.input_dim)

        feats_positive = self.features(part_positive)
        feats_negative = self.features(part_negative)

        return feats_positive, feats_negative

    def part_features_for_rf(self, all_image_infos, all_segmentaion_infos, cub_parts, IDs, part_filter_names, N_part=10, N_bg=100):
        positives = []
        negatives = []
        for i, img_id in enumerate(IDs):
            feats_positive, feats_negative = self.part_for_image(all_image_infos, all_segmentaion_infos, cub_parts, img_id, part_filter_names, N_part, N_bg)

            positives.append(feats_positive)
            negatives.append(feats_negative)
        X_pos = np.vstack(positives)
        y_pos = np.ones((X_pos.shape[0]), np.int)
        X_neg = np.vstack(negatives)
        y_neg = np.zeros((X_neg.shape[0]), np.int)

        X = np.vstack((X_pos, X_neg))
        y = np.concatenate((y_pos, y_neg))

        return X, y


class BerkeleyAnnotationsHelper(object):
    train_file_name = 'bird_train.mat'
    test_file_name = 'bird_test.mat'

    def __init__(self, base_path, IDtrain, IDtest):
        self.base_path = base_path
        self.IDtrain = IDtrain
        self.IDtest = IDtest

        self.train_path = os.path.join(self.base_path, self.train_file_name)
        self.test_path = os.path.join(self.base_path, self.test_file_name)

        b_train_anno = scipy.io.loadmat(self.train_path)
        self.b_train_anno = b_train_anno['data']

        b_test_anno = scipy.io.loadmat(self.test_path)
        self.b_test_anno = b_test_anno['data']

    def get_train_berkeley_annotation(self, train_id, name):
        p = 0
        if name == 'head':
            p = 1
        elif name == 'body':
            p = 2
        elif name == 'bbox':
            p = 3
        res = self.b_train_anno[0, train_id][p][0]
        ymin, xmin, ymax, xmax = res[0], res[1], res[2], res[3]

        return xmin, xmax, ymin, ymax

    def get_test_berkeley_annotation(self, test_id, name):
        p = 0
        if name == 'bbox':
            p = 1
        elif name == 'head':
            p = 2
        elif name == 'body':
            p = 3
        res = self.b_test_anno[0, test_id][p][0]
        ymin, xmin, ymax, xmax = res[0], res[1], res[2], res[3]

        return xmin, xmax, ymin, ymax

    def get_berkeley_annotation(self, img_id, name):
        train_where = np.argwhere(self.IDtrain == img_id)
        test_where = np.argwhere(self.IDtest == img_id)
        if train_where.shape[0] == 1:
            return self.get_train_berkeley_annotation(train_where[0, 0], name)
        elif test_where.shape[0] == 1:
            return self.get_test_berkeley_annotation(test_where[0, 0], name)
        else:
            raise Exception('Not found!')
