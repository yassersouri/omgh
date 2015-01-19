import numpy as np
import cv2
from parts import *
import sys
import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe


def thresh_segment(seg, thresh):
    return seg >= thresh


def thresh_segment_max(seg):
    return thresh_segment(seg, np.max(seg))


def thresh_segment_mean(seg):
    return thresh_segment(seg, np.mean(seg))


def gen_part_points(part_rect, seg, parts):
    N = 100
    xmin, xmax, ymin, ymax = part_rect

    xs = np.random.uniform(low=xmin, high=xmax, size=N)
    ys = np.random.uniform(low=ymin, high=ymax, size=N)

    parts = Parts()

    for x, y in zip(xs, ys):
        if seg[x, y, 0]:
            parts.append(Part(-1, '?', -1, int(round(y)), int(round(x)), 1))
    print len(parts)
    return parts


def gen_bg_points(part_rect, seg, parts):
    N = 500
    h, w = seg.shape[0], seg.shape[1]
    xmin, xmax, ymin, ymax = part_rect

    xs = np.random.uniform(low=0, high=h-1, size=N)
    ys = np.random.uniform(low=0, high=w-1, size=N)

    parts = Parts()

    for x, y in zip(xs, ys):
        if (xmin <= x <= xmax and ymin <= y <= ymax):
            if not seg[x-2:x+3, y-2:y+3, 0].sum() and False:  # is this really necessary!?
                parts.append(Part(-1, '?', -1, int(round(y)), int(round(x)), 1))
        else:
            parts.append(Part(-1, '?', -1, int((round(y))), int(round(x)), 1))

    print len(parts)
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
