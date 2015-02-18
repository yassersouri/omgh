import numpy as np
import scipy.io
import cv2
from parts import *
import sys
import os
import settings
import sklearn.neighbors
import skimage.feature
import skimage.color
import skimage.transform
import utils
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe


def thresh_segment(seg, thresh):
    return seg >= thresh


def thresh_segment_max(seg):
    return thresh_segment(seg, np.max(seg))


def thresh_segment_mean(seg):
    return thresh_segment(seg, np.mean(seg))


def gen_part_points(part_rect, seg, N=10):
    xmin, xmax, ymin, ymax = part_rect

    xs = np.random.uniform(low=xmin+1, high=xmax-1, size=N)
    ys = np.random.uniform(low=ymin+1, high=ymax-1, size=N)

    parts = Parts()

    for x, y in zip(xs, ys):
        if seg[x, y, 0]:
            parts.append(Part(-1, '?', -1, int(round(y)), int(round(x)), 1))
    return parts


def gen_bg_points(part_rect, seg, N=100):
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

    @staticmethod
    def get_bvlc_net(test_phase=True, gpu_mode=True):
        net = caffe.Classifier(settings.DEFAULT_MODEL_FILE, settings.DEFAULT_PRETRAINED_FILE, mean=np.load(settings.ILSVRC_MEAN), channel_swap=(2, 1, 0), raw_scale=255)
        if test_phase:
            net.set_phase_test()
        if gpu_mode:
            net.set_mode_gpu()

        return net

    @staticmethod
    def get_custom_net(model_def, pretrained_file, test_phase=True, gpu_mode=True):
        net = caffe.Classifier(model_def, pretrained_file, mean=np.load(settings.ILSVRC_MEAN),  channel_swap=(2, 1, 0), raw_scale=255)
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

        self.ffeats = np.concatenate([self.feats[k] for k in self.layers], axis=2)

    def features(self, points, layers=None):
        n_points = len(points)
        if layers is None:
            layers = self.layers
        n_features = sum(self.num_feats[l] for l in layers)
        features = np.zeros((n_points, n_features))

        for i, point in enumerate(points):
            x, y = point.y - 1, point.x - 1  # not because I'm idoit, but because of other things!
            # feat_layers = [self.feats[l][x, y, :] for l in layers]
            features[i, :] = self.ffeats[x, y, :]

        return features

    def part_for_image(self, all_image_infos, all_segmentaion_infos, cub_parts, img_id, part_filter_names, N_part=10, N_bg=100):
        img = caffe.io.load_image(all_image_infos[img_id])
        seg = thresh_segment_mean(caffe.io.load_image(all_segmentaion_infos[img_id]))

        self.init_with_image(img)

        parts = cub_parts.for_image(img_id)
        part_parts = parts.filter_by_name(part_filter_names)
        part_positive = gen_part_points(part_parts.get_rect_info(img.shape), seg, N_part)
        part_negative = gen_bg_points(part_parts.get_rect_info(img.shape), seg, N_bg)

        part_positive.norm_for_size(img.shape[1], img.shape[0], self.input_dim)
        part_negative.norm_for_size(img.shape[1], img.shape[0], self.input_dim)

        feats_positive = self.features(part_positive)
        feats_negative = self.features(part_negative)

        return feats_positive, feats_negative

    def part_for_image_local(self, all_image_infos, all_segmentaion_infos, bah, img_id, part_name, N_part, N_bg):
        img = caffe.io.load_image(all_image_infos[img_id])
        seg = thresh_segment_mean(caffe.io.load_image(all_segmentaion_infos[img_id]))

        self.init_with_image(img)

        part_rect_info = bah.get_berkeley_annotation(img_id, part_name)
        part_positive = gen_part_points(part_rect_info, seg, N_part)
        part_negative = gen_bg_points(part_rect_info, seg, N_bg)

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

    def part_features_for_local_rf(self, all_image_infos, all_segmentaion_infos, bah, IDs, part_name, N_part=10, N_bg=100):
        positives = []
        negatives = []

        for i, img_id in enumerate(IDs):
            feats_positive, feats_negative = self.part_for_image_local(all_image_infos, all_segmentaion_infos, bah, img_id, part_name, N_part, N_bg)

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


class SSFeatureLoader(object):
    """
    Search space feature loaders.

    This will help NNFinder to find the nearest neighbors in the space
    provided by this feature loader
    """
    instance_split = 10

    def __init__(self, ss_storage):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def load_all(self):
        return self.instance

    def load_train(self):
        # FIXME: this will only work for the CUB dataset
        return self.instance[self.IDtrain - 1, :]

    def load_test(self):
        # FIXME this will only work for the CUB dataset
        return self.instance[self.IDtest - 1, :]

    def load_one(self, img):
        raise NotImplementedError


class DeepSSFeatureLoader(SSFeatureLoader):
    CAFFENET_LAYER_DIM = {
        'fc7': 4096,
        'fc6': 4096,
        'pool5': 9216,
        'conv5': 43264,
        'conv4': 64896,
        'conv3': 64896
    }

    def __init__(self, dataset, ss_storage, net=None, net_name=None, layer_name='pool5', crop_index=0):
        self.dataset = dataset
        self.ss_storage = ss_storage
        if net is None:
            self.net = DeepHelper.get_bvlc_net()
        else:
            self.net = net

        self.net_name = net_name
        self.layer_name = layer_name
        self.crop_index = crop_index

    def get_name(self):
        return 'deeploader(net:%s, layer:%s)' % (self.net_name, self.layer_name)

    def setup(self):
        self.IDtrain, self.IDtest = self.dataset.get_train_test_id()
        self.dataset_size = sum(1 for _ in self.dataset.get_all_images())
        self.ss_storage.super_name = 'ss_features'
        self.ss_storage.sub_name = self.net_name
        self.ss_storage.instance_path = self.ss_storage.get_instance_path(self.ss_storage.super_name, self.ss_storage.sub_name, 'feat_cache_%s' % self.layer_name)

        # load the instance if exists
        if self.ss_storage.check_exists_large(self.ss_storage.instance_path):
            self.instance = self.ss_storage.load_large_instance(self.ss_storage.instance_path, self.instance_split)

        # calculate the instance
        else:
            self.instance = self._calculate()
            self.ss_storage.save_large_instance(self.ss_storage.instance_path, self.instance, self.instance_split)

    def _calculate(self):
        instance = np.zeros((self.dataset_size, self.CAFFENET_LAYER_DIM[self.layer_name]))
        for i, info in enumerate(self.dataset.get_all_images(cropped=True)):
            img = caffe.io.load_image(info['img_file'])
            self.net.predict([img], oversample=False)
            instance[i, :] = self.net.blobs[self.layer_name].data[self.crop_index].flatten()
        return instance


class HOGSSFeatureLoader(SSFeatureLoader):
    HOG_RESIZE = (227, 227)
    HOG_DIM = 26244

    def __init__(self, dataset, ss_storage):
        self.dataset = dataset
        self.ss_storage = ss_storage

    def get_name(self):
        return 'hogloader'

    def setup(self):
        self.IDtrain, self.IDtest = self.dataset.get_train_test_id()
        self.dataset_size = sum(1 for _ in self.dataset.get_all_images())
        self.ss_storage.super_name = 'ss_features'
        self.ss_storage.sub_name = 'hog'
        self.ss_storage.instance_path = self.ss_storage.get_instance_path(self.ss_storage.super_name, self.ss_storage.sub_name, 'hog')

        # load the instance if exists
        if self.ss_storage.check_exists_large(self.ss_storage.instance_path):
            self.instance = self.ss_storage.load_large_instance(self.ss_storage.instance_path, self.instance_split)

        # calculate the instance
        else:
            self.instance = self._calculate()
            self.ss_storage.save_large_instance(self.ss_storage.instance_path, self.instance, self.instance_split)

    def _calculate(self):
        instance = np.zeros((self.dataset_size, self.HOG_DIM))
        for i, info in enumerate(self.dataset.get_all_images(cropped=True)):
            img = caffe.io.load_image(info['img_file'])
            img_g = skimage.color.rgb2gray(img)
            img_r = skimage.transform.resize(img_g, self.HOG_RESIZE)

            instance[i, :] = skimage.feature.hog(img_r, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        return instance


class GISTFeatureLoader(SSFeatureLoader):
    pass


class NNFinder(object):

    def __init__(self, final_storage, ssfeature_loader, dataset, normalize=True):
        self.final_storage = final_storage
        self.ssfeature_loader = ssfeature_loader
        self.feature_loader_name = ssfeature_loader.get_name()
        self.normalize = normalize
        self.dataset = dataset

    def setup(self):
        self.final_storage.super_name = 'NNS'
        self.final_storage.sub_name = self.feature_loader_name
        self.final_storage.instance_path = self.final_storage.get_instance_path(self.final_storage.super_name, self.final_storage.sub_name, '%s.mat' % self.normalize)
        self._pre_calculate()
        self.IDtrain, self.IDtest = self.dataset.get_train_test_id()

    def _pre_calculate(self):
        if self.final_storage.check_exists(self.final_storage.instance_path):
            self.NNS = self.final_storage.load_instance(self.final_storage.instance_path)
        else:
            self.ssfeature_loader.setup()
            self.Xtrain = self.ssfeature_loader.load_train()
            self.Xtest = self.ssfeature_loader.load_test()
            if self.normalize:
                self.Xtrain = utils.l2_feat_norm(self.Xtrain)
                self.Xtest = utils.l2_feat_norm(self.Xtest)

            nn_model = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=2)
            nn_model.fit(self.Xtrain)
            self.NNS = nn_model.kneighbors(self.Xtest, 1, return_distance=False)
            self.final_storage.save_instance(self.final_storage.instance_path, self.NNS)

        # this needs change for larges n_neighbors
        self.NNS = self.NNS.T[0]

    def find_in_train(self, img_id):
        # what is the test index of this img_id?
        try:
            test_index = np.argwhere(self.IDtest == img_id)[0][0]
        except IndexError:
            raise IndexError('img_id is not in test set!')
        nn_id = self.IDtrain[self.NNS[test_index]]
        return nn_id
