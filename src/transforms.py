import abc
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
import numpy as np
import joblib
from datetime import datetime as dt


class Transform(object):

    def __init__(self, storage):
        self.STORAGE_SUPER_NAME = 'transforms'
        self.FILE_NAMES_EXT = 'mat'
        self.MODEL_NAME_EXT = 'pkl'
        self.storage = storage
        self.super_folder = self.storage.get_super_folder(
            self.STORAGE_SUPER_NAME)
        self.storage.ensure_dir(self.super_folder)

    @abc.abstractmethod
    def fit(self):
        """ do preperation """

    @abc.abstractmethod
    def transform(self):
        """ transform the data to the new domain """


class PCA_Transform(Transform):

    def __init__(self, storage, n_components=50):
        super(PCA_Transform, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'pca_%d' % n_components
        self.STORAGE_MODEL_NAME = 'pca_model_%s' % n_components
        self.MODEL_NAME = '%s.%s' % (
            self.STORAGE_MODEL_NAME, self.MODEL_NAME_EXT)
        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)
        self.model_path = self.storage.get_model_path(
            self.STORAGE_SUPER_NAME, self.MODEL_NAME)
        self._transform = None
        self.n_components = n_components

    def fit(self, data_generator, force=False):
        if force or not self.storage.check_exists(self.model_path):
            self._transform = PCA(n_components=self.n_components)

            def mid_generator():
                for t, des in data_generator:
                    yield des

            X = np.vstack(mid_generator())
            self._transform.fit(X)
            joblib.dump(self._transform, self.model_path)
        else:
            self._transform = joblib.load(self.model_path)

    def transform(self, data_generator, force=False):
        """
        returns a generator.
        """
        for t, des in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)

            if force or not self.storage.check_exists(instance_path):
                result = self._transform.transform(des)
                self.storage.save_instance(instance_path, result)
            else:
                result = self.storage.load_instance(instance_path)

            yield t, result


class GMMUniversalVocabulary(Transform):

    def __init__(self, storage, n_components=256, covariance_type='diag',
                 n_iter=50, n_init=5):
        assert covariance_type in ['spherical', 'tied', 'diag', 'full']

        super(GMMUniversalVocabulary, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'gmm_universal_vocab_%d_%s_%d_%d' % (
            n_components, covariance_type, n_iter, n_init)
        self.STORAGE_MODEL_NAME = 'gmm_universal_%d_%s_%d_%d' % (
            n_components, covariance_type, n_iter, n_init)
        self.MODEL_NAME = '%s.%s' % (
            self.STORAGE_MODEL_NAME, self.MODEL_NAME_EXT)
        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)
        self.model_path = self.storage.get_model_path(
            self.STORAGE_SUPER_NAME, self.MODEL_NAME)
        self._transform = None
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.n_init = n_init

    def fit(self, data_generator, force=False, test=False):
        if force or not self.storage.check_exists(self.model_path):
            self._transform = GMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter, n_init=self.n_init)

            def mid_generator():
                for t, des in data_generator:
                    yield des

            X = np.vstack(mid_generator())
            if test:
                X = X[0:100000, :]
            print X.shape
            a = dt.now()
            self._transform.fit(X)
            b = dt.now()
            print 'fitting gmm: \t', (b - a)
            joblib.dump(self._transform, self.model_path)
        else:
            print 'loaded'
            self._transform = joblib.load(self.model_path)

    def transform(self, data_generator, force=False):
        """
        returns a generator.
        """
        for t, des in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)

            if force or not self.storage.check_exists(instance_path):
                result = self._transform.predict(des)
                hist = np.bincount(result)
                # l2 normalize the histogram
                hist = hist / np.linalg.norm(hist)
                self.storage.save_instance(instance_path, hist)
            else:
                hist = self.storage.load_instance(instance_path)
                # the following line is the result of scipy loading what
                # it saved differently!
                hist = hist[0, :]

            yield t, hist
