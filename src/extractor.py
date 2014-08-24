import cv2
import abc
import os


class BaseExtractor(object):
    def __init__(self, storage):
        self.STORAGE_SUPER_NAME = 'features'
        self.FILE_NAMES_EXT = 'mat'
        self.storage = storage
        self.super_folder = self.storage.get_super_folder(self.STORAGE_SUPER_NAME)
        self.storage.ensure_dir(self.super_folder)

    @abc.abstractmethod
    def extract(self):
        """Extracts Features From Images"""


class SIFT_SIFT_Extractor(BaseExtractor):
    def __init__(self, storage):
        super(SIFT_SIFT_Extractor, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'sift_sift'
        self.sub_folder = self.storage.get_sub_folder(self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)

        self._keypoint_detector = cv2.FeatureDetector_create("SIFT")
        self._keypoint_extractor = cv2.DescriptorExtractor_create("SIFT")

    def extract(self, dataset, kind, force=False):
        assert kind in ['train', 'test']

        if kind == 'train':
            data = dataset.get_train()
        elif kind == 'test':
            data = dataset.get_test()

        for t in data:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
            if (not self.storage.check_exists(instance_path)) or force:
                img = cv2.imread(t['img_file'])
                kp = self._keypoint_detector.detect(img, None)
                kp, des = self._keypoint_extractor.compute(img, kp)
                self.storage.save_instance(instance_path, des)
            else:
                des = self.storage.load_instance(instance_path)

            yield t, des
