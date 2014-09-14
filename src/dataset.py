import os
import abc
from pascal_utils import VOC2006AnnotationParser, all_classes
import numpy as np


class Dataset(object):

    def __init__(self, base_path):
        self.base_path = base_path

    @abc.abstractmethod
    def get_train(self):
        """ return a generator object that yields dictionares """

    @abc.abstractmethod
    def get_test(self):
        """ return a generator object that yields dictionares """


class CUB_200_2011(Dataset):
    NAME = 'CUB_200_2011'
    IMAGES_FOLDER_NAME = 'images'
    IMAGES_FILE_NAME = 'images.txt'
    TRAIN_TEST_SPLIT_FILE_NAME = 'train_test_split.txt'
    CLASS_LABEL_FILE_NAME = 'image_class_labels.txt'
    BBOX_FILE_NAME = 'bounding_boxes.txt'
    SPLIT_FILE_TRAIN_INDICATOR = '1'
    SPLIT_FILE_TEST_INDICATOR = '0'

    def __init__(self, base_path):
        super(CUB_200_2011, self).__init__(base_path)
        self.images_folder = os.path.join(
            self.base_path, self.IMAGES_FOLDER_NAME)
        self.images_file = os.path.join(
            self.base_path, self.IMAGES_FILE_NAME)
        self.train_test_split_file = os.path.join(
            self.base_path, self.TRAIN_TEST_SPLIT_FILE_NAME)
        self.class_label_file = os.path.join(
            self.base_path, self.CLASS_LABEL_FILE_NAME)
        self.bbox_file = os.path.join(
            self.base_path, self.BBOX_FILE_NAME)

    def get_all_images(self):
        with open(self.images_file, 'r') as images_file:
            for line in images_file:
                parts = line.split()
                assert len(parts) == 2
                yield {'img_id': parts[0],
                       'img_file': os.path.join(self.images_folder, parts[1])}

    def get_train_test(self, read_extractor, xDim=4096):
        trains = []
        tests = []
        indicators = []
        with open(self.train_test_split_file, 'r') as split_file:
            for line in split_file:
                parts = line.split()
                assert len(parts) == 2
                img_id = parts[0]
                indicator = parts[1]
                indicators.append(indicator)
                if indicator == self.SPLIT_FILE_TRAIN_INDICATOR:
                    trains.append(img_id)
                elif indicator == self.SPLIT_FILE_TEST_INDICATOR:
                    tests.append(img_id)
                else:
                    raise Exception("Unknown indicator, %s" % indicator)

        Xtrain = np.zeros((len(trains), xDim), dtype=np.float32)
        ytrain = np.zeros((len(trains)), dtype=np.int)
        Xtest = np.zeros((len(tests), xDim), dtype=np.float32)
        ytest = np.zeros((len(tests)), dtype=np.int)

        with open(self.class_label_file, 'r') as class_label:
            line_num = 0
            train_num = 0
            test_num = 0
            for line in class_label:
                parts = line.split()
                assert len(parts) == 2
                img_id = parts[0]
                img_cls = int(parts[1])
                indicator = indicators[line_num]
                if indicator == self.SPLIT_FILE_TRAIN_INDICATOR:
                    # training
                    Xtrain[train_num, :] = read_extractor(img_id)
                    ytrain[train_num] = img_cls
                    train_num += 1
                else:
                    # testing
                    Xtest[test_num, :] = read_extractor(img_id)
                    ytest[test_num] = img_cls
                    test_num += 1

                line_num += 1

        return Xtrain, ytrain, Xtest, ytest

    def get_bbox(self):
        bbox = np.genfromtxt(self.bbox_file, delimiter=' ')
        bbox = bbox[:, 1:]
        return bbox


class PASCAL_VOC_2006(Dataset):
    NAME = 'PASCAL_VOC_2006'
    ANNOTATIONS_FOLDER_NAME = 'Annotations'
    SETS_FOLDER_NAME = 'ImageSets'
    IMAGES_FOLDER_NAME = 'PNGImages'
    CLASSES = ['bicycle', 'bus', 'car', 'motorbike',
               'cat', 'cow', 'dog', 'horse', 'sheep', 'person']
    SETS_FILE_EXT = 'txt'
    ANNOTATIONS_FILE_EXT = 'txt'
    IMAGE_FILE_EXT = 'png'
    SETS_NAME = ['train', 'test', 'val', 'trainval']
    POSITIVE = '1'
    DIFFICULT = '0'
    NEGATIVE = '-1'

    def __init__(self, base_path):
        super(PASCAL_VOC_2006, self).__init__(base_path)
        self.annotations = os.path.join(
            self.base_path, self.ANNOTATIONS_FOLDER_NAME)
        self.sets = os.path.join(self.base_path, self.SETS_FOLDER_NAME)
        self.images = os.path.join(self.base_path, self.IMAGES_FOLDER_NAME)

    def classes(self):
        return self.CLASSES

    def get_train(self):
        return self.get_set('trainval', object_class=None,
                            difficult=True, trunc=True)

    def get_test(self):
        return self.get_set('test', object_class=None,
                            difficult=True, trunc=True)

    def get_set(self, kind, object_class=None, difficult=False, trunc=True):
        """
        This function returns a generator object.
        `kind` must be one of: ['train', 'test', 'val', 'trainval']
        """
        assert kind in self.SETS_NAME
        if object_class is not None:
            assert object_class in self.CLASSES
            set_file_name = "%s_%s.%s" % (
                object_class, kind, self.SETS_FILE_EXT)
        else:
            set_file_name = "%s.%s" % (kind, self.SETS_FILE_EXT)

        set_file_path = os.path.join(self.sets, set_file_name)

        return self._parse_set(set_file_path, difficult, trunc)

    def _parse_set(self, set_file_path, difficult, trunc):
        with open(set_file_path) as set_file:
            for line in set_file:
                parts = line.split()
                if len(parts) > 1:
                    image_id = parts[0]
                    is_here = True if parts[1] == self.POSITIVE or parts[
                        1] == self.DIFFICULT else False
                else:
                    image_id = parts[0]
                    is_here = True

                if is_here:
                    image_annotations_file = os.path.join(
                        self.annotations, "%s.%s" %
                        (image_id, self.ANNOTATIONS_FILE_EXT))
                    image_file = os.path.join(
                        self.images, "%s.%s" % (image_id, self.IMAGE_FILE_EXT))
                    with open(image_annotations_file, 'r') as content_file:
                        image_annotations_file_content = content_file.read()

                    annon_parser = VOC2006AnnotationParser(
                        image_annotations_file_content)
                    objects = annon_parser.get_objects()

                    if len(objects) == 0:
                        continue

                    all_classes_in_image = all_classes(objects)

                    yield {'img_id': image_id, 'img_file': image_file,
                           'classes': all_classes_in_image, 'objects': objects}
