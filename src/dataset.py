import os
import abc
from pascal_utils import VOC2006AnnotationParser, all_classes


class Dataset(object):

    def __init__(self, base_path):
        self.base_path = base_path

    @abc.abstractmethod
    def get_train(self):
        """ return a generator object that yields dictionares """

    @abc.abstractmethod
    def get_test(self):
        """ return a generator object that yields dictionares """


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
