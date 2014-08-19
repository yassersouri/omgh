import os


class Dataset(object):
    def __init__(self, base_path):
        self.base_path = base_path


class PASCAL_VOC_2006(Dataset):
    ANNOTATIONS_FOLDER_NAME = 'Annotations'
    SETS_FOLDER_NAME = 'ImageSets'
    IMAGES_FOLDER_NAME = 'PNGImages'
    CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'cat', 'cow', 'dog', 'horse', 'sheep', 'person']
    SETS_FILE_EXT = 'txt'
    SETS_NAME = ['train', 'test', 'val', 'trainval']

    def __init__(self, base_path):
        super(PASCAL_VOC_2006, self).__init__(base_path)
        self.annotations = os.path.join(self.base_path, self.ANNOTATIONS_FOLDER_NAME)
        self.sets = os.path.join(self.base_path, self.SETS_FOLDER_NAME)
        self.images = os.path.join(self.base_path, self.IMAGES_FOLDER_NAME)

    def classes(self):
        return self.CLASSES

    def sets(self, kind, objectClass=None):
        """
        This function returns a generator object.
        `kind` must be one of: ['train', 'test', 'val', 'trainval']
        """
        assert kind in self.SETS_NAME
        if objectClass is not None:
            assert objectClass in self.CLASSES
            set_file_name = "%s_%s.%s" % (objectClass, kind, self.SETS_FILE_EXT)
        else:
            set_file_name = "%s.%s" % (kind, self.SETS_FILE_EXT)

        set_file_path = os.path.join(self.sets, set_file_name)
