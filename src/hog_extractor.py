from extractor import BaseExtractor
import numpy as np
import cv2

class HOG(BaseExtractor):

    def __init__(self, storage):
        super(HOG, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'hog_normalized'

        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)

        self.hog = cv2.HOGDescriptor()
        self.base_size = 256

    def extract(self, data_generator, bbox, force=False):
        for t in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
            if force or not self.storage.check_exists(instance_path):
                img = cv2.imread(t['img_file'])

                # crop
                x,y,w,h = bbox[int(t['img_id']) - 1]
                img_c = img[y:y+h, x:x+w]
                img_r = cv2.resize(img_c, (self.base_size, self.base_size))

                # compute hog
                des = self.hog.compute(img_r)

                # normalize
                des = des / np.linalg.norm(des)
                
                self.storage.save_instance(instance_path, des)
            else:
                des = self.storage.load_instance(instance_path)
                if len(des.shape) > 1:
                    des = des[0, :]

            yield t, des