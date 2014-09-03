from extractor import BaseExtractor
import numpy as np
import sys
sys.path.append('/home/yasser/installs/caffe-0.9999/python/')
import caffe


class CNN_Features_CAFFE_REFERENCE(BaseExtractor):

    def __init__(self, storage, model_file, pretrained_file, ilsvrc_mean):
        super(CNN_Features_CAFFE_REFERENCE, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'cnn_feature_caffe_reference'
        self.feature_layer = 'fc7'
        self.center_crop_index = 4

        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)

        self.model_file = model_file
        self.pretrained_file = pretrained_file
        self.ilsvrc_mean = ilsvrc_mean

        self.net = caffe.Classifier(self.model_file,
                                    self.pretrained_file)
        self.net.set_phase_test()
        self.net.set_mode_cpu()
        self.net.set_mean('data', np.load(self.ilsvrc_mean))
        self.net.set_raw_scale('data', 255)
        self.net.set_channel_swap('data', (2, 1, 0))

    def extract(self, data_generator, force=False):
        for t in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
            if force or not self.storage.check_exists(instance_path):
                self.net.predict([caffe.io.load_image(t['img_file'])])

                des = self.net.blobs[self.feature_layer].data[
                    self.center_crop_index][:, 0, 0]

                self.storage.save_instance(instance_path, des)
            else:
                des = self.storage.load_instance(instance_path)
                if len(des.shape) > 1:
                    des = des[0, :]

            yield t, des

