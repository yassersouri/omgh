from extractor import BaseExtractor
import numpy as np
import sys
sys.path.append('/home/ipl/installs/caffe-rc/python/')
import caffe
import settings


class CNN_Features_CAFFE_REFERENCE(BaseExtractor):

    def __init__(self, storage, model_file=settings.MODEL_FILE, pretrained_file=settings.PRETRAINED_FILE, image_mean=settings.ILSVRC_MEAN):
        super(CNN_Features_CAFFE_REFERENCE, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'cnn_feature_caffe_reference'
        self.feature_layer = 'fc7'
        self.center_crop_index = 4

        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)

        self.model_file = model_file
        self.pretrained_file = pretrained_file
        self.image_mean = image_mean

        self.net = caffe.Classifier(self.model_file,
                                    self.pretrained_file,
                                    mean=np.load(self.image_mean),
                                    channel_swap=(2, 1, 0),
                                    raw_scale=255)  # ,
                                    # image_dims=(256, 256))
        self.net.set_mode_gpu()

    def extract_all(self, data_generator, flip=False, crop=False, force=False, bbox=None):
        for t in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
            if force or not self.storage.check_exists(instance_path):
                im = caffe.io.load_image(t['img_file'])
                if crop:
                    assert bbox is not None
                    # TODO: move to sepatate funciton
                    x, y, w, h = bbox[int(t['img_id']) - 1]
                    im = im[y:y+h, :x+w]

                if flip:
                    im = np.fliplr(im)
                self.net.predict([im])

                des = self.net.blobs[self.feature_layer].data[
                    self.center_crop_index][:, 0, 0]

                self.storage.save_instance(instance_path, des)
            else:
                des = self.storage.load_instance(instance_path)
                if len(des.shape) > 1:
                    des = des[0, :]

            yield t, des

    def extract_one(self, img_id):
        instance_name = "%s.%s" % (img_id, self.FILE_NAMES_EXT)
        instance_path = self.storage.get_instance_path(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
        if not self.storage.check_exists(instance_path):
            # TODO: fix this
            raise Exception("Calculate deep features first then load them.")
        else:
            des = self.storage.load_instance(instance_path)
            if len(des.shape) > 1:
                des = des[0, :]
            return des
