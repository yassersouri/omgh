from extractor import BaseExtractor
import numpy as np
import sys
sys.path.append('/home/ipl/installs/caffe-rc/python/')
import caffe
import settings


class CNN_Features_CAFFE_REFERENCE(BaseExtractor):

    def __init__(self, storage, model_file=settings.DEFAULT_MODEL_FILE, pretrained_file=settings.DEFAULT_PRETRAINED_FILE, image_mean=settings.ILSVRC_MEAN, full=False, make_net=True, feature_layer='fc7', crop_index=4, xDim=4096):
        super(CNN_Features_CAFFE_REFERENCE, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'cnn_feature_caffe_reference'
        self.full = full
        if self.full:
            self.STORAGE_SUB_NAME = 'cnn_feature_caffe_reference_full'
        self.feature_layer = feature_layer
        self.feature_crop_index = crop_index
        self.xDim = xDim
        if self.full:
            self.full_length = settings.FULL_LENGTH

        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)

        self.model_file = model_file
        self.pretrained_file = pretrained_file
        self.image_mean = image_mean

        if make_net:
            self.net = caffe.Classifier(self.model_file,
                                        self.pretrained_file,
                                        mean=np.load(self.image_mean),
                                        channel_swap=(2, 1, 0),
                                        raw_scale=255)
            if self.full:
                self.net = caffe.Classifier(self.model_file,
                                            self.pretrained_file,
                                            mean=np.load(self.image_mean),
                                            channel_swap=(2, 1, 0),
                                            raw_scale=255,
                                            image_dims=(256, 256))
            self.net.set_mode_gpu()

    def extract_all(self, data_generator, flip=False, crop=False, force=False, bbox=None):
        for t in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
            if force or not self.storage.check_exists(instance_path):
                try:
                    im = caffe.io.load_image(t['img_file'])
                    if crop:
                        assert bbox is not None
                        # TODO: move to sepatate funciton
                        x, y, w, h = bbox[int(t['img_id']) - 1]
                        im = im[y:y+h, x:x+w]

                    if flip:
                        im = np.fliplr(im)
                    self.net.predict([im])

                    if self.full:
                        des = self.net.blobs[self.feature_layer].data[:, :, 0, 0]
                    else:
                        des = self.net.blobs[self.feature_layer].data[
                            self.feature_crop_index][:, 0, 0]
                except IOError:
                    # here for the missing images we put zero in their place.
                    # it is not obvious that this is the best thing to do.
                    des = self.dummy_extract_one(t['img_id'])

                self.storage.save_instance(instance_path, des)
            else:
                des = self.storage.load_instance(instance_path)
                if self.full:
                    pass
                else:
                    des = des.flatten()

            yield t, des

    def extract_one(self, img_id):
        instance_name = "%s.%s" % (img_id, self.FILE_NAMES_EXT)
        instance_path = self.storage.get_instance_path(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
        if not self.storage.check_exists(instance_path):
            # TODO: fix this
            # P.S: I don't think I will fix this!
            raise Exception("Calculate deep features first then load them.")
        else:
            des = self.storage.load_instance(instance_path)
            if self.full:
                pass
            else:
                des = des.flatten()
            return des

    def dummy_extract_one(self, img_id, xDim=None):
        if xDim is None:
            xDim = self.xDim
        return np.zeros(xDim)


class Berkeley_Extractor(CNN_Features_CAFFE_REFERENCE):

    def __init__(self, storage, model_file=settings.BERKELEY_MODEL_FILE, pretrained_file=settings.BERKELEY_CROP_PRET, image_mean=settings.ILSVRC_MEAN, make_net=True, xDim=4096):
        super(CNN_Features_CAFFE_REFERENCE, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'cnn_feature_berkeley'
        self.feature_layer = 'fc7'
        self.feature_crop_index = 0
        self.xDim = xDim

        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)

        self.model_file = model_file
        self.pretrained_file = pretrained_file
        self.image_mean = image_mean
        self.full = False

        if make_net:
            self.net = caffe.Classifier(self.model_file,
                                        self.pretrained_file,
                                        mean=np.load(self.image_mean),
                                        channel_swap=(2, 1, 0),
                                        raw_scale=255, gpu=True)

    def extract_all(self, data_generator, force=False):
        for t in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
            if force or not self.storage.check_exists(instance_path):
                try:
                    im = caffe.io.load_image(t['img_file'])
                    des = self.net.predict([im], oversample=False).flatten()
                except IOError:
                    # here for the missing images we put zero in their place.
                    # it is not obvious that this is the best thing to do.
                    des = self.dummy_extract_one(t['img_id'])
                self.storage.save_instance(instance_path, des)
            else:
                des = self.storage.load_instance(instance_path).flatten()

            yield t, des
