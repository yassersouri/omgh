from extractor import BaseExtractor
import numpy as np
import sys
sys.path.append('/home/ipl/installs/caffe-rc/python/')
import caffe


class CNN_Features_CAFFE_REFERENCE(BaseExtractor):

    def __init__(self, storage, model_file, pretrained_file, image_mean):
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
                                    channel_swap=(2,1,0),
                                    raw_scale=255,
                                    image_dims=(256, 256))
        self.net.set_mode_gpu()

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


    def extract_cropped(self, data_generator, bbox, force=False):
        for t in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
            if force or not self.storage.check_exists(instance_path):
                #TODO: move to sepatate funciton
                im_cropped = caffe.io.load_image(t['img_file'])
                x,y,w,h = bbox[int(t['img_id'])-1]
                im_cropped = im_cropped[y:y+h,x:x+w]

                self.net.predict([im_cropped])

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
            raise Exception("Calculate deep features first then load them.")
        else:
            des = self.storage.load_instance(instance_path)
            if len(des.shape) > 1:
                des = des[0, :]
            return des
