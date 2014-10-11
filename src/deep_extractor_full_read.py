from extractor import BaseExtractor


class CNN_Features_CAFFE_REFERENCE(BaseExtractor):
    def __init__(self, storage):
        super(CNN_Features_CAFFE_REFERENCE, self).__init__(storage)
        self.STORAGE_SUB_NAME = 'cnn_feature_caffe_reference_full'
        self.feature_layer = 'fc7'
        self.center_crop_index = 4
        self.center_crop_index_mirror = 9

        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)
        self.storage.ensure_dir(self.sub_folder)

    # def extract(self, data_generator, force=False):
    #     for t in data_generator:
    #         instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
    #         instance_path = self.storage.get_instance_path(
    #             self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
    #         if force or not self.storage.check_exists(instance_path):
    #             raise Exception(
    #                 "Calculate deep features first then load them.")
    #         else:
    #             des = self.storage.load_instance_full(instance_path)

    #         yield t, des

    def extract_one(self, img_id, mirror=False):
        instance_name = "%s.%s" % (img_id, self.FILE_NAMES_EXT)
        instance_path = self.storage.get_instance_path(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
        if not self.storage.check_exists(instance_path):
            raise Exception("Calculate deep features first then load them.")
        else:
            des = self.storage.load_full_instance(instance_path)
            if mirror:
                return des[self.feature_layer][self.center_crop_index_mirror][:, 0, 0]
            else:
                return des[self.feature_layer][self.center_crop_index][:, 0, 0]
