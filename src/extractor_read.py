from extractor import BaseExtractor


class Extractor_Read(BaseExtractor):
    def __init__(self, storage, subname='hog_normalized'):
        super(Extractor_Read, self).__init__(storage)
        self.STORAGE_SUB_NAME = subname

        self.sub_folder = self.storage.get_sub_folder(
            self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME)

    def extract(self, data_generator, force=False):
        for t in data_generator:
            instance_name = "%s.%s" % (t['img_id'], self.FILE_NAMES_EXT)
            instance_path = self.storage.get_instance_path(
                self.STORAGE_SUPER_NAME, self.STORAGE_SUB_NAME, instance_name)
            if force or not self.storage.check_exists(instance_path):
                raise Exception(
                    "Calculate deep features first then load them.")
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
