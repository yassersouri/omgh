import os
import scipy.io


class datastore(object):

    def __init__(self, base_path, global_key='global_key'):
        self.base_path = base_path
        self.global_key = global_key

    @classmethod
    def ensure_dir(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def get_super_folder(self, super_name):
        super_folder = os.path.join(self.base_path, super_name)
        return super_folder

    def get_sub_folder(self, super_name, sub_name):
        super_folder = os.path.join(self.base_path, super_name)
        sub_folder = os.path.join(super_folder, sub_name)
        return sub_folder

    def get_instance_path(self, super_name, sub_name, instance_name):
        sub_folder = self.get_sub_folder(super_name, sub_name)
        return os.path.join(sub_folder, instance_name)

    def get_model_path(self, super_name, model_name):
        super_folder = self.get_super_folder(super_name)
        return os.path.join(super_folder, model_name)

    def check_exists(self, instance_path):
        if os.path.exists(instance_path):
            return True
        else:
            return False

    def save_instance(self, instance_path, instance):
        scipy.io.savemat(
            instance_path, {self.global_key: instance}, do_compression=True)

    def save_full_instance(self, instance_path, instance):
        scipy.io.savemat(instance_path, instance, do_compression=True)

    def load_instance(self, instance_path):
        instance = scipy.io.loadmat(instance_path)
        return instance[self.global_key]

    def load_full_instance(self, instance_path):
        instance = scipy.io.loadmat(instance_path)
        return instance
