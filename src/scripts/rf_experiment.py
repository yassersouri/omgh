import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
import utils
sys.path.append(settings.CAFFE_PYTHON_PATH)
from storage import datastore
from dataset import CUB_200_2011
from parts import Parts
import cub_utils
import click
import numpy as np
import sklearn.svm
import sklearn.ensemble
import sklearn.metrics
from time import time


@click.command()
def main():
    instance_split = 10

    dh = cub_utils.DeepHelper()
    cub = CUB_200_2011(settings.CUB_ROOT)
    cub_parts = cub.get_parts()
    IDtrain, IDtest = cub.get_train_test_id()
    all_image_infos = cub.get_all_image_infos()
    all_segmentaion_infos = cub.get_all_segmentation_infos()

    rf_safe = datastore('rf')
    rf_safe.super_name = 'features'
    rf_safe.sub_name = 'head'

    Xtrain_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'Xtrain')
    Xtest_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'Xtest')
    ytrain_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'ytrain.mat')
    ytest_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'ytest.mat')

    tic = time()
    if rf_safe.check_exists(Xtrain_ip) and rf_safe.check_exists(ytrain_ip):
        Xtrain = rf_safe.load_large_instance(Xtrain_ip, instance_split)
        ytrain = rf_safe.load_instance(ytrain_ip)
    else:
        Xtrain, ytrain = dh.part_features_for_rf(all_image_infos, all_segmentaion_infos, cub_parts, IDtrain, Parts.HEAD_PART_NAMES)

        rf_safe.save_large_instance(Xtrain_ip, Xtrain, instance_split)
        rf_safe.save_instance(ytrain_ip, ytrain)

    if rf_safe.check_exists(Xtest_ip) and rf_safe.check_exists(ytest_ip):
        Xtest = rf_safe.load_large_instance(Xtest_ip, instance_split)
        ytest = rf_safe.load_instance(ytest_ip)
    else:
        Xtest, ytest = dh.part_features_for_rf(all_image_infos, all_segmentaion_infos, cub_parts, IDtest, Parts.HEAD_PART_NAMES)

        rf_safe.save_large_instance(Xtest_ip, Xtest, instance_split)
        rf_safe.save_instance(ytest_ip, ytest)
    toc = time()
    print 'loaded or calculated in', toc - tic

if __name__ == '__main__':
    main()
