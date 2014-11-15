import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from matplotlib import pylab as plt
from sklearn import svm
from sklearn import metrics
from datetime import datetime as dt
import numpy as np

from dataset import CUB_200_2011
from storage import datastore
from deep_extractor_read import CNN_Features_CAFFE_REFERENCE

cub = CUB_200_2011(
    '/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011')

from deep_extractor_full_read import CNN_Features_CAFFE_REFERENCE
features_storage_caug = datastore('/home/ipl/datastores/cub-caffe-features-cropped-full')

feature_extractor_caug = CNN_Features_CAFFE_REFERENCE(features_storage_caug)

Xtrain_caug, ytrain_caug, Xtest_c, ytest_c = cub.get_train_test(feature_extractor_caug.extract_one)