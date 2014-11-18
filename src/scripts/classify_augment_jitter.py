import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
from datetime import datetime as dt
import numpy
import settings
import utils


cub_full = CUB_200_2011(settings.CUB_ROOT, full=True)
cub = CUB_200_2011(settings.CUB_ROOT, full=False)
features_storage = datastore(settings.storage('ccc'))
feature_extractor_full = CNN_Features_CAFFE_REFERENCE(features_storage, full=True)
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage, full=False)

features_storage_f = datastore(settings.storage('ccfc'))
feature_extractor_full_f = CNN_Features_CAFFE_REFERENCE(features_storage_f, full=True)
feature_extractor_f = CNN_Features_CAFFE_REFERENCE(features_storage_f, full=False)

Xtrain, ytrain, Xtest, ytest = cub_full.get_train_test(feature_extractor_full.extract_one, feature_extractor.extract_one)
Xtrain_f, ytrain_f, Xtest_f, ytest_f = cub_full.get_train_test(feature_extractor_full_f.extract_one, feature_extractor_f.extract_one)

print Xtrain.shape, ytrain.shape
print Xtest.shape, ytest.shape

from sklearn import svm
from sklearn.metrics import accuracy_score

a = dt.now()
model = svm.LinearSVC(C=0.0001)
model.fit(numpy.concatenate((Xtrain, Xtrain_f)), numpy.concatenate((ytrain, ytrain_f)))
b = dt.now()
print 'fitted in: %s' % (b - a)

a = dt.now()
predictions = model.predict(Xtest)
b = dt.now()
print 'predicted in: %s' % (b - a)

print 'accuracy', accuracy_score(ytest, predictions)
print 'mean accuracy', utils.mean_accuracy(ytest, predictions)
