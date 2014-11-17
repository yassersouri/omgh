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


cub = CUB_200_2011(settings.CUB_ROOT)
features_storage = datastore(settings.storage('ccr'))
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage)

features_storage_c = datastore(settings.storage('ccc'))
feature_extractor_c = CNN_Features_CAFFE_REFERENCE(features_storage_c)

Xtrain, ytrain, Xtest, ytest = cub.get_train_test(feature_extractor.extract_one)
Xtrain_c, ytrain_c, Xtest_c, ytest_c = cub.get_train_test(feature_extractor_c.extract_one)

print Xtrain.shape, ytrain.shape
print Xtest.shape, ytest.shape

from sklearn import svm
from sklearn.metrics import accuracy_score

a = dt.now()
model = svm.LinearSVC(C=0.0001)
model.fit(numpy.concatenate((Xtrain, Xtrain_c), 1), ytrain)
b = dt.now()
print 'fitted in: %s' % (b - a)

a = dt.now()
predictions = model.predict(numpy.concatenate((Xtest, Xtest_c), 1))
b = dt.now()
print 'predicted in: %s' % (b - a)

print 'accuracy', accuracy_score(ytest, predictions)
print 'mean accuracy', utils.mean_accuracy(ytest, predictions)
