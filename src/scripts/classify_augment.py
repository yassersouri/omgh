import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
from datetime import datetime as dt
import numpy
import settings


cub = CUB_200_2011(settings.CUB_ROOT)
features_storage = datastore(settings.storage('ccc'))
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage)

features_storage_f = datastore(settings.storage('ccfc'))
feature_extractor_f = CNN_Features_CAFFE_REFERENCE(features_storage_f)

Xtrain, ytrain, Xtest, ytest = cub.get_train_test(feature_extractor.extract_one)
Xtrain_f, ytrain_f, Xtest_f, ytest_f = cub.get_train_test(feature_extractor_f.extract_one)

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

print accuracy_score(ytest, predictions)
