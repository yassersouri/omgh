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

features_storage_f = datastore(settings.storage('ccf'))
feature_extractor_f = CNN_Features_CAFFE_REFERENCE(features_storage_f)

features_storage_fc = datastore(settings.storage('ccfc'))
feature_extractor_fc = CNN_Features_CAFFE_REFERENCE(features_storage_fc)

Xtrain, ytrain, Xtest, ytest = cub.get_train_test(feature_extractor.extract_one)
Xtrain_c, ytrain_c, Xtest_c, ytest_c = cub.get_train_test(feature_extractor_c.extract_one)
Xtrain_f, ytrain_f, Xtest_f, ytest_f = cub.get_train_test(feature_extractor_f.extract_one)
Xtrain_fc, ytrain_fc, Xtest_fc, ytest_fc = cub.get_train_test(feature_extractor_fc.extract_one)

print Xtrain.shape, ytrain.shape
print Xtest.shape, ytest.shape

from sklearn import svm
from sklearn.metrics import accuracy_score

Xtrain_aug = numpy.concatenate((Xtrain, Xtrain_f))
Xtrain_c_aug = numpy.concatenate((Xtrain_c, Xtrain_fc))
ytrain_aug = numpy.concatenate((ytrain, ytrain_f))

a = dt.now()
model = svm.LinearSVC(C=0.0001)
model.fit(numpy.concatenate((Xtrain_aug, Xtrain_c_aug), 1), ytrain_aug)
b = dt.now()
print 'fitted in: %s' % (b - a)

a = dt.now()
predictions = model.predict(numpy.concatenate((Xtest, Xtest_c), 1))
b = dt.now()
print 'predicted in: %s' % (b - a)

print 'accuracy', accuracy_score(ytest, predictions)
print 'mean accuracy', utils.mean_accuracy(ytest, predictions)
