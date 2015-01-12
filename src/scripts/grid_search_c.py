import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
from datetime import datetime as dt
import settings
import utils
import numpy as np


cub = CUB_200_2011(settings.CUB_ROOT)
features_storage_r = datastore(settings.storage('ccrft'))
feature_extractor_r = CNN_Features_CAFFE_REFERENCE(features_storage_r, make_net=False)

features_storage_c = datastore(settings.storage('cccft'))
feature_extractor_c = CNN_Features_CAFFE_REFERENCE(features_storage_c, make_net=False)

features_storage_p_h = datastore(settings.storage('ccpheadft-100000'))
feature_extractor_p_h = CNN_Features_CAFFE_REFERENCE(features_storage_p_h, make_net=False)

features_storage_p_h = datastore(settings.storage('ccpheadft-100000'))
feature_extractor_p_h = CNN_Features_CAFFE_REFERENCE(features_storage_p_h, make_net=False)

features_storage_p_b = datastore(settings.storage('ccpbodyft-10000'))
feature_extractor_p_b = CNN_Features_CAFFE_REFERENCE(features_storage_p_b, make_net=False)

Xtrain_r, ytrain_r, Xtest_r, ytest_r = cub.get_train_test(feature_extractor_r.extract_one)
Xtrain_c, ytrain_c, Xtest_c, ytest_c = cub.get_train_test(feature_extractor_c.extract_one)
Xtrain_p_h, ytrain_p_h, Xtest_p_h, ytest_p_h = cub.get_train_test(feature_extractor_p_h.extract_one)
Xtrain_p_b, ytrain_p_b, Xtest_p_b, ytest_p_b = cub.get_train_test(feature_extractor_p_b.extract_one)

Xtrain = np.concatenate((Xtrain_r, Xtrain_c, Xtrain_p_h, Xtrain_p_b), axis=1)
Xtest = np.concatenate((Xtest_r, Xtest_c, Xtest_p_h, Xtest_p_b), axis=1)

import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

CS = numpy.array([100, 10, 1, 0.1, 0.01, 0.001, 0.0001])
model = svm.LinearSVC()
grid_search = GridSearchCV(estimator=model, param_grid=dict(C=CS), n_jobs=3)

grid_search.fit(Xtrain, ytrain_r)

print 'best c:', grid_search.best_params_


a = dt.now()
model = svm.LinearSVC(C=grid_search.best_params_['C'])
model.fit(Xtrain, ytrain_r)
b = dt.now()
print 'fitted in: %s' % (b - a)

a = dt.now()
predictions = model.predict(Xtest)
b = dt.now()
print 'predicted in: %s' % (b - a)

print 'accuracy', accuracy_score(ytest_r, predictions)
print 'mean accuracy', utils.mean_accuracy(ytest_r, predictions)
