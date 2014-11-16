import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
from datetime import datetime as dt
import settings


cub = CUB_200_2011(settings.CUB_ROOT)
features_storage = datastore(settings.storage('ccc'))

feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage)

Xtrain, ytrain, Xtest, ytest = cub.get_train_test(feature_extractor.extract_one)

print Xtrain.shape, ytrain.shape
print Xtest.shape, ytest.shape

import numpy
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

CS = numpy.array([3, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
model = svm.LinearSVC()
grid_search = GridSearchCV(estimator=model, param_grid=dict(C=CS), n_jobs=-1)

grid_search.fit(Xtrain, ytrain)

print 'best c:', grid_search.best_params_


a = dt.now()
model = svm.LinearSVC(C=grid_search.best_params_['C'])
model.fit(Xtrain, ytrain)
b = dt.now()
print 'fitted in: %s' % (b - a)

a = dt.now()
predictions = model.predict(Xtest)
b = dt.now()
print 'predicted in: %s' % (b - a)

print accuracy_score(ytest, predictions)
