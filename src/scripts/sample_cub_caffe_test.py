from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
from datetime import datetime as dt


cub = CUB_200_2011(
    '/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011')
features_storage = datastore('/home/ipl/datastores/cub-caffe-features/')

feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage)

Xtrain, ytrain, Xtest, ytest = cub.get_train_test(feature_extractor.extract_one)

print Xtrain.shape, ytrain.shape
print Xtest.shape, ytest.shape

from sklearn import svm
from sklearn.metrics import classification_report

a = dt.now()
model = svm.LinearSVC()
model.fit(Xtrain, ytrain)
b = dt.now()
print 'fitted in: %s', (b - a)

a = dt.now()
predictions = model.predict(Xtest)
b = dt.now()
print 'predicted in: %s', (b - a)

print classification_report(ytest, predictions)
