import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
import pyprind

cub = CUB_200_2011(settings.CUB_ROOT)

features_storage = datastore(settings.storage('ccr'))
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage)

features_storage_flipped = datastore(settings.storage('ccf'))
feature_extractor_flipped = CNN_Features_CAFFE_REFERENCE(features_storage_flipped)

features_storage_cropped = datastore(settings.storage('ccc'))
feature_extractor_cropped = CNN_Features_CAFFE_REFERENCE(features_storage_cropped)

features_storage_flipped_cropped = datastore(settings.storage('ccfc'))
feature_extractor_flipped_cropped = CNN_Features_CAFFE_REFERENCE(features_storage_flipped_cropped)

number_of_images_in_dataset = sum(1 for _ in cub.get_all_images())

print 'cub, regular'
bar = pyprind.ProgBar(number_of_images_in_dataset)
for t, des in feature_extractor.extract_all(cub.get_all_images(), flip=False, crop=False, bbox=cub.get_bbox()):
    bar.update()
print '----------------------'

print 'cub, flipped'
bar = pyprind.ProgBar(number_of_images_in_dataset)
for t, des in feature_extractor_flipped.extract_all(cub.get_all_images(), flip=True, crop=False, bbox=cub.get_bbox()):
    bar.update()
print '----------------------'

print 'cub, cropped'
bar = pyprind.ProgBar(number_of_images_in_dataset)
for t, des in feature_extractor_cropped.extract_all(cub.get_all_images(), flip=False, crop=True, bbox=cub.get_bbox()):
    bar.update()
print '----------------------'

print 'cub, flipped & cropped'
bar = pyprind.ProgBar(number_of_images_in_dataset)
for t, des in feature_extractor_flipped_cropped.extract_all(cub.get_all_images(), flip=True, crop=True, bbox=cub.get_bbox()):
    bar.update()
print '----------------------'
