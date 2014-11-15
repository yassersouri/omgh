import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
from datetime import datetime as dt
import pyprind

cub = CUB_200_2011(settings.CUB_ROOT)

features_storage = datastore(os.path.join(settings.STORAGE_BASE, 'cub-caffe-features'))
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage,
    settings.MODEL_FILE, settings.PRETRAINED_FILE, settings.ILSVRC_MEAN)

features_storage_flipped = datastore(os.path.join(settings.STORAGE_BASE, 'cub-caffe-features-flipped'))
feature_extractor_flipped = CNN_Features_CAFFE_REFERENCE(features_storage_flipped,
    settings.MODEL_FILE, settings.PRETRAINED_FILE, settings.ILSVRC_MEAN)

features_storage_cropped = datastore(os.path.join(settings.STORAGE_BASE, 'cub-caffe-features-cropped'))
feature_extractor_cropped = CNN_Features_CAFFE_REFERENCE(features_storage_cropped,
    settings.MODEL_FILE, settings.PRETRAINED_FILE, settings.ILSVRC_MEAN)

features_storage_flipped_cropped = datastore(os.path.join(settings.STORAGE_BASE, 'cub-caffe-features-flipped-cropped'))
feature_extractor_flipped_cropped = CNN_Features_CAFFE_REFERENCE(features_storage_flipped_cropped,
    settings.MODEL_FILE, settings.PRETRAINED_FILE, settings.ILSVRC_MEAN)

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
