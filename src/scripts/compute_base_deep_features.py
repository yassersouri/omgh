import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
from dataset import CUB_200_2011, CUB_200_2011_Parts_Head, CUB_200_2011_Parts_Body
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
import pyprind

cub = CUB_200_2011(settings.CUB_ROOT, full=False)
cub_head = CUB_200_2011_Parts_Head(settings.CUB_ROOT)
cub_body = CUB_200_2011_Parts_Body(settings.CUB_ROOT)

features_storage = datastore(settings.storage('ccr'))
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage, full=False)

features_storage_ft = datastore(settings.storage('ccrft'))
feature_extractor_ft = CNN_Features_CAFFE_REFERENCE(features_storage_ft, full=True)

features_storage_flipped = datastore(settings.storage('ccf'))
feature_extractor_flipped = CNN_Features_CAFFE_REFERENCE(features_storage_flipped, full=False)

features_storage_flipped_ft = datastore(settings.storage('ccfft'))
feature_extractor_flipped_ft = CNN_Features_CAFFE_REFERENCE(features_storage_flipped_ft, full=True)

features_storage_cropped = datastore(settings.storage('ccc'))
feature_extractor_cropped = CNN_Features_CAFFE_REFERENCE(features_storage_cropped, full=False)

features_storage_cropped_ft = datastore(settings.storage('cccft'))
feature_extractor_cropped_ft = CNN_Features_CAFFE_REFERENCE(features_storage_cropped_ft, full=True)

features_storage_flipped_cropped = datastore(settings.storage('ccfc'))
feature_extractor_flipped_cropped = CNN_Features_CAFFE_REFERENCE(features_storage_flipped_cropped, full=False)

features_storage_flipped_cropped_ft = datastore(settings.storage('ccfcft'))
feature_extractor_flipped_cropped_ft = CNN_Features_CAFFE_REFERENCE(features_storage_flipped_cropped_ft, full=True)

features_storage_part_head = datastore(settings.storage('ccphead'))
feature_extractor_part_head = CNN_Features_CAFFE_REFERENCE(features_storage_part_head, full=False)

features_storage_part_body = datastore(settings.storage('ccpbody'))
feature_extractor_part_body = CNN_Features_CAFFE_REFERENCE(features_storage_part_body, full=False)

number_of_images_in_dataset = sum(1 for _ in cub.get_all_images())

print 'cub, regular'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor.extract_all(cub.get_all_images(), flip=False, crop=False, bbox=None, force=False):
    bar.update()
print '----------------------'

print 'cub, regular, ft'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_ft.extract_all(cub.get_all_images(), flip=False, crop=False, bbox=None, force=False):
    bar.update()
print '----------------------'

print 'cub, flipped'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_flipped.extract_all(cub.get_all_images(), flip=True, crop=False, bbox=None, force=False):
    bar.update()
print '----------------------'

print 'cub, flipped, ft'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_flipped_ft.extract_all(cub.get_all_images(), flip=True, crop=False, bbox=None, force=False):
    bar.update()
print '----------------------'

print 'cub, cropped'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_cropped.extract_all(cub.get_all_images(), flip=False, crop=True, bbox=cub.get_bbox(), force=False):
    bar.update()
print '----------------------'

print 'cub, cropped, ft'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_cropped_ft.extract_all(cub.get_all_images(), flip=False, crop=True, bbox=cub.get_bbox(), force=False):
    bar.update()
print '----------------------'

print 'cub, flipped & cropped'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_flipped_cropped.extract_all(cub.get_all_images(), flip=True, crop=True, bbox=cub.get_bbox(), force=False):
    bar.update()
print '----------------------'

print 'cub, flipped & cropped, ft'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_flipped_cropped_ft.extract_all(cub.get_all_images(), flip=True, crop=True, bbox=cub.get_bbox(), force=False):
    bar.update()
print '----------------------'

print 'cub, head'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_part_head.extract_all(cub_head.get_all_images(), flip=False, crop=False, bbox=None, force=False):
    bar.update()
print '----------------------'

print 'cub, body'
bar = pyprind.ProgBar(number_of_images_in_dataset, width=100)
for t, des in feature_extractor_part_body.extract_all(cub_body.get_all_images(), flip=False, crop=False, bbox=None, force=False):
    bar.update()
print '----------------------'
