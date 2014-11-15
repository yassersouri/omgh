from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
from datetime import datetime as dt
import pyprind

cub = CUB_200_2011(
    '/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011')
model_file = '/home/ipl/installs/caffe-rc/models/bvlc_reference_caffenet/deploy.prototxt'
pretrained_file = '/home/ipl/installs/caffe-rc/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
ilsvrc_mean = '/home/ipl/installs/caffe-rc/python/caffe/imagenet/ilsvrc_2012_mean.npy'

features_storage = datastore('/home/ipl/datastores/cub-caffe-features/')
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage,
    model_file, pretrained_file, ilsvrc_mean)

features_storage_flipped = datastore('/home/ipl/datastores/cub-caffe-features-flipped/')
feature_extractor_flipped = CNN_Features_CAFFE_REFERENCE(features_storage_flipped,
    model_file, pretrained_file, ilsvrc_mean)

features_storage_cropped = datastore('/home/ipl/datastores/cub-caffe-features-cropped/')
feature_extractor_cropped = CNN_Features_CAFFE_REFERENCE(features_storage_cropped,
    model_file, pretrained_file, ilsvrc_mean)

features_storage_flipped_cropped = datastore('/home/ipl/datastores/cub-caffe-features-flipped-cropped/')
feature_extractor_flipped_cropped = CNN_Features_CAFFE_REFERENCE(features_storage_flipped_cropped,
    model_file, pretrained_file, ilsvrc_mean)

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
