from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE


cub = CUB_200_2011(
    '/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011')
features_storage = datastore('/home/ipl/datastores/cub-caffe-features-cropped/')
model_file = '/home/ipl/installs/caffe-0.9999/examples/imagenet/imagenet_deploy.prototxt'
pretrained_file = '/home/ipl/installs/caffe-0.9999/examples/imagenet/caffe_reference_imagenet_model'
ilsvrc_mean = '/home/ipl/installs/caffe-0.9999/python/caffe/imagenet/ilsvrc_2012_mean.npy'
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage,
    model_file, pretrained_file, ilsvrc_mean)

for t, des in feature_extractor.extract_cropped(cub.get_all_images(), cub.get_bbox()):
    print t['img_id']
    print des.shape

