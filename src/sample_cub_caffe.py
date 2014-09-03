from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE


cub = CUB_200_2011(
    '/home/yasser/repo/datasets/CUB_200_2011/CUB_200_2011')
features_storage = datastore('/home/yasser/repo/datastores/cub-caffe-features/')
model_file = '/home/yasser/installs/caffe-0.9999/examples/imagenet/imagenet_deploy.prototxt'
pretrained_file = '/home/yasser/installs/caffe-0.9999/examples/imagenet/caffe_reference_imagenet_model'
ilsvrc_mean = '/home/yasser/installs/caffe-0.9999/python/caffe/imagenet/ilsvrc_2012_mean.npy'
feature_extractor = CNN_Features_CAFFE_REFERENCE(features_storage,
    model_file, pretrained_file, ilsvrc_mean)

for t, des in feature_extractor.extract(cub.get_all_images()):
    print t['img_id']
    print des.shape

