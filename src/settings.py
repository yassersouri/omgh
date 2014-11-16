import os
CUB_ROOT = '/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011'
MODEL_FILE = '/home/ipl/installs/caffe-rc/models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED_FILE = '/home/ipl/installs/caffe-rc/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
ILSVRC_MEAN = '/home/ipl/installs/caffe-rc/python/caffe/imagenet/ilsvrc_2012_mean.npy'

STORAGE_BASE = '/home/ipl/datastores/'

STORAGE_NAMES = {
    'ccr': 'cub-caffe-features',
    'ccf': 'cub-caffe-features-flipped',
    'ccc': 'cub-caffe-features-cropped',
    'ccfc': 'cub-caffe-features-flipped-cropped',
    'oldccc': '../repo/datastores/cub-caffe-features-cropped'
}


def storage(sname):
    return os.path.join(STORAGE_BASE, STORAGE_NAMES[sname])
