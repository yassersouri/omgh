import os
CUB_ROOT = '/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011'
MODEL_FILE = '/home/ipl/installs/caffe-rc/models/finetune_cub/deploy.prototxt'
PRETRAINED_FILE = '/home/ipl/installs/caffe-rc/models/finetune_cub/finetune_cub_iter_100000.caffemodel'
ILSVRC_MEAN = '/home/ipl/installs/caffe-rc/python/caffe/imagenet/ilsvrc_2012_mean.npy'

STORAGE_BASE = '/home/ipl/datastores/'
FULL_LENGTH = 10

STORAGE_NAMES = {
    'ccr': 'cub-caffe-features',
    'ccf': 'cub-caffe-features-flipped',
    'ccc': 'cub-caffe-features-cropped',
    'ccfc': 'cub-caffe-features-flipped-cropped',
    'oldccc': '../repo/datastores/cub-caffe-features-cropped',
    'ccrft': 'cub-caffe-features-ft',
    'cccft': 'cub-caffe-features-cropped-ft',
    'ccfft': 'cub-caffe-features-flipped-ft',
    'ccfcft': 'cub-caffe-features-flipped-cropped-ft'
}


def storage(sname):
    return os.path.join(STORAGE_BASE, STORAGE_NAMES[sname])
