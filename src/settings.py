import os

CAFFE_ROOT = '/home/ipl/installs/caffe-rc'
CAFFE_PYTHON_PATH = os.path.join(CAFFE_ROOT, 'python')
CUB_ROOT = '/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011'
CUB_IMAGES_FOLDER = '%s/images/' % CUB_ROOT
DEFAULT_MODEL_FILE = '%s/models/bvlc_reference_caffenet/deploy.prototxt' % CAFFE_ROOT
DEFAULT_PRETRAINED_FILE = '%s/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel' % CAFFE_ROOT
ILSVRC_MEAN = '%s/python/caffe/imagenet/ilsvrc_2012_mean.npy' % CAFFE_ROOT
MODEL_FILE_TEMP = '%s/models/%s/deploy.prototxt'
PRETRAINED_FILE_TEMP = '%s/models/%s/%s_iter_%d.caffemodel'

STORAGE_BASE = '/home/ipl/datastores/'
PREDICTIONS_BASE = '%spredictions/' % STORAGE_BASE
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
    'ccfcft': 'cub-caffe-features-flipped-cropped-ft',
    'ccsc': 'cub-caffe-features-segmented-cropped',
    'cache-cccftt': 'cache-cccftt'
}

MODEL_NAMES = {
    'def': DEFAULT_MODEL_FILE
}

PRETRAINED_NAMES = {
    'def': DEFAULT_PRETRAINED_FILE
}


def dyn_aug(sname, folder_name, full_name, iter_len=10):
    for i in range(iter_len):
        iteration = (i+1) * 10000
        name = '%s-%d' % (sname, iteration)
        STORAGE_NAMES[name] = name
        MODEL_NAMES[name] = MODEL_FILE_TEMP % (CAFFE_ROOT, folder_name)
        PRETRAINED_NAMES[name] = PRETRAINED_FILE_TEMP % (CAFFE_ROOT, folder_name, full_name, iteration)

dyn_aug('cccftv1', 'finetune_cub_cropped_val_1', 'finetune_cub', 10)
dyn_aug('cccftv1_2', 'finetune_cub_cropped_val_1_2', 'finetune_cub', 10)
dyn_aug('cccftv2', 'finetune_cub_cropped_val_2', 'finetune_cub', 10)
dyn_aug('cccftt', 'finetune_cub_cropped', 'finetune_cub_cropped', 10)
dyn_aug('ccrftt', 'finetune_cub', 'finetune_cub', 10)
dyn_aug('ccpheadft', 'finetune_cub_part_head', 'finetune_cub_part_head', 10)


def storage(sname):
    return os.path.join(STORAGE_BASE, STORAGE_NAMES[sname])


def model(mname):
    return MODEL_NAMES[mname]


def pretrained(pname):
    return PRETRAINED_NAMES[pname]
