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

BERKELEY_BASE_PATH = '/home/ipl/repo/part-based-RCNN'
BERKELEY_ANNOTATION_BASE_PATH = os.path.join(BERKELEY_BASE_PATH, 'annotations')
BERKELEY_CACHE_FOLDER = os.path.join(BERKELEY_BASE_PATH, 'caches')
BERKELEY_MODEL_FILE = os.path.join(BERKELEY_CACHE_FOLDER, 'cub_finetune_deploy_fc7.prototxt')
BERKELEY_CROP_PRET = os.path.join(BERKELEY_CACHE_FOLDER, 'CUB_bbox_finetune.caffe_model')
BERKELEY_HEAD_PRET = os.path.join(BERKELEY_CACHE_FOLDER, 'CUB_body_finetune.caffe_model')
BERKELEY_BODY_PRET = os.path.join(BERKELEY_CACHE_FOLDER, 'CUB_head_finetune.caffe_model')

STORAGE_NAMES = {
    'ccr': 'cub-caffe-features',
    'ccf': 'cub-caffe-features-flipped',
    'ccc': 'cub-caffe-features-cropped',
    'ccphead': 'cub-caffe-features-part-head',
    'ccpbody': 'cub-caffe-features-part-body',
    'ccfc': 'cub-caffe-features-flipped-cropped',
    'oldccc': '../repo/datastores/cub-caffe-features-cropped',
    'ccrft': 'cub-caffe-features-ft',
    'cccft': 'cub-caffe-features-cropped-ft',
    'ccfft': 'cub-caffe-features-flipped-ft',
    'ccfcft': 'cub-caffe-features-flipped-cropped-ft',
    'ccsc': 'cub-caffe-features-segmented-cropped',
    'cache-cccftt': 'cache-cccftt',
    'nn-parts': 'nn-parts',
    'rf': 'rf',
    'bmbc': 'berkeley-model-berkeley-crop',
    'bmbcflp': 'berkeley-model-berkeley-crop-flipped',
    'bmbh': 'berkeley-model-berkeley-head',
    'bmbb': 'berkeley-model-berkeley-body',
    'nn-cache': 'nn-cache',
    'ss-cache': 'ss-cache',
    'ccrp5': 'ccrp5'
}

MODEL_NAMES = {
    'def': DEFAULT_MODEL_FILE
}

PRETRAINED_NAMES = {
    'def': DEFAULT_PRETRAINED_FILE
}


def dyn_aug(sname, folder_name, full_name, iter_len=10, iter_step=10000):
    for i in range(iter_len):
        iteration = (i+1) * iter_step
        name = '%s-%d' % (sname, iteration)
        STORAGE_NAMES[name] = name
        MODEL_NAMES[name] = MODEL_FILE_TEMP % (CAFFE_ROOT, folder_name)
        PRETRAINED_NAMES[name] = PRETRAINED_FILE_TEMP % (CAFFE_ROOT, folder_name, full_name, iteration)

dyn_aug('cccftv1', 'finetune_cub_cropped_val_1', 'finetune_cub', 10)
dyn_aug('cccftv1_2', 'finetune_cub_cropped_val_1_2', 'finetune_cub', 10)
dyn_aug('cccftv2', 'finetune_cub_cropped_val_2', 'finetune_cub', 10)
dyn_aug('cccftt', 'finetune_cub_cropped', 'finetune_cub_cropped', 10)
dyn_aug('ccrftt', 'finetune_cub', 'finetune_cub', 10)
dyn_aug('cccfttflp', 'finetune_cub_cropped', 'finetune_cub_cropped', 10)
dyn_aug('ccrfttflp', 'finetune_cub', 'finetune_cub', 10)
dyn_aug('ccpheadft', 'finetune_cub_part_head', 'finetune_cub_part_head', 10)
dyn_aug('ccpbodyft', 'finetune_cub_part_body', 'finetune_cub_part_body', 10)
dyn_aug('ccpheadrfft', 'finetune_cub_part_head_rf', 'finetune_cub_part_head_rf', 10)
dyn_aug('ccrft2st', 'finetune_2step_cub', 'cub_2step_step2', 4, 2500)
dyn_aug('ccrft2stflp', 'finetune_2step_cub', 'cub_2step_step2', 4, 2500)
dyn_aug('cccft2st', 'finetune_2step_cub_cropped', 'cub_2step_step2', 2, 25000)
dyn_aug('cccft2stflp', 'finetune_2step_cub_cropped', 'cub_2step_step2', 2, 25000)
dyn_aug('ccpheadrfftn', 'finetune_cub_part_head_rf_new', 'finetune_cub_part', 10)
dyn_aug('ccpbodyrfftn', 'finetune_cub_part_body_rf_new', 'finetune_cub_part', 10)
dyn_aug('ccpheadrfftnflp', 'finetune_cub_part_head_rf_new', 'finetune_cub_part', 10)
dyn_aug('ccpbodyrfftnflp', 'finetune_cub_part_body_rf_new', 'finetune_cub_part', 10)
dyn_aug('ccpheadrfftn2st', 'finetune_2step_cub_part_head_rf', 'cub_2step_step2', 2, 25000)
dyn_aug('ccpbodyrfftn2st', 'finetune_2step_cub_part_body_rf', 'cub_2step_step2', 2, 25000)
dyn_aug('ccpheadrfftn2stflp', 'finetune_2step_cub_part_head_rf', 'cub_2step_step2', 2, 25000)
dyn_aug('ccpbodyrfftn2stflp', 'finetune_2step_cub_part_body_rf', 'cub_2step_step2', 2, 25000)


def storage(sname):
    return os.path.join(STORAGE_BASE, STORAGE_NAMES[sname], )


def model(mname):
    return MODEL_NAMES[mname]


def pretrained(pname):
    return PRETRAINED_NAMES[pname]
