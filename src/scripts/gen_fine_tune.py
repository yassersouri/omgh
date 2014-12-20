import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import CUB_200_2011, CUB_200_2011_Segmented, CUB_200_2011_Parts_Head
import settings
import utils
import sklearn.cross_validation

DO_TEST = True
DO_TRAIN = True

data_folder = 'cub-part-head'
DO_CROP = False

base_folder = '/home/ipl/installs/caffe-rc/data/%s/' % data_folder

utils.ensure_dir(base_folder)

fine_tune_test_file = '%s/test.txt' % base_folder
fine_tune_train_file = '%s/train.txt' % base_folder
fine_tune_val_file = '%s/val.txt' % base_folder
fine_tune_train_val_file = '%s/trainval.txt' % base_folder


# cub = CUB_200_2011(settings.CUB_ROOT)
cub = CUB_200_2011_Parts_Head(settings.CUB_ROOT)
class_dict = cub.get_class_dict()

IDtrain, IDtest = cub.get_train_test_id()

if DO_TEST:
    test_file = open(fine_tune_test_file, 'w')
    all_images = cub.get_all_images(cropped=DO_CROP)
    for img_inf in all_images:
        img_id = img_inf['img_id']
        img_file = img_inf['img_file']
        if int(img_id) in IDtest:
            test_file.write("%s %s\n" % (img_file, str(class_dict[img_id] - 1)))
    test_file.close()


IDtrain_train, IDtrain_val = sklearn.cross_validation.train_test_split(IDtrain, test_size=0.20, random_state=1367)

if DO_TRAIN:
    train_file = open(fine_tune_train_file, 'w')
    all_images = cub.get_all_images(cropped=DO_CROP)
    for img_inf in all_images:
        img_id = img_inf['img_id']
        img_file = img_inf['img_file']
        if int(img_id) in IDtrain_train:
            train_file.write("%s %s\n" % (img_file, str(class_dict[img_id] - 1)))
    train_file.close()

    val_file = open(fine_tune_val_file, 'w')
    all_images = cub.get_all_images(cropped=DO_CROP)
    for img_inf in all_images:
        img_id = img_inf['img_id']
        img_file = img_inf['img_file']
        if int(img_id) in IDtrain_val:
            val_file.write("%s %s\n" % (img_file, str(class_dict[img_id] - 1)))
    val_file.close()

    trainval_file = open(fine_tune_train_val_file, 'w')
    all_images = cub.get_all_images(cropped=DO_CROP)
    for img_inf in all_images:
        img_id = img_inf['img_id']
        img_file = img_inf['img_file']
        if int(img_id) in IDtrain:
            trainval_file.write("%s %s\n" % (img_file, str(class_dict[img_id] - 1)))
    trainval_file.close()

print 'DONE'
