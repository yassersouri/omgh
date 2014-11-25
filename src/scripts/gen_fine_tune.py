import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import CUB_200_2011
import settings
import sklearn.cross_validation

DO_TEST = True
DO_TRAIN = True

fine_tune_test_file = '/home/ipl/installs/caffe-rc/data/cub/test.txt'
fine_tune_train_file = '/home/ipl/installs/caffe-rc/data/cub/train.txt'
fine_tune_val_file = '/home/ipl/installs/caffe-rc/data/cub/val.txt'


cub = CUB_200_2011(settings.CUB_ROOT)
class_dict = cub.get_class_dict()

IDtrain, IDtest = cub.get_train_test_id()
print IDtrain[:100]

if DO_TEST:
    test_file = open(fine_tune_test_file, 'w')
    all_images = cub.get_all_images()
    for img_inf in all_images:
        img_id = img_inf['img_id']
        img_file = img_inf['img_file']
        if int(img_id) in IDtest:
            test_file.write("%s %s\n" % (img_file, str(class_dict[img_id] - 1)))
    test_file.close()


IDtrain_train, IDtrain_val = sklearn.cross_validation.train_test_split(IDtrain, test_size=0.20, random_state=92204744)

if DO_TRAIN:
    train_file = open(fine_tune_train_file, 'w')
    all_images = cub.get_all_images()
    for img_inf in all_images:
        img_id = img_inf['img_id']
        img_file = img_inf['img_file']
        if int(img_id) in IDtrain_train:
            train_file.write("%s %s\n" % (img_file, str(class_dict[img_id] - 1)))
    train_file.close()

    val_file = open(fine_tune_val_file, 'w')
    all_images = cub.get_all_images()
    for img_inf in all_images:
        img_id = img_inf['img_id']
        img_file = img_inf['img_file']
        if int(img_id) in IDtrain_val:
            val_file.write("%s %s\n" % (img_file, str(class_dict[img_id] - 1)))
    val_file.close()

print 'DONE'
