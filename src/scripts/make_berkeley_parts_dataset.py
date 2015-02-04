import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)
import utils
import cub_utils
from dataset import CUB_200_2011
import parts
import cv2

b_head_folder = os.path.join(settings.CUB_ROOT, 'images_b_head')
b_body_folder = os.path.join(settings.CUB_ROOT, 'images_b_body')
b_crop_folder = os.path.join(settings.CUB_ROOT, 'images_b_cropped')


def main():
    cub = CUB_200_2011(settings.CUB_ROOT)
    cub_images = cub.get_all_images()
    IDtrain, IDtest = cub.get_train_test_id()
    bah = cub_utils.BerkeleyAnnotationsHelper(settings.BERKELEY_ANNOTATION_BASE_PATH, IDtrain, IDtest)
    for i, image in enumerate(cub_images):
        image_path = image['img_file']
        image_id = int(image['img_id'])

        rel_image_path = image_path[len(settings.CUB_IMAGES_FOLDER):]
        o_image = cv2.imread(image_path)

        head_info = bah.get_berkeley_annotation(image_id, 'head')
        body_info = bah.get_berkeley_annotation(image_id, 'body')
        crop_info = bah.get_berkeley_annotation(image_id, 'bbox')

        if -1 in head_info:
            print 'NO HEAD \t IMG-ID: %d' % image_id
        else:
            head_image = utils.get_rect(o_image, head_info)
            head_out_path = os.path.join(b_head_folder, rel_image_path)
            utils.ensure_dir(head_out_path)
            cv2.imwrite(head_out_path, head_image)

        if -1 in body_info:
            print 'NO BODY \t IMG-ID: %d' % image_id
        else:
            body_image = utils.get_rect(o_image, body_info)
            body_out_path = os.path.join(b_body_folder, rel_image_path)
            utils.ensure_dir(body_out_path)
            cv2.imwrite(body_out_path, body_image)

        if -1 in crop_info:
            print 'NO CROP \t IMG-ID: %d' % image_id
        else:
            crop_image = utils.get_rect(o_image, crop_info)
            crop_out_path = os.path.join(b_crop_folder, rel_image_path)
            utils.ensure_dir(os.path.dirname(crop_out_path))
            cv2.imwrite(crop_out_path, crop_image)


if __name__ == '__main__':
    main()
