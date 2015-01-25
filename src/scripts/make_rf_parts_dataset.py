import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe
import click
import cub_utils
import utils
from dataset import CUB_200_2011
from parts import Parts, gen_dense_points, Part
import cv2
from storage import datastore
import sklearn.ensemble
import scipy.stats
import skimage.measure
import skimage.morphology
import numpy as np


@click.command()
@click.argument('out-path', type=click.Path(exists=True))
def main(out_path):
    instance_split = 10
    cub = CUB_200_2011(settings.CUB_ROOT)
    cub_images = cub.get_all_images()
    dh = cub_utils.DeepHelper()

    rf_safe = datastore(settings.storage('rf'))
    rf_safe.super_name = 'features'
    rf_safe.sub_name = 'head-points'
    rf_safe.other_sub_name = 'head-final-features'

    Xtrain_rf_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'Xtrain_rf')
    ytrain_rf_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'ytrain_rf.mat')

    Xtrain_rf = rf_safe.load_large_instance(Xtrain_rf_ip, instance_split)
    ytrain_rf = rf_safe.load_instance(ytrain_rf_ip)
    ytrain_rf = ytrain_rf[0, :]

    model_rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, bootstrap=False, max_depth=10, n_jobs=3, random_state=None, verbose=0)
    model_rf.fit(Xtrain_rf, ytrain_rf)

    dense_points = gen_dense_points(227, 227)

    for i, image in enumerate(cub_images):
        print i
        image_path = image['img_file']
        rel_image_path = image_path[len(settings.CUB_IMAGES_FOLDER):]

        o_image = cv2.imread(image_path)
        img = caffe.io.load_image(image_path)

        dh.init_with_image(img)
        X = dh.features(dense_points)
        preds_prob = model_rf.predict_proba(X)
        max_prob = np.max(preds_prob[:, 1])
        preds_prob = preds_prob[:, 1].reshape((227, 227)).T
        preds = preds_prob >= (max_prob/2)
        preds = skimage.morphology.closing(preds, skimage.morphology.square(10))
        preds = skimage.morphology.remove_small_objects(preds, min_size=10, connectivity=1)
        L, N = skimage.measure.label(preds, return_num=True, background=0)
        L_no_bg = L[L != -1].flatten()
        vals, counts = scipy.stats.mode(L_no_bg)
        part_label = int(vals[0])
        indices = np.where(L == part_label)
        xmin = indices[0].min()
        xmax = indices[0].max()
        ymin = indices[1].min()
        ymax = indices[1].max()
        pmin = Part(-1, '?', -1, xmin, ymin, 1)
        pmax = Part(-1, '?', -1, xmax, ymax, 1)
        rect_parts = Parts(parts=[pmin, pmax])
        rect_parts.denorm_for_size(img.shape[0], img.shape[1], size=227)
        rect_info = rect_parts[0].x, rect_parts[1].x, rect_parts[0].y, rect_parts[1].y

        t_img_part = Parts().get_rect(o_image, rect_info=rect_info)

        out_image_path = os.path.join(out_path, rel_image_path)
        utils.ensure_dir(os.path.dirname(out_image_path))
        cv2.imwrite(out_image_path, t_img_part)


if __name__ == '__main__':
    main()
