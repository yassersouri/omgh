import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
import utils
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe
from storage import datastore
from dataset import CUB_200_2011
from deep_extractor import CNN_Features_CAFFE_REFERENCE, Berkeley_Extractor
from parts import Parts, gen_dense_points, Part
import cub_utils
import click
import numpy as np
import scipy.stats
import skimage.measure
import skimage.morphology
import sklearn.svm
import sklearn.ensemble
import sklearn.metrics
from time import time


@click.command()
@click.option('c', '--svm-c', type=click.FLOAT, default=0.0001)
@click.option('f', '--force', type=click.BOOL, default=False)
def main(c, f):
    instance_split = 10
    feat_layer = 'fc7'
    load_rf_test = False
    recalculate_training = True
    C = c
    force = f

    dh = cub_utils.DeepHelper()
    cub = CUB_200_2011(settings.CUB_ROOT)
    cub_parts = cub.get_parts()
    IDtrain, IDtest = cub.get_train_test_id()
    all_image_infos = cub.get_all_image_infos()
    all_segmentaion_infos = cub.get_all_segmentation_infos()

    rf_safe = datastore(settings.storage('rf'))
    rf_safe.super_name = 'features'
    rf_safe.sub_name = 'head-points'
    rf_safe.other_sub_name = 'head-final-features'

    Xtrain_rf_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'Xtrain_rf')
    Xtest_rf_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'Xtest_rf')
    ytrain_rf_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'ytrain_rf.mat')
    ytest_rf_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.sub_name, 'ytest_rf.mat')
    Xtrain_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.other_sub_name, 'Xtrain')
    Xtest_ip = rf_safe.get_instance_path(rf_safe.super_name, rf_safe.other_sub_name, 'Xtest')

    tic = time()
    if rf_safe.check_exists(ytrain_rf_ip) and not force:
        print 'loading'
        Xtrain_rf = rf_safe.load_large_instance(Xtrain_rf_ip, instance_split)
        ytrain_rf = rf_safe.load_instance(ytrain_rf_ip)
        ytrain_rf = ytrain_rf[0, :]
    else:
        print 'calculating'
        Xtrain_rf, ytrain_rf = dh.part_features_for_rf(all_image_infos, all_segmentaion_infos, cub_parts, IDtrain, Parts.HEAD_PART_NAMES)

        rf_safe.save_large_instance(Xtrain_rf_ip, Xtrain_rf, instance_split)
        rf_safe.save_instance(ytrain_rf_ip, ytrain_rf)

    if load_rf_test:
        if rf_safe.check_exists(ytest_rf_ip) and not force:
            Xtest_rf = rf_safe.load_large_instance(Xtest_rf_ip, instance_split)
            ytest_rf = rf_safe.load_instance(ytest_rf_ip)
            ytest_rf = ytest_rf[0, :]
        else:
            Xtest_rf, ytest_rf = dh.part_features_for_rf(all_image_infos, all_segmentaion_infos, cub_parts, IDtest, Parts.HEAD_PART_NAMES)

            rf_safe.save_large_instance(Xtest_rf_ip, Xtest_rf, instance_split)
            rf_safe.save_instance(ytest_rf_ip, ytest_rf)
    toc = time()
    print 'loaded or calculated in', toc - tic

    tic = time()
    model_rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, bootstrap=False, max_depth=10, n_jobs=3, random_state=None, verbose=0)
    model_rf.fit(Xtrain_rf, ytrain_rf)
    toc = time()
    print 'fitted rf model in', toc - tic

    dense_points = gen_dense_points(227, 227)

    # load whole and bbox and head part data
    # load data
    tic = time()
    features_storage_r = datastore(settings.storage('ccrft2st-10000'))
    feature_extractor_r = CNN_Features_CAFFE_REFERENCE(features_storage_r, make_net=False)

    features_storage_c = datastore(settings.storage('ccrft2st-2500'))
    feature_extractor_c = CNN_Features_CAFFE_REFERENCE(features_storage_c, make_net=False)

    features_storage_p_h = datastore(settings.storage('ccpheadft-100000'))
    feature_extractor_p_h = CNN_Features_CAFFE_REFERENCE(features_storage_p_h, make_net=False)

    Xtrain_r, ytrain_r, Xtest_r, ytest_r = cub.get_train_test(feature_extractor_r.extract_one)
    Xtrain_c, ytrain_c, Xtest_c, ytest_c = cub.get_train_test(feature_extractor_c.extract_one)
    Xtrain_p_h, ytrain_p_h, Xtest_p_h, ytest_p_h = cub.get_train_test(feature_extractor_p_h.extract_one)

    toc = time()
    print 'loaded whole and bbox and head part data in', toc - tic

    def compute_estimated_part_data(model_name, shape, IDS, model_rf):
        net = caffe.Classifier(settings.model(model_name), settings.pretrained(model_name), mean=np.load(settings.ILSVRC_MEAN), channel_swap=(2, 1, 0), raw_scale=255)
        net.set_phase_test()
        net.set_mode_gpu()
        # compute estimated head data
        new_Xtest_part = np.zeros(shape)

        for i, t_id in enumerate(IDS):
            print i
            img = caffe.io.load_image(all_image_infos[t_id])
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

            t_img_part = Parts().get_rect(img, rect_info=rect_info)
            try:
                net.predict([t_img_part], oversample=False)
            except Exception:
                print '------', t_id, '----------'

            new_Xtest_part[i, :] = net.blobs[feat_layer].data[0].flatten()

        return new_Xtest_part

    tic = time()
    if rf_safe.check_exists_large(Xtest_ip) and not force:
        print 'loading test'
        Xtest_p_h = rf_safe.load_large_instance(Xtest_ip, instance_split)
    else:
        print 'calculating test'
        Xtest_p_h = compute_estimated_part_data('ccpheadrfft-100000', Xtest_p_h.shape, IDtest, model_rf)

        rf_safe.save_large_instance(Xtest_ip, Xtest_p_h, instance_split)

    if recalculate_training:
        if rf_safe.check_exists_large(Xtrain_ip) and not force:
            print 'loading train'
            Xtrain_p_h = rf_safe.load_large_instance(Xtrain_ip, instance_split)
        else:
            print 'calculating train'
            Xtrain_p_h = compute_estimated_part_data('ccpheadrfft-100000', Xtrain_p_h.shape, IDtrain, model_rf)

            rf_safe.save_large_instance(Xtrain_ip, Xtrain_p_h, instance_split)

    toc = time()
    print 'features loaded or calculated in', toc - tic

    Xtrain = np.concatenate((Xtrain_r, Xtrain_c, Xtrain_p_h), axis=1)
    Xtest = np.concatenate((Xtest_r, Xtest_c, Xtest_p_h), axis=1)
    ytrain = ytrain_r
    ytest = ytest_r

    print Xtrain.shape, Xtest.shape

    # do classification
    tic = time()
    model = sklearn.svm.LinearSVC(C=C)
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtest)
    toc = time() - tic

    print 'classification in', toc
    print '--------------------'
    print 'C:', C
    print '--------------------'
    print 'accuracy', sklearn.metrics.accuracy_score(ytest, predictions), 'mean accuracy', utils.mean_accuracy(ytest, predictions)
    print '===================='


if __name__ == '__main__':
    main()
