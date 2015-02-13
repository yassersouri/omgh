import numpy
import sklearn.metrics
import os
import cv2
import numpy as np


def mean_accuracy(groundtruth, predictions):
    groundtruth_cm = sklearn.metrics.confusion_matrix(groundtruth, groundtruth).astype(numpy.float32)
    predictions_cm = sklearn.metrics.confusion_matrix(predictions, groundtruth).astype(numpy.float32)
    return numpy.mean(numpy.diag(predictions_cm) / numpy.diag(groundtruth_cm))


def ensure_dir(address):
    if not os.path.exists(address):
        os.makedirs(address)


def draw_bbox(img, bbox, color=100, width=2):
    try:
        bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    except:
        bx, by, bw, bh = bbox
    new_img = img.copy()
    cv2.rectangle(new_img, (bx, by), (bx+bw, by+bh), color, width)
    return new_img


def get_rect(img, rect_info):
    xmin, xmax, ymin, ymax = rect_info
    return img[xmin:xmax, ymin:ymax]


def get_rect_from_bbox(img, bbox):
    by, bx, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    return img[bx:bx+bh, by:by+bw]


def l2_feat_norm(feat):
    row_norms = np.linalg.norm(feat, axis=1)
    new_feat = feat / row_norms[:, np.newaxis]
    return new_feat
