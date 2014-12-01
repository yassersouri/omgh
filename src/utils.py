import numpy
import sklearn.metrics
import os


def mean_accuracy(groundtruth, predictions):
    groundtruth_cm = sklearn.metrics.confusion_matrix(groundtruth, groundtruth).astype(numpy.float32)
    predictions_cm = sklearn.metrics.confusion_matrix(predictions, groundtruth).astype(numpy.float32)
    return numpy.mean(numpy.diag(predictions_cm) / numpy.diag(groundtruth_cm))


def ensure_dir(address):
    if not os.path.exists(address):
        os.makedirs(address)
