import numpy
import sklearn.metrics


def mean_accuracy(groundtruth, predictions):
    groundtruth_cm = sklearn.metrics.confusion_matrix(groundtruth, groundtruth).astype(numpy.float32)
    predictions_cm = sklearn.metrics.confusion_matrix(predictions, groundtruth).astype(numpy.float32)
    return numpy.mean(numpy.diag(predictions_cm) / numpy.diag(groundtruth_cm))