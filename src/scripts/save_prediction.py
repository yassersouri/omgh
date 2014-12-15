import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import CUB_200_2011, CUB_200_2011_Segmented
from storage import datastore
from deep_extractor import CNN_Features_CAFFE_REFERENCE
import settings
import utils
import click
from sklearn import svm
from sklearn.metrics import accuracy_score


@click.command()
@click.argument('sname')
@click.option('--svm-c', type=click.FLOAT, default=0.0001)
@click.option('--segmented', type=click.BOOL, default=False)
def main(sname, svm_c, segmented):
    if segmented:
        cub = CUB_200_2011_Segmented(settings.CUB_ROOT)
    else:
        cub = CUB_200_2011(settings.CUB_ROOT)
    ft_storage = datastore(settings.storage(sname))
    ft_extractor = CNN_Features_CAFFE_REFERENCE(ft_storage, make_net=False)

    Xtrain, ytrain, Xtest, ytest = cub.get_train_test(ft_extractor.extract_one)
    model = svm.LinearSVC(C=svm_c)
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtest)

    print 'accuracy', accuracy_score(ytest, predictions)
    print 'mean accuracy', utils.mean_accuracy(ytest, predictions)

    pred_storage = datastore(settings.PREDICTIONS_BASE, global_key='preds')
    storage_path = pred_storage.get_instance_path('preds', sname, '%s.mat' % sname)
    pred_storage.ensure_dir(os.path.dirname(storage_path))
    pred_storage.save_instance(storage_path, predictions)

if __name__ == '__main__':
    main()
