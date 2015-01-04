import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import settings
from storage import datastore
from dataset import CUB_200_2011
import click
import sklearn.neighbors
from time import time


@click.command()
def main():
    storage_name = 'cache-cccftt'
    layer = 'pool5'
    name = '%s-%s' % ('cccftt', 100000)
    normalize_feat = True
    n_neighbors = 1

    A = 100
    N = 100

    cub = CUB_200_2011(settings.CUB_ROOT)

    safe = datastore(settings.storage(storage_name))
    safe.super_name = 'features'
    safe.sub_name = name

    instance_path = safe.get_instance_path(safe.super_name, safe.sub_name, 'feat_cache_%s' % layer)
    feat = safe.load_large_instance(instance_path, 4)

    # should we normalize the feats?
    if normalize_feat:
        # snippit from : http://stackoverflow.com/a/8904762/428321
        row_sums = feat.sum(axis=1)
        new_feat = feat / row_sums[:, np.newaxis]
        feat = new_feat

    IDtrain, IDtest = cub.get_train_test_id()
    # bbox = cub.get_bbox()
    # parts = cub.get_parts()

    # the following line is not really a good idea. Only works for this dataset.
    Xtrain = feat[IDtrain-1, :]
    Xtest = feat[IDtest-1, :]

    nn_model = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric='minkowski', p=2)
    tic = time()
    nn_model.fit(Xtrain)
    toc = time() - tic
    print 'fitted in: ', toc

    print 'test ids'
    print ', '.join([str(x) for x in IDtest[A:A+N]])

    tic = time()
    NNS = nn_model.kneighbors(Xtest[A:A+N], 1, return_distance=False).T
    toc = time() - tic
    print 'found in: ', toc
    print 'train ids'
    print ', '.join([str(x) for x in NNS[0]])


if __name__ == '__main__':
    main()
