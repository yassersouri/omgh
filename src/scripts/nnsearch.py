import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
sys.path.append('/home/ipl/installs/caffe-rc/python/')
import caffe
import settings
import utils
from storage import datastore
from dataset import CUB_200_2011
import click


@click.command()
def main():
    storage_name = 'cache-cccftt'
    layer = 'pool5'
    name = '%s-%s' % ('cccftt', 100000)

    cub = CUB_200_2011(settings.CUB_ROOT)

    safe = datastore(settings.storage(storage_name))
    safe.super_name = 'features'
    safe.sub_name = name

    instance_path = safe.get_instance_path(safe.super_name, safe.sub_name, 'feat_cache_%s' % layer)
    feat = safe.load_large_instance(instance_path, 4)
    print feat.shape


if __name__ == '__main__':
    main()