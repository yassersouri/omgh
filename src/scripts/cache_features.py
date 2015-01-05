import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
sys.path.append('/home/ipl/installs/caffe-rc/python/')
import caffe
import settings
from storage import datastore
from dataset import CUB_200_2011
import click


@click.command()
@click.argument('model-name', type=str)
@click.argument('iteration', type=int)
@click.argument('storage-name', type=str)
def main(model_name, iteration, storage_name):
    name = '%s-%s' % (model_name, iteration)
    print settings.model(name), settings.pretrained(name)

    safe = datastore(settings.storage(storage_name))
    safe.super_name = 'features'
    safe.sub_name = name

    layer_names = ['fc7', 'fc6', 'pool5', 'conv5', 'conv4', 'conv3']
    layer_dims = [4096,   4096,  9216,    43264,   64896,   64896]

    net = caffe.Classifier(settings.model(name), settings.pretrained(name), mean=np.load(settings.ILSVRC_MEAN), channel_swap=(2, 1, 0), raw_scale=255)
    net.set_mode_gpu()
    net.set_phase_test()

    cub = CUB_200_2011(settings.CUB_ROOT)

    dataset_size = sum(1 for _ in cub.get_all_images())

    instance = {}
    for layer, dim in zip(layer_names, layer_dims):
        instance[layer] = np.zeros((dataset_size, dim))
        print instance[layer].shape

    for i, info in enumerate(cub.get_all_images(cropped=True)):
        print info['img_id']
        img = caffe.io.load_image(info['img_file'])
        net.predict([img], oversample=False)
        for layer in layer_names:
            instance[layer][i, :] = net.blobs[layer].data[0].flatten()

    for layer in layer_names:
        safe.save_large_instance(safe.get_instance_path(safe.super_name, safe.sub_name, 'feat_cache_%s' % layer), instance[layer], 4)


if __name__ == '__main__':
    main()
