import cv2
import click
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)
import caffe


def vis_square(ax, data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    ax.imshow(data)


@click.command()
@click.option('--vid-file', '-f', type=click.Path(exists=True), default='/home/ipl/Desktop/yaser-face2.mov')
@click.option('--layer', '-l', default='conv5')
@click.option('--index', '-i', type=click.INT, default=100)
def main(vid_file, layer, index):
    cap = cv2.VideoCapture()
    cap.open(vid_file)

    net = caffe.Classifier('/home/ipl/installs/caffe-rc/models/bvlc_reference_caffenet/deploy.prototxt', '/home/ipl/installs/caffe-rc/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel', mean=np.load('/home/ipl/installs/caffe-rc/python/caffe/imagenet/ilsvrc_2012_mean.npy'), channel_swap=(2, 1, 0), raw_scale=255)
    net.set_phase_test()
    net.set_mode_gpu()

    fig_img = plt.figure(figsize=(15, 10))
    ax_img = fig_img.add_subplot(121)

    ax_feat = fig_img.add_subplot(122)
    fig_img.show()

    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        net.predict([frame], oversample=False)
        feat = net.blobs[layer].data[0, :, :, :]

        ax_img.imshow(frame)
        # ax_feat.matshow(feat)
        vis_square(ax_feat, feat, padval=1)

        plt.draw()

        key = cv2.waitKey(1)
        if key >= 30:
            break

    cap.release()


if __name__ == '__main__':
    main()
