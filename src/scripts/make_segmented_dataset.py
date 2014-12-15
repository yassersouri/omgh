import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)

import skimage.io
import caffe
import numpy as np
import click
from glob import glob
import utils


@click.command()
@click.option('--cub-path', prompt=True, type=click.Path(exists=True), default='/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011/images/')
@click.option('--seg-path', prompt=True, type=click.Path(exists=True), default='/home/ipl/datasets/CUB-200-2011/segmentations/')
@click.option('--out-path', prompt=True, type=click.Path(exists=True))
def main(cub_path, seg_path, out_path):
    cub_images = glob(os.path.join(cub_path, '*/*.jpg'))
    for image_path in cub_images:
        rel_image_path = image_path[len(cub_path):]
        print rel_image_path
        o_image = caffe.io.load_image(image_path)

        # I know, I had to change the extention from jpg to png!
        seg_image_path = os.path.join(seg_path, rel_image_path)[:-3] + 'png'

        s_image = caffe.io.load_image(seg_image_path)

        # The following line could be improved when we can use the imagenet
        # mean file
        background = np.ones_like(s_image) * 0.5
        threshold = np.median(s_image) # one could use anything else, like max or min!

        so_image = np.where(s_image <= threshold, background, o_image)
        out_image_path = os.path.join(out_path, rel_image_path)
        utils.ensure_dir(os.path.dirname(out_image_path))
        skimage.io.imsave(out_image_path, so_image)









if __name__ == '__main__':
    main()