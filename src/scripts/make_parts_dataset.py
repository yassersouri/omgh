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
from dataset import CUB_200_2011
from parts import Parts


@click.command()
@click.argument('out-path', type=click.Path(exists=True))
def main(out_path):
    cub = CUB_200_2011(settings.CUB_ROOT)
    cub_images = cub.get_all_images()
    for image in cub_images:
        image_path = image['img_file']
        image_id = image['img_id']
        cub_parts = cub.get_parts()

        rel_image_path = image_path[len(settings.CUB_IMAGES_FOLDER):]
        o_image = caffe.io.load_image(image_path)

        parts = cub_parts.for_image(image_id)
        head_parts = parts.filter_by_name(Parts.HEAD_PART_NAMES)

        if len(head_parts) <= 2:
            print "#parts:%d \tID:%d \tName:%s" % (len(head_parts), int(image_id), rel_image_path)
            if len(head_parts) <= 1:
                continue

        part_image = head_parts.get_gray_out_rect(o_image)
        if 0 in part_image.shape:
            print "#parts:%d \tID:%d \tName:%s + Shape:%s" % (len(head_parts), int(image_id), rel_image_path, str(part_image.shape))
            # continue
        
        out_image_path = os.path.join(out_path, rel_image_path)
        utils.ensure_dir(os.path.dirname(out_image_path))
        skimage.io.imsave(out_image_path, part_image)


if __name__ == '__main__':
    main()