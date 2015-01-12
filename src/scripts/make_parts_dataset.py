import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
sys.path.append(settings.CAFFE_PYTHON_PATH)
import click
import utils
from dataset import CUB_200_2011
from parts import Parts
import cv2


@click.command()
@click.argument('out-path', type=click.Path(exists=True))
def main(out_path):
    cub = CUB_200_2011(settings.CUB_ROOT)
    cub_images = cub.get_all_images()
    cub_parts = cub.get_parts()
    for i, image in enumerate(cub_images):
        image_path = image['img_file']
        image_id = image['img_id']

        rel_image_path = image_path[len(settings.CUB_IMAGES_FOLDER):]
        o_image = cv2.imread(image_path)

        parts = cub_parts.for_image(image_id)
        p_parts = parts.filter_by_name(Parts.BODY_PART_NAMES)

        if len(p_parts) <= 2:
            print "#parts:%d \tID:%d \tName:%s" % (len(p_parts), int(image_id), rel_image_path)

        part_image = p_parts.get_rect(o_image, alpha=0.6)

        out_image_path = os.path.join(out_path, rel_image_path)
        utils.ensure_dir(os.path.dirname(out_image_path))
        cv2.imwrite(out_image_path, part_image)


if __name__ == '__main__':
    main()
