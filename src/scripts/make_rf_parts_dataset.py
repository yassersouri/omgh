import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
import click
import cub_utils
import utils
from dataset import CUB_200_2011
import rects
import cv2
from storage import datastore


@click.command()
@click.argument('out-path', type=click.Path(exists=True))
def main(out_path):
    cub = CUB_200_2011(settings.CUB_ROOT)
    BRGh = rects.BerkeleyRG(settings.BERKELEY_ANNOTATION_BASE_PATH, cub, 'head')
    RFRGh = rects.RandomForestRG(datastore(settings.storage('rf')), BRGh, cub_utils.DeepHelper.get_bvlc_net(), 'caffenet', cub, random_state=313, point_gen_strategy='unif', use_seg=True, pt_n_part=20, pt_n_bg=100)

    for i, image in enumerate(cub.get_all_images()):
        print i
        image_path = image['img_file']
        img_id = int(image['img_id'])
        rel_image_path = image['img_file_rel']

        o_image = cv2.imread(image_path)
        rect = RFRGh.generate(img_id)
        t_img_part = rect.get_rect(o_image)

        out_image_path = os.path.join(out_path, rel_image_path)
        utils.ensure_dir(os.path.dirname(out_image_path))
        cv2.imwrite(out_image_path, t_img_part)


if __name__ == '__main__':
    main()
