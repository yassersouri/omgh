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
@click.option('--part', type=click.Choice(['head', 'body']), default='body')
@click.option('--random-state', type=click.INT, default=313)
@click.option('--pgs', type=click.Choice(['unif', 'rand', 'norm']), default='unif')
def main(out_path, part, random_state, pgs):
    cub = CUB_200_2011(settings.CUB_ROOT)
    lfrg = rects.BerkeleyRG(settings.BERKELEY_ANNOTATION_BASE_PATH, cub, part)
    RG = rects.RandomForestRG(datastore(settings.storage('rf')), lfrg, cub_utils.DeepHelper.get_custom_net(settings.model('cccftt-60000'), settings.pretrained('cccftt-60000')), 'caffenet', cub, random_state=random_state, point_gen_strategy=pgs, use_seg=True, pt_n_part=20, pt_n_bg=100)
    RG.setup()

    for i, image in enumerate(cub.get_all_images()):
        print i
        image_path = image['img_file']
        img_id = int(image['img_id'])
        rel_image_path = image['img_file_rel']

        o_image = cv2.imread(image_path)
        rect = RG.generate(img_id)
        t_img_part = rect.get_rect(o_image)

        out_image_path = os.path.join(out_path, rel_image_path)
        utils.ensure_dir(os.path.dirname(out_image_path))
        cv2.imwrite(out_image_path, t_img_part)


if __name__ == '__main__':
    main()
