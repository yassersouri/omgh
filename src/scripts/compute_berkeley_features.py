import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import settings
from dataset import CUB_200_2011
from storage import datastore
from deep_extractor import Berkeley_Extractor
import pyprind
import click
import cub_utils


@click.command()
@click.option('--force', type=click.BOOL, default=False)
def main(force):
    cub_head = CUB_200_2011(settings.CUB_ROOT, images_folder_name='images_b_head')
    cub_body = CUB_200_2011(settings.CUB_ROOT, images_folder_name='images_b_body')
    cub_crop = CUB_200_2011(settings.CUB_ROOT, images_folder_name='images_b_cropped')

    st_head = datastore(settings.storage('bmbh'))
    st_body = datastore(settings.storage('bmbb'))
    st_crop = datastore(settings.storage('bmbcflp'))

    ext_head = Berkeley_Extractor(st_head, pretrained_file=settings.BERKELEY_HEAD_PRET)
    ext_body = Berkeley_Extractor(st_body, pretrained_file=settings.BERKELEY_BODY_PRET)
    ext_crop = Berkeley_Extractor(st_crop, pretrained_file=settings.BERKELEY_CROP_PRET)

    number_of_images_in_dataset = sum(1 for _ in cub_crop.get_all_images())

    bar = pyprind.ProgBar(number_of_images_in_dataset, width=80)
    for t, des in ext_crop.extract_all(cub_crop.get_all_images(), flip=True, force=force):
        bar.update()
    print 'DONE CROP'

    bar = pyprind.ProgBar(number_of_images_in_dataset, width=80)
    for t, des in ext_head.extract_all(cub_head.get_all_images(), force=force):
        bar.update()
    print 'DONE HEAD'

    bar = pyprind.ProgBar(number_of_images_in_dataset, width=80)
    for t, des in ext_body.extract_all(cub_body.get_all_images(), force=force):
        bar.update()
    print 'DONE BODY'

if __name__ == '__main__':
    main()
