from dataset import CUB_200_2011
from storage import datastore
from hog_extractor import HOG

cub = CUB_200_2011(
    '/home/ipl/datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011')
features_storage = datastore('/home/ipl/datastores/hog_features_cropped_normalized/')

feature_extractor = HOG(features_storage)

for t, des in feature_extractor.extract(cub.get_all_images(), cub.get_bbox()):
    print t['img_id']
    print des.shape
