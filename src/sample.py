from dataset import PASCAL_VOC_2006
from storage import datastore
from extractor import SIFT_SIFT_Extractor
from transforms import PCA_Transform


voc2006 = PASCAL_VOC_2006('/Users/yasser/sharif-repo/Datasets/VOCdevkit/VOC2006')
temp_storage = datastore('/Users/yasser/datastores/toobreh/exp1')
feature_extractor = SIFT_SIFT_Extractor(temp_storage)

# just precompute the keypoints one
# for t, des in feature_extractor.extract(voc2006, 'train'):
#     print t['img_id']
# for t, des in feature_extractor.extract(voc2006, 'test'):
#     print t['img_id']

pca_t = PCA_Transform(temp_storage, 50)
pca_t.fit(feature_extractor.extract(voc2006, 'train'))
