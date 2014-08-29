from dataset import PASCAL_VOC_2006
from storage import datastore
from extractor import SIFT_SIFT_Extractor
from transforms import PCA_Transform, GMMUniversalVocabulary
from datatime import datatime as dt


COMPLETE = False
CALC_FEATURES = True
CALC_PCA_TRANSFORMS = True

voc2006 = PASCAL_VOC_2006(
    '/Users/yasser/sharif-repo/Datasets/VOCdevkit/VOC2006')
features_storage = datastore('/Users/yasser/datastores/toobreh/features')
feature_extractor = SIFT_SIFT_Extractor(features_storage)

if COMPLETE or CALC_FEATURES:
    a = dt.now()
    for t, des in feature_extractor.extract(voc2006, 'train'):
        print t['img_id']
    for t, des in feature_extractor.extract(voc2006, 'test'):
        print t['img_id']
    b = dt.now()
    print 'extracting and saving all features: \t', (b - a)

transforms_storage = datastore(
    '/Users/yasser/datastores/toobreh/exp1/transforms')
pca_t = PCA_Transform(transforms_storage, 50)
a = dt.now()
pca_t.fit(feature_extractor.extract(voc2006, 'train'))
b = dt.now()
print 'fitting pca: \t', (b - a)


if COMPLETE or CALC_PCA_TRANSFORMS:
    a = dt.now()
    for t, des in pca_t.transform(feature_extractor.extract(voc2006, 'train')):
        print t['img_id']
    for t, des in pca_t.transform(feature_extractor.extract(voc2006, 'test')):
        print t['img_id']
    b = dt.now()
    print 'calculating all pcas: \t', (b - a)

uni_vocab = GMMUniversalVocabulary(
    transforms_storage, n_components=1, covariance_type='diag',
    n_iter=1, n_init=1)
a = dt.now()
uni_vocab.fit(
    pca_t.transform(feature_extractor.extract(voc2006, 'train')), test=True)
b = dt.now()
print 'fitting gmm: \t', (b - a)

if COMPLETE:
    for t, des in uni_vocab.transform(pca_t.transform(
            feature_extractor.extract(voc2006, 'train')), force=True):
        print t['img_id']
    for t, des in uni_vocab.transform(pca_t.transform(
            feature_extractor.extract(voc2006, 'test')), force=True):
        print t['img_id']
