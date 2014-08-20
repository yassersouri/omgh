from dataset import PASCAL_VOC_2006


d = PASCAL_VOC_2006('/Users/yasser/sharif-repo/Datasets/VOCdevkit/VOC2006')
for I in d.get_set('train', 'person'):
    print I['img']
    print len(I['objects'])