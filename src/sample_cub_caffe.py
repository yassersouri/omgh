from dataset import CUB_200_2011
import cv2


cub = CUB_200_2011(
    '/Users/yasser/sharif-repo/Datasets/CUB-200-2011/CUB_200_2011/CUB_200_2011')

for d in cub.get_all_images():
    cv2.imshow('t', cv2.imread(d['img_file']))
    key = cv2.waitKey(50)
    if key == 27:
        break
