import os
import cv2 as cv
from consts import Mask_Alg
from cell_detection_v2 import get_cells

def get_all_masks(path_to_images: str, path_to_masks: str):

    os.makedirs(path_to_masks)
    images_names = os.listdir(path_to_images)

    for name in images_names:
        image = cv.imread(path_to_images + '/' + name, 0)
        image_mask = get_cells(image, Mask_Alg.BLOB_DETECT)[0]
        cv.imwrite(path_to_masks + '/' + name, image_mask)

def main():
    path_to_images = 'valid_bf_imgs_dir_png'
    path_to_masks = 'valid_bf_imgs_masks_dir_png'

    get_all_masks(path_to_images, path_to_masks)

if __name__=='__main__':
    main()