# using pure cv method to find the license plate letters
import os
import math
import glob
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img


# fun to find mask of image
def find_masks(img):
    origin_img = img.copy()

    # convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # blur the image
    img = cv2.GaussianBlur(img, (5, 5), 0, 0)
    # mask
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    # # show the origin image and the mask
    # plt.subplot(121)
    # # to rgb
    # origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(origin_img, cmap='gray')
    # plt.title('origin')
    # plt.subplot(122)
    # plt.imshow(img, cmap='gray')
    # plt.title('mask')
    # plt.show()
    return img


# list of all img files under the dir_path
def get_img_list(dir_path):
    img_list = []
    for ext in ['jpg', 'jpeg']:
        img_list.extend(glob.glob(os.path.join(dir_path, '*.{}'.format(ext))))
    return img_list


# convert origin img to jpg mode and save it, and save the same name as mask in png mode
def save_imgs(img_path_list):
    for img_path in img_path_list:
        print(img_path)
        # save as jpg mode if the img is not jpg mode
        if img_path[-4:] != '.jpg':
            # save the origin img to jpg mode via PIL
            Image.open(img_path).convert('RGB').save(img_path[:-4] + '.jpg')

        if img_path[-4:] == '.png':
            continue

        img = cv_imread(img_path)
        mask = find_masks(img)
        # save the mask to png mode via PIL
        Image.fromarray(mask).convert('L').save(img_path[:-4] + '.png')
        # break

# main
if __name__ == '__main__':
    # get the img_list
    dir_path = 'train'
    img_list = get_img_list(dir_path)
    save_imgs(img_path_list=img_list)
