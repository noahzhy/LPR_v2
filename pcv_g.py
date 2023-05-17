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
def find_masks(img, debug=False):
    origin_img = img.copy()

    # # to rgb
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 增强对比度
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # blur the image
    # img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    # hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # normalize the image
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


    # Define the ranges for green and white in HSV color space
    color_lower = (00, 00, 180)
    color_upper = (255, 100, 255)

    # Create masks for green and white regions
    color_mask = cv2.inRange(img, color_lower, color_upper)

    return color_mask

# list of all img files under the dir_path
def get_img_list(dir_path):
    img_list = []
    for ext in ['jpg', 'jpeg']:
        img_list.extend(glob.glob(os.path.join(dir_path, '*.{}'.format(ext))))
    return img_list


# show the img and mask from given img_path list, pick 9 pairs randomly
def show_imgs(img_path_list, debug=False):
    # shuffle the img_path_list
    random.shuffle(img_path_list)
    # pick 9 pairs randomly
    hstack_list = []
    for img_path in img_path_list[:9]:
        print(img_path)
        img = cv_imread(img_path)
        mask = find_masks(img, debug=debug)
        # keep mask to 3 channels
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        hstack_list.append(np.hstack((img, mask)))
    # show the 9 pairs in 3 rows and 3 cols
    plt.figure(figsize=(12, 6))
    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, i * 3 + j + 1)
            plt.imshow(hstack_list[i * 3 + j])
            plt.xticks([])
            plt.yticks([])
    plt.show()


# convert origin img to jpg mode and save it, and save the same name as mask in png mode
def save_imgs(img_path_list):
    # shuffle the img_path_list
    random.shuffle(img_path_list)
    for img_path in img_path_list:
        print(img_path)
        # save as jpg mode if the img is not jpg mode
        if img_path[-4:] != '.jpg':
            # save the origin img to jpg mode via PIL
            Image.open(img_path).convert('RGB').save(img_path[:-4] + '.jpg')

        img = cv_imread(img_path)
        mask = find_masks(img)
        # save the mask to png mode via PIL
        Image.fromarray(mask).convert('L').save(img_path[:-4] + '.png')
        # break

# main
if __name__ == '__main__':
    # get the img_list
    dir_path = 'aihub\double\green'
    img_list = get_img_list(dir_path)
    save_imgs(img_path_list=img_list)
    # show_imgs(img_path_list=img_list, debug=True)
