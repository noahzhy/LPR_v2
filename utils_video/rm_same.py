# remove 100% same images
import os
import sys
import glob

import cv2
import numpy as np
from PIL import *


# 感知哈希算法
def dhash(image, hash_size=4):
    # 缩放尺寸和灰度转换
    image = cv2.resize(image, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算横向相邻像素差值
    diff = gray[:, 1:] > gray[:, :-1]
    # 转换为hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


# function to remove 90% same images
def rm_same_images(images_root):
    ready_to_remove = []
    # list all images in given dir
    images = glob.glob(os.path.join(images_root, '*.jpg'))
    # list all images to store hash in map
    images_hash = {}
    for image in images:
        img = cv2.imread(image)
        # hash
        image_hash = dhash(img)
        # store hash in map
        if image_hash in images_hash:
            images_hash[image_hash].append(image)
        else:
            images_hash[image_hash] = [image]
        
    # remove 90% same image
    for key in images_hash:
        if len(images_hash[key]) > 1:
            ready_to_remove.extend(images_hash[key][1:])

    # remove
    for image in ready_to_remove:
        os.remove(image)


# main
if __name__ == '__main__':
    path = r"D:\road_test_video\frames"
    rm_same_images(path)

