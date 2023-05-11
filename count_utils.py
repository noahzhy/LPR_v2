import os
import glob
import numpy as np
from PIL import Image


# list all images in a directory, (.png, .jpg)
def list_images(dir_path):
    images = []
    for ext in ('*.png', '*.jpg'):
        images.extend(glob.glob(os.path.join(dir_path, ext)))

    # count the number of images which name with space character
    count = 0
    for img in images:
        if ' ' in img:
            count += 1

    # print total number of images and number of images which name with space character
    print(f"Total number of images: {len(images)}")
    print(f"Number of images which name with space character: {count}")
    return images


# main
if __name__ == '__main__':
    list_images('train')