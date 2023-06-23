# list all images in a directory
import glob
import os
import shutil
import numpy as np
import random


# list all images in a directory
def list_images(dir_path):
    return glob.glob(dir_path + "/*.jpg")



# function to rename all images in a directory
# input: directory path
# output: None
def rename_images(dir_path):
    images = list_images(dir_path)
    for i, image in enumerate(images):
        # split via '_' and keep the first part
        label = os.path.basename(image).split('_')[0]
        # random 8 digits
        random = str(np.random.randint(10000000, 99999999))
        # rename
        os.rename(image, dir_path + '/' + label + '_' + random + '.jpg')

# main
if __name__ == '__main__':
    rename_images('road')
