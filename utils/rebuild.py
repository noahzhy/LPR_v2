import glob
import os
import shutil
import sys


# list all images in given dir
def list_images(dir_path):
    return glob.glob(os.path.join(dir_path, '*.jpg'))


# pick up labels from image path
def get_labels(imgs):
    labels = []
    for img in imgs:
        label = os.path.basename(img).split('.')[0].split('_')[0]
        labels.append(label)
    return labels


# main  
if __name__ == '__main__':
    # get path
    path = ""
    _imgs = list_images(path)

