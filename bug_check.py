import os
import glob
import shutil


# list all .jpg files in data folder
def list_all_files(path):
    imgs = glob.glob(path + '/*.jpg')
    # get only file name split by '_' and get first element
    labels = [os.path.basename(img).split('_')[0] for img in imgs]
    # remove space in labels
    labels = [label.replace(' ', '') for label in labels]
    # pick length which is bigger than 8
    labels = [label for label in labels if len(label) > 8]
    for label in labels:
        print(label)


# main
if __name__ == '__main__':
    list_all_files(path='test')
