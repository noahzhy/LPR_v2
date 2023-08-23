import os
import glob
import shutil
import random


# list all images in the directory
def list_images(path, use_shuffle=True):
    # return the list of all image paths in the dataset
    image_paths = sorted(list(glob.glob(os.path.join(path, "*.jpg"))))
    if use_shuffle:
        random.shuffle(image_paths)
    return image_paths


# split dataset into train and test
def split_dataset(path, train_size=0.8):
    # get all image paths
    image_paths = list_images(path)
    # get train size
    train_num = int(len(image_paths) * train_size)
    # split dataset
    train_paths = image_paths[:train_num]
    test_paths = image_paths[train_num:]
    # create train and test dir
    train_dir = os.path.join(path, "train_v2")
    test_dir = os.path.join(path, "test_v2")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    # move images to train dir
    for train_path in train_paths:
        shutil.move(train_path, train_dir)
    # move images to test dir
    for test_path in test_paths:
        shutil.move(test_path, test_dir)


# main
if __name__ == '__main__':
    # # split dataset
    # split_dataset("data")
    # print("Done!")
    # list all images
    image_paths = list_images("data/test")
    # find same name mask end with png
    for image_path in image_paths:
        mask_path = image_path.replace(".jpg", ".png")
        # remove 'train' string in path
        mask_path = mask_path.replace("test", "")
        if os.path.exists(mask_path):
            shutil.move(mask_path, "data/test")
    print("Done!")
