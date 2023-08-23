import os
import glob
import numpy as np
from PIL import Image


# list all images in a directory, (.png, .jpg) as pairs
def list_all_images(dir_path):
    images = []
    for ext in ['jpg']:
        images.extend(glob.glob(os.path.join(dir_path, '*.{}'.format(ext))))
    return images


# divide the images into train and test
def divide_train_test(images, train_ratio=0.9):
    # shuffle the images
    np.random.shuffle(images)
    # divide the images into train and test
    train_images = images[:int(len(images) * train_ratio)]
    test_images = images[int(len(images) * train_ratio):]
    return train_images, test_images


# move divided images to train and test dir
def move_images(images, dst_dir):
    for image in images:
        # move
        os.rename(image, os.path.join(dst_dir, os.path.basename(image)))
        # find the same name png file
        png_file = image[:-4] + '.png'
        if os.path.exists(png_file):
            os.rename(png_file, os.path.join(dst_dir, os.path.basename(png_file)))


def divide_train_test(root_dir):
    # list all images in a directory, (.png, .jpg) as pairs
    images = list_all_images('total')
    # divide the images into train and test
    train_images, test_images = divide_train_test(images, train_ratio=0.9)
    # move divided images to train and test dir
    move_images(train_images, 'train')
    move_images(test_images, 'test')


def count_dataset_single_and_double_ratio(dir_path):
    # list all images in a directory, (.png, .jpg) as pairs
    images = list_all_images(dir_path)
    # if name contains space, it is double
    double_images = [x for x in images if x.find(' ') != -1]
    # single images
    single_images = [x for x in images if x.find(' ') == -1]

    print('total images: ', len(images))
    print('double images: ', len(double_images))
    print('single images: ', len(single_images))


# main 
if __name__ == '__main__':
    # divide the images into train and test
    # divide_train_test('total')

    count_dataset_single_and_double_ratio('data')
