import os
import random
import glob



# function to split data into train and test

def split_data(dir_path, train_ratio=0.9):
    # get all images
    images = glob.glob(dir_path + '/*.*')
    # shuffle images
    random.shuffle(images)
    # split into train and test
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    test_images = images[train_size:]
    return train_images, test_images


# move images to train and test folders
def move_images(images):
    target_folder = 'E:/projects/LPR_v2/test'
    for image in images:
        # move image
        os.rename(image, target_folder + '/' + os.path.basename(image))


if __name__ == '__main__':
    # # split data
    # train_images, test_images = split_data('E:/projects/LPR_v2/data', train_ratio=0.9)
    # # move images
    # move_images(test_images)

    # list all images in data folder
    images = glob.glob('E:/projects/LPR_v2/data/*.*')
    for image in images:
        # which name is included '물' in image name
        if '물' in image:
            # move image
            print(image)
    
