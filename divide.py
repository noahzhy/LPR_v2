import os
import cv2
import numpy as np

# Get the path of the folder containing the image files
folder_path = os.path.join(os.getcwd(), "double_train")

# Create two empty lists to store the green and yellow image files
green_images = []
yellow_images = []


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img


def cv_imwrite(path, image):
    # Get the current encoding
    encoding = sys.getfilesystemencoding()
    # Convert the path to the specified encoding
    path = path.encode(encoding)
    # Save the image to the path
    cv2.imwrite(path, image)


def is_green_yellow(image_path):
    image = cv_imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the ranges for green and yellow in HSV color space
    green_lower = (36, 25, 25)
    green_upper = (86, 255, 255)
    yellow_lower = (15, 25, 25)
    yellow_upper = (35, 255, 255)

    # Create masks for green and yellow regions
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Count the number of green and yellow pixels
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # Determine if the image is predominantly green or yellow
    if green_pixels > yellow_pixels:
        return 'green'
    else:
        return 'yellow'


def divide_green_yellow_images(folder_path):
    # Create folders for green and yellow images
    green_folder = os.path.join(folder_path, 'green')
    yellow_folder = os.path.join(folder_path, 'yellow')

    if not os.path.exists(green_folder):
        os.makedirs(green_folder)

    if not os.path.exists(yellow_folder):
        os.makedirs(yellow_folder)

    # Iterate over the image files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            color = is_green_yellow(image_path)

            # Move the image file to the respective folder
            if color == 'green':
                new_path = os.path.join(green_folder, filename)
            else:
                new_path = os.path.join(yellow_folder, filename)

            os.rename(image_path, new_path)
            print(f"Moved {filename} to {color} folder.")


# Specify the folder path containing the image files
folder_path = 'double'

# Divide the green and yellow images in the folder
divide_green_yellow_images(folder_path)
