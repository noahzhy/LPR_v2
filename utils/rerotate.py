import os
import cv2
import numpy as np
import glob
import time
import shutil
import random


def cv2_imread(path):
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return cv_img


def cv2_imwrite(path, img):
    cv2.imencode('.jpg', img)[1].tofile(path)


# list all images in the directory
def list_images(path, use_shuffle=True):
    # return the list of all image paths in the dataset
    image_paths = sorted(list(glob.glob(os.path.join(path, "*.jpg"))))
    if use_shuffle:
        random.shuffle(image_paths)
    return image_paths


# 使用霍夫变换检测直线
def detect_line(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # blur the image
    gray = cv2.GaussianBlur(gray, (1, 5), 0, 0)
    # 二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    # canny边缘检测
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    # 霍夫直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # # return None if no line is detected
    if lines is None:
        return img, None

    if len(lines) > 1:
        # get middle line
        length = len(lines)
        x1, y1, x2, y2 = lines[length // 2][0]
        # x1, y1, x2, y2 = lines[-1][0]
    else:
        x1, y1, x2, y2 = lines[0][0]
    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # 将图片旋转到水平
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    if abs(angle) > 45:
        return img, None
    if abs(angle) > 0:
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return img, M
    return img, None

# main
if __name__ == '__main__':
    # list all images
    image_paths = list_images("data", use_shuffle=True)
    # loop over the image paths
    for image_path in image_paths:
        # load the image and show it
        image = cv2_imread(image_path)
        image, M = detect_line(image)
        if M is not None:
            # find same name mask end with png
            mask_path = image_path[:-4] + '.png'
            # affine transform M to mask
            mask = cv2_imread(mask_path)
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
            # save mask
            cv2_imwrite(mask_path, mask)

        # show image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        print(image_path)
        # save image
        cv2_imwrite(image_path, image)
        # break


    cv2.destroyAllWindows()
