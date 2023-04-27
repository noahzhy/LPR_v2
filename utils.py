import os
import sys
import time
import glob
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import *


MAX_LABEL_LEN = 8

CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopqABCDEFGHIJKLMNOPQ "
print(len(CHARS))
# to utf-8
CHARS = CHARS.encode('utf-8').decode('utf-8')
print(CHARS)
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}

# South Korea city
koreaCity = {
    '서울':'A', '부산':'B', '대구':'C', '인천':'D',
    '광주':'E', '대전':'F', '울산':'G', '세종':'H',
    '경기':'I', '강원':'J', '충북':'K', '충남':'L',
    '전북':'M', '전남':'N', '경북':'O', '경남':'P',
    '제주':'Q'
}


# open image via path and keep Green channel only
def open_image(path, channel='G'):
    img = Image.open(path)
    img = img.convert('RGB')
    r, g, b = img.split()
    if channel == 'R':
        return r
    elif channel == 'G':
        return g
    elif channel == 'B':
        return b
    else:
        return img.convert('L')


# create a new image with black background
def create_image(raw_img, width=96, height=48):
    img = Image.new('L', (width, height), (0))
    # resize the raw image to fit the target image width, and keep the ratio
    raw_img = raw_img.resize(
        (width, int(raw_img.size[1] * width / raw_img.size[0])),
        Image.ANTIALIAS
    )
    # paste the raw image to the target image at top
    img.paste(raw_img, (0, 0))
    return img


# function to binarize the image, threshold is 128
def binarize(img, threshold=128):
    ba = np.array(img)
    ba[ba < threshold] = 0
    ba[ba >= threshold] = 255
    return Image.fromarray(ba)


# 对图片进行二值化处理，otsu算法
def otsu_binarize(img):
    img = np.array(img)
    # img = img.convert('L')
    # img = np.array(img)
    w, h = img.shape
    # 计算灰度直方图
    hist = np.zeros(256)
    for i in range(w):
        for j in range(h):
            hist[img[i, j]] += 1
    # 归一化
    hist = hist / (w * h)
    # 计算类间方差
    max_var = 0
    threshold = 0
    for t in range(256):
        # 计算类间方差
        w0 = np.sum(hist[:t])
        w1 = np.sum(hist[t:])
        u0 = np.sum([i * hist[i] for i in range(t)]) / w0
        u1 = np.sum([i * hist[i] for i in range(t, 256)]) / w1
        var = w0 * w1 * (u0 - u1) ** 2
        # 寻找最大类间方差
        if var > max_var:
            max_var = var
            threshold = t
    # 二值化
    img[img < threshold] = 0
    img[img >= threshold] = 255
    return Image.fromarray(img)


# function to_label
# input: string
# output: list of ints
# example: "제주79바4470" -> [83, 7, 9, 45, 4, 4, 7, 0]
def to_label(text):
    # text = text.encode('utf-8').decode('utf-8')
    # if first two characters are korea city
    if text[:2] in koreaCity:
        text = koreaCity[text[:2]] + text[2:]

    # to list of ints
    ints = []
    for c in text:
        ints.append(CHARS_DICT[c])
    return ints


# function string_to_ints
# input: string
# output: list of ints
# example: "123" -> [1, 2, 3]
def string_to_ints(text):
    ints = []
    for c in text:
        ints.append(CHARS_DICT[c])
    return ints


# function ints_to_string
# input: list of ints
# output: string
# example: [1, 2, 3] -> "123"
def ints_to_string(ints):
    string = ""
    for i in ints:
        string += DECODE_DICT[i]
    return string


# test_txt = "제주79바4470"
# print(to_label(test_txt))

# function path to label
# input: path
# output: label
# example: "data/제주79바4470.jpg" -> [83, 7, 9, 45, 4, 4, 7, 0]
def path_to_label(path):
    label = os.path.basename(path).split('.')[0].split('_')[0]
    return to_label(label)


# function to generate batch
# input: batch_size, images, labels
# output: batch_images, batch_labels
class LPGenerate(Sequence):
    def __init__(self, batch_size, dir_path="data", target_size=(48, 96), shuffle=True):
        self.batch_size = batch_size
        self.images = glob.glob(dir_path + '/*.*')
        # shuffle the images
        np.random.shuffle(self.images)
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = self.__data_generation(batch_images)
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batches):
        # swap height and width
        width, height = self.target_size
        X = np.empty((self.batch_size, *(height, width), 1))
        Y = np.full((self.batch_size, MAX_LABEL_LEN), len(CHARS)+1, dtype=int)

        for i, img_path in enumerate(batches):
            img = open_image(img_path, channel='G')
            img = create_image(img, height, width)
            img = img.rotate(270, expand=True)
            label = path_to_label(img_path)

            X[i,] = np.expand_dims(img, axis=-1) / 255.0
            Y[i,][:len(label)] = np.array(label)

        return [X, Y], Y


if __name__ == "__main__":
    dataLoader = LPGenerate(5, shuffle=True)
    for i in range(0, len(dataLoader)):
        x, y = dataLoader[i]
        # print('%s, %s => %s' % (x['input_image'].shape, x['label'].shape, y.shape))
        img_data, label_data = x
        # show image
        img = Image.fromarray(np.squeeze(img_data[0] * 255).astype(np.uint8))
        img.show()

        print(img_data.shape)
        print(label_data)
        print(y)
        break
