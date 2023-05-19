import os
import sys
import time
import glob
import random
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import *
from keras.preprocessing.image import ImageDataGenerator


MAX_LABEL_LEN = 10

CHARS = " 0가A조a서B무b1나C호c어D부d2다E고e저F수f3라G노g허H우h4마I도i거J주j5바K로k너L배l6사M모m더N구n7아O보o러P누p8자Q소q머두하9오버루"
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}
# print('CHARS_DICT: ', CHARS_DICT)
# print('DECODE_DICT: ', DECODE_DICT)
# South Korea city
koreaCity = {
    '서울': 'A', '부산': 'B', '대구': 'C', '인천': 'D',
    '광주': 'E', '대전': 'F', '울산': 'G', '세종': 'H',
    '경기': 'I', '강원': 'J', '충북': 'K', '충남': 'L',
    '전북': 'M', '전남': 'N', '경북': 'O', '경남': 'P',
    '제주': 'Q'
}


def str2list(string):
    res = []
    # if first two chars are both korean letters
    if string[:2] in koreaCity.keys():
        # korean city
        korea_city_letter = koreaCity[string[:2]]
        string = korea_city_letter + string[2:]

    # if string include space
    if string.find(' ') != -1:
        string = " " + string

    for char in string:
        # if not in CHARS_DICT, append space
        if char not in CHARS_DICT.keys():
            # print('\nchar not in CHARS_DICT: ', char)
            res.append(len(CHARS_DICT))
        else:
            res.append(CHARS_DICT[char])
    return res


def decoder(file_path):
    lp = ""
    # get file base name from file path
    f_base = os.path.basename(file_path).split('.')[0]

    # split via '_' to a list
    f_base = f_base.split('_')
    for i in f_base:
        if len(lp+i) > MAX_LABEL_LEN+1:
            break
        lp += i

    return np.array(str2list(lp))


# open image via path and keep Green channel only
def open_image(path, channel='L'):
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
    # to image if raw_img is numpy array
    if type(raw_img) == np.ndarray:
        raw_img = Image.fromarray(raw_img)

    img = Image.new('L', (width, height), (0))
    # resize the raw image to fit the target image width, and keep the ratio
    raw_img = raw_img.resize(
        (width, int(raw_img.size[1] * width / raw_img.size[0])),
        Image.Resampling.LANCZOS
    )
    img.paste(raw_img, (0, 0))
    return img


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


def decode_label(ints):
    # remove -1
    ints = [i for i in ints if i != -1]
    # remove duplicates
    return ints_to_string(ints)


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
    def __init__(self, batch_size, dir_path="train", target_size=(64, 128), shuffle=True, sample_num=5000, evl_mode=False):
        self.batch_size = batch_size
        self.total_images = glob.glob(dir_path + '/*.jpg')
        self.sample_num = sample_num
        if sample_num != -1:
            # fix random seed
            np.random.seed(0)
            self.images = np.random.choice(self.total_images, sample_num)
        else:
            self.images = self.total_images
        self.target_size = target_size
        self.shuffle = shuffle
        self.evl_mode = evl_mode
        self.on_epoch_end()

    def find_matching_mask(self, img_path):
        # find matching mask path
        mask_path = img_path.replace('.jpg', '.png')
        # read as numpy array
        mask = np.array(Image.open(mask_path).convert('L'))
        # to binary if x > 128 else 0
        mask = np.where(mask > 128, 1, 0)
        h, w = self.target_size
        mask = create_image(mask, width=w, height=h)
        return mask

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
            if self.sample_num != -1:
                self.images = np.random.choice(self.total_images, self.sample_num)

    def __data_generation(self, batches):
        height, width = self.target_size
        X = np.empty((self.batch_size, *(height, width), 1))
        M = np.zeros((self.batch_size, *(height, width), 1))
        # CTC loss
        C = np.full((self.batch_size, MAX_LABEL_LEN), len(CHARS), dtype=int)

        for i, img_path in enumerate(batches):
            img = open_image(img_path, channel='L')
            img = create_image(img, width=width, height=height)

            if not self.evl_mode:
                mask = self.find_matching_mask(img_path)

            c_labels = decoder(file_path=img_path)
            if len(c_labels) > MAX_LABEL_LEN:
                print("label length is over than max length: ", c_labels, img_path)

            X[i,] = np.expand_dims(img, axis=-1) / 255.0
            # print(img_path, c_labels)
            C[i,][:len(c_labels)] = c_labels
            if not self.evl_mode:
                M[i,] = np.expand_dims(mask, axis=-1)

        Y = [C, M]
        return [X, Y], Y


if __name__ == "__main__":
    dataLoader = LPGenerate(5, shuffle=True, dir_path="train", sample_num=100)
    for i in range(0, len(dataLoader)):
        x, y = dataLoader[i]
        img_data, m = x
        ctc, mask = m
        # using matplotlib to show image and mask
        fig = plt.figure(figsize=(10, 10))
        for j in range(0, 5):
            ax = fig.add_subplot(5, 2, 2 * j + 1)
            ax.imshow(np.squeeze(img_data[j]), cmap='gray')
            ax = fig.add_subplot(5, 2, 2 * j + 2)
            ax.imshow(np.squeeze(mask[j]), cmap='gray')
        plt.show()
        break
