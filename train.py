import os
import sys

import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer
from keras import backend as K

from ctc import CTCLayer

from model import TemporalConvNet
from cnn import *


MAX_LABEL_LEN = 8

CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopqABCDEFGHIJKLMNOPQ"  # exclude IO
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}
print(CHARS_DICT)
print(DECODE_DICT)

print(len(CHARS_DICT))

data_dir = Path("./data/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split("_")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

unique_characters = len(characters)
print(unique_characters)

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Characters present: ", characters)

# print(labels[:10], images[:10])



def cnn_tcn_model():
    # inputs for cnn, image size: 48x192
    inputs = Input(shape=(48, 192, 1), name='the_input')
    cnn = CNN().model
    out_cnn = cnn(inputs)

    # inputs = Input(name='the_input', shape=[128, 1], dtype='float32')
    labels = Input(name='the_labels', shape=[MAX_LABEL_LEN], dtype='float32')

    x = TemporalConvNet(256, 6, 64)
    x = x.model(out_cnn)
    # Output layer
    x = Dense(unique_labels+1, activation="softmax", name="dense2")(x)
    output = CTCLayer("ctc_loss")(labels, x)

    model = Model(inputs=[inputs, labels], outputs=output, name="tcn")
    model.compile(optimizer="adam")
    model.summary()
    model.save('model.h5')

    return model


def train():
    model = cnn_tcn_model()

    history = model.fit(
        x=[train_x, train_y],
        y=np.zeros(len(train_x)),
        batch_size=32,
        epochs=10,
        validation_data=([test_x, test_y], np.zeros(len(test_x))),
    )



if __name__ == "__main__":
    pass
    # model = cnn_tcn_model()
    # img = np.random.rand(1, 48, 192, 1)
    # label = np.random.rand(1, MAX_LABEL_LEN)
    # y_pred = model.predict([img, label])
    
    # print(y_pred)
