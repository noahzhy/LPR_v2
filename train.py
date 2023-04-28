import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *
from keras import backend as K

from utils import *
from ctc import CTCLayer
from tinyLPR import *
# from bnn_tcn import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"

bs = 64
input_shape = (96, 48, 1)
dataLoader = LPGenerate(bs, shuffle=True)
model = TinyLPR(
    shape=input_shape,
    output_dim=86,
    train=True,
).build(input_shape)

# model = TinyLPR(
#     output_dim=86,
#     train=True,
# ).build(input_shape)


def train():
    model.compile(
        optimizer=Adam(lr=0.001),
        loss=lambda y_true, y_pred: y_pred,
    )

    history = model.fit(
        dataLoader,
        batch_size=bs,
        epochs=50,
    )
    model.save(filepath='tinyLPR_bnn.h5')


def test():
    test_model = TinyLPR(
        shape=input_shape,
        output_dim=86,
        train=False,
    ).build(input_shape)
    # load
    test_model.load_weights('tinyLPR_best.h5')
    img_path = 'data/서울31아5565_1625746016.jpg'
    test_img = Image.open(img_path)
    # create empty Image
    img = create_image(test_img)
    # rotate 270 with PIL
    img = img.rotate(270, expand=True)
    # convert to numpy array
    img = np.array(img)
    model_input = np.expand_dims(img, axis=-1) / 255.0
    model_input = np.expand_dims(model_input, axis=0)
    # predict
    y_pred = test_model.predict(model_input)
    # decode
    decoded = K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], greedy=True)[0][0]
    out = K.get_value(decoded)
    # print
    print(out)


if __name__ == "__main__":
    train()
    # test()
