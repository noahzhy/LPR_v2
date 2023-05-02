import os
import sys
import random
import glob

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers.optimizer_v2 import *
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
# adam
from keras.optimizers import *

from cosine import *
# from mcallback import *
from utils import *
from ctc import CTCLayer
# from tinyLPR import *
from tinyLPR_v2 import TinyLPR
# from bnn_tcn import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# training config
BATCH_SIZE = 64
TRAIN_SAMPLE = len(glob.glob('data/*'))
NUM_EPOCHS = 200
WARMUP_EPOCH = 10
LEARNING_RATE = 2e-3

input_shape = (96, 48, 1)
trainGen = LPGenerate(BATCH_SIZE, shuffle=True)
valGen = LPGenerate(BATCH_SIZE, shuffle=False, dir_path='test')
blocks = 2

model = TinyLPR(
    shape=input_shape,
    output_dim=86,
    train=True,
    filters=128,
    blocks=blocks,
).build(input_shape)

test_model = TinyLPR(
    shape=input_shape,
    output_dim=86,
    train=False,
    blocks=blocks,
).build(input_shape)

# model.load_weights(filepath='best_model_945.h5')

epoch_step = TRAIN_SAMPLE // BATCH_SIZE
warmup_batches = WARMUP_EPOCH * epoch_step
total_steps = int(NUM_EPOCHS * epoch_step)
# Compute the number of warmup batches.
warmup_steps = int(WARMUP_EPOCH * epoch_step)

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=1e-6,
    warmup_steps=warmup_steps,
    hold_base_rate_steps=5,
)


def train(model, train_data, val_data):
    model.compile(
        loss=lambda y_true, y_pred: y_pred,
        optimizer=Adam(learning_rate=LEARNING_RATE),
    )
    callbacks_list = [
        ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_loss',
            save_best_only=True,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=50,
            verbose=0,
            mode='auto'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            mode='auto',
            factor=0.5,
            patience=20,
        ),
        TensorBoard(log_dir='./logs'),
        warm_up_lr,
    ]
    model.fit(
        train_data,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        validation_data=val_data,
    )

def evl():
    test_model.load_weights(filepath='best_model.h5')
    test_model.train = False
    test_imgs = glob.glob('test/*.*')
    # get 200 random images
    test_imgs = random.sample(test_imgs, 200)
    total = 0
    correct = 0
    for img_path in test_imgs:
        # load image
        test_img = Image.open(img_path)
        # create empty Image
        img = create_image(test_img)
        # rotate 270 with PIL
        img = img.rotate(270, expand=True)
        img = np.array(img)
        model_input = np.expand_dims(img, axis=-1) / 255.0
        model_input = np.expand_dims(model_input, axis=0)
        # predict
        y_pred = test_model.predict(model_input)
        # decode
        decoded = K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], greedy=True)[0][0]
        out = K.get_value(decoded)[0].tolist()
        # remove -1 value in list
        out = [i for i in out if i != -1]
        # get label
        label = path_to_label(path=img_path)
        # compare
        if out == label:
            correct += 1
        total += 1

    print('total: ', total, 'correct: ', correct, 'acc: ', correct/total)


def save_without_ctc(model):
    model.load_weights(filepath='best_model_945.h5')
    model.summary()
    model.save(filepath='final_model.h5', include_optimizer=False)


if __name__ == '__main__':
    model.summary()
    # train(model, trainGen, valGen)
    # evl()
    save_without_ctc(test_model)
