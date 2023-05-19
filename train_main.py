import os
import glob

import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers.optimizer_v2 import *
from keras.optimizers import *

from cosine import *
from utils import *
from mbv3s import TinyLPR


os.environ["CUDA_VISIBLE_DEVICES"]="0"

# training config
MAX_LABEL_LENGTH = 10
BATCH_SIZE = 256

TRAIN_SAMPLE = 189131
TEST_SAMPLE = 21015
# TRAIN_SAMPLE = 10000
# TEST_SAMPLE = 1000
NUM_EPOCHS = 100
WARMUP_EPOCH = 10
LEARNING_RATE = 3e-4

input_shape = (64, 128, 1)
char_num = 85
train_dataloader = LPGenerate(BATCH_SIZE, shuffle=True, sample_num=-1, dir_path='train', target_size=input_shape[:2])
test_dataloader = LPGenerate(BATCH_SIZE, shuffle=False, sample_num=-1, dir_path='test', target_size=input_shape[:2])

epoch_step = TRAIN_SAMPLE // BATCH_SIZE
warmup_batches = WARMUP_EPOCH * epoch_step
total_steps = NUM_EPOCHS * epoch_step
warmup_steps = WARMUP_EPOCH * epoch_step


model = TinyLPR(
    shape=input_shape,
    output_dim=char_num+1,
    train=True,
).build(input_shape=[
    (BATCH_SIZE, *input_shape),
    (BATCH_SIZE, MAX_LABEL_LENGTH),
    (BATCH_SIZE, *input_shape),
])

# model.load_weights('best_model_v3.h5', by_name=True, skip_mismatch=True)
model.load_weights('best_model_final.h5', by_name=True, skip_mismatch=True)

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=1e-6,
    warmup_steps=warmup_steps,
)


def train(model, train_data, test_data):
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
        validation_data=test_data,
    )


def evl():
    BATCH_SIZE = 800
    evl_dataloader = LPGenerate(BATCH_SIZE, shuffle=False, dir_path='train', target_size=input_shape[:2], evl_mode=True)
    test_model = TinyLPR(
        shape=input_shape,
        output_dim=char_num+1,
        train=False,
    ).build(input_shape=(BATCH_SIZE, *input_shape))

    test_img, test_label = evl_dataloader.__getitem__(0)
    test_label = test_label[0]

    # test_model.load_weights(filepath='best_model.h5', by_name=True)
    test_model.load_weights(filepath='best_model.h5')

    y_pred = test_model.predict(test_img[0])
    shape = y_pred.shape
    ctc_decode = tf.keras.backend.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = tf.keras.backend.get_value(ctc_decode)[:, :MAX_LABEL_LEN]
    print(out)

    correct = 0
    single_correct = 0
    double_correct = 0
    sum_single = 1e-8
    sum_double = 1e-8
    format_err = 1e-8
    for i in range(BATCH_SIZE):
        y_pred = "".join([CHARS[x] for x in out[i] if x != -1])
        label = "".join([DECODE_DICT[x] for x in test_label[i] if x != len(DECODE_DICT)])
        # print("Label: {}, \tPrediction: {}".format(label, y_pred))

        label = label.strip()
        y_pred = y_pred.strip()

        if label.find(' ') != -1:
            sum_double += 1
        else:
            sum_single += 1

        if y_pred == label:
            correct += 1
            # if label with space
            if label.find(' ') != -1:
                double_correct += 1
            else:
                single_correct += 1

        else:
            if not is_valid(y_pred):
                format_err += 1
            else:
                print("Error: {}, \t{}".format(label, y_pred))

    # keep 2 decimal places in percentage
    print("Accuracy: {:.2f}%".format(correct / BATCH_SIZE * 100))
    print("Single Accuracy: {:.2f}%".format(single_correct / sum_single * 100))
    print("Double Accuracy: {:.2f}%".format(double_correct / sum_double * 100))
    print("Final Accuracy: {:.2f}%".format(correct / (BATCH_SIZE-format_err) * 100))


def is_valid(license_plate):
    # remove space
    license_plate = license_plate.replace(' ', '')
    regex = re.compile(r'^([가-힣]{2}\d{2}|\d{2,3})[가-힣]{1}\d{4}$')
    return regex.match(license_plate) is not None


if __name__ == '__main__':
    model.summary()
    train(model, train_dataloader, test_dataloader)
    evl()
