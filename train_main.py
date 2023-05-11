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
TRAIN_SAMPLE = 58420
# TRAIN_SAMPLE = 5000
NUM_EPOCHS = 100
WARMUP_EPOCH = 0
LEARNING_RATE = 3e-4

input_shape = (64, 128, 1)
char_num = 85
train_dataloader = LPGenerate(BATCH_SIZE, shuffle=True, sample_num=TRAIN_SAMPLE)
test_dataloader = LPGenerate(BATCH_SIZE, shuffle=False, dir_path='test')

epoch_step = TRAIN_SAMPLE // BATCH_SIZE
warmup_batches = WARMUP_EPOCH * epoch_step
total_steps = NUM_EPOCHS * epoch_step
warmup_steps = WARMUP_EPOCH * epoch_step


model = TinyLPR(
    bs=BATCH_SIZE,
    shape=input_shape,
    output_dim=char_num+1,
    train=True,
).build(input_shape=[
    (BATCH_SIZE, *input_shape),
    (BATCH_SIZE, MAX_LABEL_LENGTH),
    (BATCH_SIZE, char_num+1),
])
model.load_weights('best_model.h5')

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=1e-6,
    warmup_steps=warmup_steps,
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


def test():
    test_model = TinyLPR(
        bs=BATCH_SIZE,
        shape=input_shape,
        output_dim=char_num+1,
        train=False,
    ).build(input_shape=[(BATCH_SIZE, *input_shape)])

    test_img, test_label = test_dataloader.__getitem__(0)
    test_label = test_label[0]

    test_model.load_weights(filepath='best_model.h5')

    y_pred = test_model.predict(test_img[0])
    shape = y_pred.shape
    ctc_decode = tf.keras.backend.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = tf.keras.backend.get_value(ctc_decode)[:, :MAX_LABEL_LEN]

    correct = 0
    single_correct = 0
    double_correct = 0
    sum_single = 0
    sum_double = 0
    format_err = 0
    for i in range(BATCH_SIZE):
        y_pred = "".join([CHARS[x] for x in out[i] if x != -1])
        label = "".join([DECODE_DICT[x] for x in test_label[i] if x != 86])

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
            if not is_correct(y_pred):
                format_err += 1
            else:
                print("Error: {}, \t{}".format(label, y_pred))

    # keep 2 decimal places in percentage
    print("Accuracy: {:.2f}%".format(correct / BATCH_SIZE * 100))
    print("Single Accuracy: {:.2f}%".format(single_correct / sum_single * 100))
    print("Double Accuracy: {:.2f}%".format(double_correct / sum_double * 100))
    print("Format Error: {:.2f}%".format(format_err / BATCH_SIZE * 100))


def is_correct(string):
    # remove space
    string = string.replace(' ', '')
    if len(string) < 7:
        return False

    # if every char is number
    if string.isdigit():
        return False

    # if last four char is number
    if not string[-4:].isdigit():
        return False

    kor_count = 0
    num_count = 0
    for char in string:
        if char.isdigit():
            num_count += 1
        # if char is not number and not alphabet
        elif not char.isalpha():
            kor_count += 1

    if kor_count > 1:
        return False

    if not (num_count == 6 or num_count == 7):
        return False

    # if first char is alphabet
    if string[-5].isdigit():
        return False

    return True


if __name__ == '__main__':
    model.summary()
    # train(model, train_dataloader, test_dataloader)
    test()
