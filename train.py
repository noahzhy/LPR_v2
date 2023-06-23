import os
import glob
import time
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *

import prettytable as pt

from cosine import *
from utils import *
from tiny_lpr import TinyLPR


os.environ["CUDA_VISIBLE_DEVICES"]="0"

# del logs dir if exists
if os.path.exists('./logs'):
    shutil.rmtree('./logs')

# training config
MAX_LABEL_LENGTH = 10
BATCH_SIZE = 128

# TRAIN_SAMPLE = len(glob.glob('train/*.jpg'))
# TEST_SAMPLE = len(glob.glob('test/*.jpg'))
TRAIN_SAMPLE = 20000
TEST_SAMPLE = 1000
NUM_EPOCHS = 1
WARMUP_EPOCH = 0
LEARNING_RATE = 3e-50

metrics_keys = "val_ctc_loss"
metrics_keys = "val_loss"

optimizer = Adam(
    learning_rate=LEARNING_RATE,
    # decay=0.00001,
    amsgrad=True,
)
# optimizer = SGD(
#     learning_rate=LEARNING_RATE,
#     decay=1e-6,
#     momentum=0.9,
#     nesterov=True,
# )

input_shape = (64, 128, 1)
char_num = 85
datasetType = DatasetType.BALANCE
ratio = 1.0

train_dataloader = LPGenerate(
    BATCH_SIZE,
    shuffle=True,
    sample_num=TRAIN_SAMPLE,
    dir_path='train',
    target_size=input_shape[:2],
    datasetType=datasetType,
    ratio=ratio,
)
test_dataloader = LPGenerate(
    BATCH_SIZE,
    shuffle=False,
    sample_num=TEST_SAMPLE,
    dir_path='test',
    target_size=input_shape[:2],
    datasetType=datasetType,
    ratio=ratio,
)

epoch_step = TRAIN_SAMPLE // BATCH_SIZE
warmup_batches = WARMUP_EPOCH * epoch_step
total_steps = NUM_EPOCHS * epoch_step
warmup_steps = WARMUP_EPOCH * epoch_step

model = TinyLPR(
    shape=input_shape,
    output_dim=char_num+1,
    train=True,
    with_mask=(metrics_keys == "val_ctc_loss"),
).build(input_shape=[
    (BATCH_SIZE, *input_shape),
    (BATCH_SIZE, MAX_LABEL_LENGTH),
    (BATCH_SIZE, *input_shape),
])

# model.load_weights('best_model.h5', by_name=True, skip_mismatch=True)
model.load_weights('b1_1589.h5', by_name=True, skip_mismatch=True)

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=(LEARNING_RATE / 100),
    warmup_steps=warmup_steps,
)

def train(model, train_data, test_data):
    model.compile(
        loss=lambda y_true, y_pred: y_pred,
        optimizer=optimizer,
    )
    callbacks_list = [
        ModelCheckpoint(
            filepath='best_model.h5',
            monitor=metrics_keys,
            save_best_only=True,
        ),
        EarlyStopping(
            monitor=metrics_keys,
            patience=100,
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


def eval(dir_path='eval'):
    BATCH_SIZE = 512
    evl_dataloader = LPGenerate(
        BATCH_SIZE,
        shuffle=False,
        dir_path=dir_path,
        target_size=input_shape[:2],
        datasetType=DatasetType.FULL,
        sample_num=-1,
        evl_mode=True,
    )
    test_model = TinyLPR(
        shape=input_shape,
        output_dim=char_num+1,
        train=False,
        with_mask=False,
    ).build(input_shape=(BATCH_SIZE, *input_shape))

    test_model.load_weights(filepath='best_model.h5')
    # test_model.load_weights(filepath='s9576_d9528_fa9721.h5')

    total_count = len(evl_dataloader.images)
    sum_single = len([x for x in evl_dataloader.images if x.find(' ') == -1])
    sum_double = len([x for x in evl_dataloader.images if x.find(' ') != -1])

    single_correct = 0
    double_correct = 0
    format_err = 0
    error_chars_count = 0
    error_num_count = 0
    total_error = 0

    for i in range(len(evl_dataloader)+1):
        test_img, test_label = evl_dataloader[i]
        test_label = test_label[0]

        y_pred = test_model.predict(test_img[0])
        shape = y_pred.shape
        ctc_decode = tf.keras.backend.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
        out = tf.keras.backend.get_value(ctc_decode)[:, :MAX_LABEL_LEN]

        for j in range(test_label.shape[0]):
            y_pred = "".join([CHARS[x] for x in out[j] if x != -1])
            label = "".join([DECODE_DICT[x] for x in test_label[j] if x != len(DECODE_DICT)])
            if len(label) < 1:
                continue
            # remove front and end space
            y_pred = y_pred.strip()
            label = label.strip()

            # remove space in y_pred and label
            _y_pred = y_pred.replace(' ', '')
            _label = label.replace(' ', '')

            if _y_pred == _label:
                # if label with space
                if label.find(' ') != -1:
                    double_correct += 1
                else:
                    single_correct += 1
            else:
                if not is_valid(_y_pred):
                    format_err += 1
                else:
                    # get different chars in _y_pred and _label, position is important
                    if len(_y_pred) == len(_label):
                        for c in range(len(_y_pred)):
                            if _y_pred[c] != _label[c]:
                                if _y_pred[c].isdigit():
                                    error_num_count += 1
                                else:
                                    error_chars_count += 1

                    print("Error: {}, \t{}".format(label, y_pred))
                    total_error += 1

    correct = single_correct + double_correct

    print()
    pt_table = pt.PrettyTable(['', 'Count', 'Accuracy'])
    # keep 2 decimal places in percentage
    pt_table.add_row(['Single', "{:d}/{:d}".format(single_correct, sum_single), "{:.2f}%".format(single_correct / sum_single*100)])
    pt_table.add_row(['Double', "{:d}/{:d}".format(double_correct, sum_double), "{:.2f}%".format(double_correct / sum_double*100)])
    pt_table.add_row(['Total', "{:d}/{:d}".format(correct, total_count), "{:.2f}%".format(correct / total_count*100)])
    pt_table.add_row(['Final', "{:d}/{:d}".format(correct, total_count-format_err), "{:.2f}%".format(correct / (total_count-format_err)*100)], divider=True)
    pt_table.add_row(['Error num.', "{:d}/{:d}".format(error_num_count, total_error), "{:.2f}%".format(error_num_count / total_error*100)])
    pt_table.add_row(['Error chars', "{:d}/{:d}".format(error_chars_count, total_error), "{:.2f}%".format(error_chars_count / total_error*100)])
    print(pt_table)
    # rename files to s{}_d{}_fa{}.h5 format(accuracy keep 2 decimal places)
    os.rename('best_model.h5', 's{}_d{}_fa{}.h5'.format(
        round(single_correct / sum_single * 1e4),
        round(double_correct / sum_double * 1e4),
        round(correct / (total_count-format_err) * 1e4),
    ))


if __name__ == '__main__':
    model.summary()
    train(model, train_dataloader, test_dataloader)
    eval('eval')
    # eval('test')
