import os
import glob
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *

from cosine import *
from utils import *
from mbv3s import TinyLPR


os.environ["CUDA_VISIBLE_DEVICES"]="0"

# del logs dir if exists
if os.path.exists('./logs'):
    shutil.rmtree('./logs')

# training config
MAX_LABEL_LENGTH = 10
BATCH_SIZE = 256

# TRAIN_SAMPLE = glob.glob('train/*.jpg')
# TEST_SAMPLE = glob.glob('test/*.jpg')
TRAIN_SAMPLE = 20000
TEST_SAMPLE = 1000
NUM_EPOCHS = 200
WARMUP_EPOCH = 0
LEARNING_RATE = 3e-40

# optimizer = Adam(learning_rate=LEARNING_RATE, amsgrad=True)
# optimizer = Nadam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
optimizer = SGD(learning_rate=LEARNING_RATE, momentum=0.95, nesterov=False)

metrics_keys = "val_ctc_loss"
metrics_keys = "val_loss"

input_shape = (64, 128, 1)
char_num = 85

train_dataloader = LPGenerate(
    BATCH_SIZE,
    shuffle=True,
    sample_num=TRAIN_SAMPLE,
    dir_path='train',
    target_size=input_shape[:2],
    datasetType=DatasetType.BALANCE,
)
test_dataloader = LPGenerate(
    BATCH_SIZE,
    shuffle=False,
    sample_num=TEST_SAMPLE,
    dir_path='test',
    target_size=input_shape[:2],
    datasetType=DatasetType.BALANCE,
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

# model.load_weights('s9307_d9576_fa9410.h5', by_name=True, skip_mismatch=True)

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


def eval(dir_path='test'):
    BATCH_SIZE = 800
    evl_dataloader = LPGenerate(
        BATCH_SIZE,
        shuffle=False,
        dir_path=dir_path,
        target_size=input_shape[:2],
        datasetType=DatasetType.FULL,
        evl_mode=True,
    )
    test_model = TinyLPR(
        shape=input_shape,
        output_dim=char_num+1,
        train=False,
        with_mask=False,
    ).build(input_shape=[(BATCH_SIZE, *input_shape),])

    test_img, test_label = evl_dataloader.__getitem__(0)
    test_label = test_label[0]

    # test_model.load_weights(filepath='best_model.h5')
    test_model.load_weights(filepath='s9307_d9576_fa9410.h5')

    y_pred = test_model.predict(test_img[0])
    shape = y_pred.shape
    ctc_decode = tf.keras.backend.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = tf.keras.backend.get_value(ctc_decode)[:, :MAX_LABEL_LEN]

    correct = 0
    single_correct = 0
    double_correct = 0
    sum_single = 1e-8
    sum_double = 1e-8
    format_err = 1e-8
    error_chars_count = 0
    error_num_count = 0
    total_error = 1e-8
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

        # remove space in y_pred and label
        _y_pred = y_pred.replace(' ', '')
        _label = label.replace(' ', '')

        if _y_pred == _label:
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

    print("\nResults\n--------------------------------------------------")
    # keep 2 decimal places in percentage
    print("Original Accuracy: \t{:.2f}%".format(correct / BATCH_SIZE * 100))
    print("S. LPR   Accuracy: \t{:.2f}%".format(single_correct / sum_single * 100))
    print("D. LPR   Accuracy: \t{:.2f}%".format(double_correct / sum_double * 100))
    print("Final    Accuracy: \t{:.2f}%".format(correct / (BATCH_SIZE-format_err) * 100))
    # error chars count and error num count in percentage
    print("--------------------------------------------------")
    print("Error num.  count: \t{:.2f}%".format(error_num_count / (total_error) * 100))
    print("Error chars count: \t{:.2f}%".format(error_chars_count / (total_error) * 100))
    print()
    # rename files to s{}_d{}_fa{}.h5 format(accuracy keep 2 decimal places)
    # os.rename('best_model.h5', 's{}_d{}_fa{}.h5'.format(
    #     round(single_correct / sum_single * 1e4),
    #     round(double_correct / sum_double * 1e4),
    #     round(correct / (BATCH_SIZE-format_err) * 1e4),
    # ))


if __name__ == '__main__':
    model.summary()
    # train(model, train_dataloader, test_dataloader)
    eval('eval')
