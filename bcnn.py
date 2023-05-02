import os
import glob
import random

import numpy as np
from PIL import Image

import larq as lq
import larq_zoo as lqz
import larq_compute_engine as lce
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *

from ctc import CTCLayer
from cosine import *


MAX_LABEL_LEN = 8

# CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopqABCDEFGHIJKLMNOPQ "
CHARS = """0A가B호8C저D우9E나F고G허H주I다J노K거L배M라N도O너P구Q마a로b더c누d바e모f러g두h사i보j머k루l아m소n버o무p자q오1서2부3하4조5어6수7 """

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

def to_label(text):
    # if first two characters are korea city
    if text[:2] in koreaCity:
        text = koreaCity[text[:2]] + text[2:]

    # to list of ints
    ints = []
    for c in text:
        ints.append(CHARS_DICT[c])
    return ints


def path_to_label(path):
    label = os.path.basename(path).split('.')[0].split('_')[0]
    return to_label(label)


# load lpr dataset
class LPGenerate(tf.keras.utils.Sequence):
    def __init__(self, root_dir, batch_size=128, target_size=(32, 128, 1), shuffle=True):
        self.imgs = glob.glob(os.path.join(root_dir, '*.*'))
        self.target_size = target_size
        self.shuffle = shuffle
        # if self.shuffle:
        #     random.shuffle(self.imgs)
        self.batch_size = batch_size
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.imgs) // self.batch_size

    def __getitem__(self, idx):
        batches = self.imgs[idx * self.batch_size: (idx + 1) * self.batch_size]
        return self.__data_generation(batches)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.imgs)

    def preprocessing_img(self, img):
        h, w = self.target_size[:2]
        new_img = Image.new('L', (w, h), 0)
        # resize the raw image to fit the target image width, and keep the ratio
        img = img.resize((w, int(w * img.size[1] / img.size[0])))
        # paste the resized image to top-left corner
        new_img.paste(img, (0, 0))
        return new_img

    def __data_generation(self, batches):
        X = np.empty((self.batch_size, *self.target_size))
        Y = np.full((self.batch_size, MAX_LABEL_LEN), len(CHARS)+1, dtype=int)

        for i, img_path in enumerate(batches):
            img = Image.open(img_path).convert('L')
            img = self.preprocessing_img(img)

            label = path_to_label(img_path)
            X[i,] = np.expand_dims(img, axis=-1) / 127.5 - 1
            Y[i,][:len(label)] = np.array(label)

        return [X, Y], Y


# All quantized layers except the first will use the same options
kwargs = dict(
    use_bias=False,
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip"
)


BATCH_SIZE = 128
TRAIN_SAMPLE = 8935
NUM_EPOCHS = 200
WARMUP_EPOCH = 10
LEARNING_RATE = 2e-3

epoch_step = TRAIN_SAMPLE // BATCH_SIZE
warmup_batches = WARMUP_EPOCH * TRAIN_SAMPLE / BATCH_SIZE
total_steps = int(NUM_EPOCHS * TRAIN_SAMPLE / BATCH_SIZE)
# Compute the number of warmup batches.
warmup_steps = int(WARMUP_EPOCH * TRAIN_SAMPLE / BATCH_SIZE)

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=1e-6,
    warmup_steps=warmup_steps,
    hold_base_rate_steps=5,
)


class BCNN(Model):
    def __init__(self, input_shape=(32, 128, 1), output_dim=86, train=True, **kwargs):
        super(BCNN, self).__init__()
        self.output_dim = output_dim
        self.train = train

        nn = lqz.sota.QuickNet(
            input_shape=input_shape,
            include_top=True,
            weights=None,
        )

        self.backbone = Model(inputs=nn.input, outputs=nn.get_layer("batch_normalization_7").output)

        # dropout
        self.dropout = tf.keras.layers.Dropout(0.2)

        self.conv1 = lq.layers.QuantConv2D(128, (3, 3), padding="same", **kwargs)
        self.batchnorm1 = tf.keras.layers.BatchNormalization(scale=False)

        self.head = lq.layers.QuantConv2D(self.output_dim, (1, 1), padding="same", **kwargs)
        self.batchnorm5 = tf.keras.layers.BatchNormalization(scale=False)
        self.activation = tf.keras.layers.Activation("softmax")

        self.ctc = CTCLayer()


    def build(self):
        backbone = self.backbone
        # x = nn.get_layer("batch_normalization_12").output

        x = backbone.output

        # x = tf.split(x, 2, axis=2)
        # x = tf.concat(x, axis=1)
        # x = tf.reshape(x, shape=(-1, x.shape[1], x.shape[2] * x.shape[3]))

        x = self.dropout(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # [batch_size, 4, 16, 128] -> [batch_size, 16, 128]

        x = self.head(x)
        x = self.batchnorm5(x)
        x = self.activation(x)
        x = tf.reduce_mean(x, axis=1)

        if self.train:
            labels = Input(name='labels', shape=(MAX_LABEL_LEN,), dtype='int64')
            ctc = CTCLayer(name='ctc_loss')(labels, x)
            return Model(inputs=[backbone.input, labels], outputs=ctc, name='b_lpr')

        return Model(inputs=backbone.input, outputs=x, name='b_lpr')

    def call(self, inputs):
        return self.model(inputs.shape)(inputs)


def train(
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    train_data_path="data",
    test_data_path="test",
    saved_model_path="b_lpr.h5"
):

    model = BCNN(
        input_shape=(32, 128, 1),
        output_dim=len(CHARS) + 1,
        train=True,
    ).build()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=lambda y_true, y_pred: y_pred,
    )

    train_dataloader = LPGenerate(root_dir=train_data_path, batch_size=batch_size, shuffle=True)
    test_dataloader = LPGenerate(root_dir=test_data_path, batch_size=batch_size, shuffle=False)

    model.fit(
        train_dataloader,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=test_dataloader,
        callbacks = [
            ModelCheckpoint(
                filepath=saved_model_path,
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
    )

    model.load_weights(saved_model_path)
    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    # print(f"Test accuracy {test_acc * 100:.2f} %")

    # convert to tflite
    print("Convert to tflite")
    tflite_model_path = saved_model_path.split(".")[0] + ".tflite"
    with open(tflite_model_path, "wb") as flatbuffer_file:
        flatbuffer_bytes = lce.convert_keras_model(model)
        flatbuffer_file.write(flatbuffer_bytes)


if __name__ == '__main__':
    # train()

    test_dataloader = LPGenerate(root_dir="test", batch_size=128, shuffle=True)

    model = BCNN(
        input_shape=(32, 128, 1),
        output_dim=len(CHARS) + 1,
        train=False,
    ).build()
    model.load_weights("b_lpr.h5")

    lq.models.summary(model)

    test_img, test_label = test_dataloader.__getitem__(0)

    y_pred = model.predict(test_img[0])
    shape = y_pred[:, 2:, :].shape
    ctc_decode = tf.keras.backend.ctc_decode(
        y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1]
    )[0][0]
    out = tf.keras.backend.get_value(ctc_decode)[:, :MAX_LABEL_LEN]

    correct = 0
    for i in range(128):
        y_pred = "".join([CHARS[x] for x in out[i] if x != -1])
        label = "".join([DECODE_DICT[x] for x in test_label[i] if x != 86])
        # print(y_pred, label)
        if y_pred == label:
            correct += 1
        else:
            print("{}, \t{}".format(label, y_pred))

    print(correct / 128)

    # model.summary()
    # lq.models.summary(model)
    # # save model
    # model.save("b_lpr.h5")

    # # dataloader
    # dataloader = LPGenerate(root_dir="data", batch_size=64, shuffle=True)
    # import matplotlib.pyplot as plt

    # for i in range(10):
    #     (X, Y), _ = dataloader.__getitem__(i)
    #     plt.imshow(X[0, :, :, 0])
    #     plt.show()
    #     print(Y[0])
    #     break
