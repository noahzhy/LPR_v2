import sys
sys.path.append('./')
import keras
from keras.callbacks import Callback
import numpy as np
from utils.utils import *
from config import *
import random
import tensorflow as tf


class mCallback(Callback):
    def __init__(self, model, val_dir, n_sample=200):
        self.model = keras.models.Model(
            model.get_layer(name="image").input,
            model.get_layer(name="softmax0").output
        )
        self.val_dir = glob.glob(os.path.join(val_dir, '*.jpg'))
        self.n_sample = n_sample

    def on_epoch_end(self, epoch, logs=None):
        val_list = random.sample(self.val_dir, self.n_sample)
        val_img = np.empty((self.n_sample, HEIGHT, WIDTH, CHANNEL))
        val_label = []

        for idx, image_path in enumerate(val_list):
            img, label = image_preprocess(image_path)
            img = tf.reshape(img, [HEIGHT, WIDTH, CHANNEL])
            val_img[idx] = img
            val_label.append(label)

        if epoch % 10 == 0:
            print(val_img.shape)
            self.model.summary()
            y_pred = self.model.predict(val_img, batch_size=self.n_sample)
            acc = self.calculate_acc(y_true=val_label, y_pred=y_pred, n=self.n_sample)
            print("End epoch {} => val_acc: {}".format(epoch, acc))

    def calculate_acc(self, y_true, y_pred, n):
        counter = 0
        total = n
        for batch in zip(y_true, y_pred):
            if decode_label(batch[1]) == batch[0]:
                counter += 1
        return round(counter/total, 8)


if __name__ == '__main__':
    pass