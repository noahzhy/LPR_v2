import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer


class CTCLayer(Layer):
    def __init__(self, name=None, **kwargs):
        super(CTCLayer, self).__init__(name=name, **kwargs)
        self.loss_fn = K.ctc_batch_cost

    def call(self, y_true, y_pred):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        return self.loss_fn(y_true, y_pred, input_length, label_length)
