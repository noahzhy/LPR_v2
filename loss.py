import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Layer


class CTCLayer(Layer):
    def __init__(self, name=None, **kwargs):
        super(CTCLayer, self).__init__(name=name, **kwargs)
        self.loss_fn = K.ctc_batch_cost
        self.alpha = 0.5
        self.gamma = 2.0

    def call(self, y_true, y_pred):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        p = tf.exp(-loss)
        focal_ctc_loss = tf.multiply(tf.multiply(self.alpha, tf.pow((1 - p), self.gamma)), loss)
        loss = tf.reduce_mean(focal_ctc_loss)
        return loss


class ACELayer(Layer):
    def __init__(self, name=None, **kwargs):
        super(ACELayer, self).__init__(name=name, **kwargs)
        self.softmax = None
        self.label = None
        
    def call(self, label, inputs):
        bs, h, w, c = inputs.shape.as_list()
        # print(inputs.shape.as_list())
        T_ = h * w

        inputs = tf.reshape(inputs, (bs, T_, -1))
        inputs = inputs + 1e-10

        self.softmax = inputs
        nums, dist = label[:,0], label[:,1:]
        nums = T_ - nums

        inputs = tf.reduce_sum(inputs, axis=1)
        inputs = inputs / T_
        label = label / T_

        # convert to float32
        label = tf.cast(label, tf.float32)
        mul = tf.math.log(inputs) * label
        loss = (-tf.reduce_sum(mul)) / bs
        return loss

    def decode_batch(self):
        out_best = tf.argmax(self.softmax, 2).numpy()
        pre_result = [0]*self.bs

        for j in range(self.bs):
            pre_result[j] = out_best[j][out_best[j]!=0].astype(np.int32)

        return pre_result
