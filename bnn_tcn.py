import os
import larq as lq
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *


kwargs = dict(
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip"
)

class BNN_TCN(Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.dense1 = lq.layers.QuantDense(
            512, kernel_quantizer="ste_sign", kernel_constraint="weight_clip"
        )
        self.dense2 = lq.layers.QuantDense(10, activation="softmax", **kwargs)

    def ResBlock(self, factor):
        dilation_rate = 2 ** factor

        c = lq.layers.QuantConv1D(
            filters=64,
            kernel_size=3,
            padding="causal",
            dilation_rate=dilation_rate,
            **kwargs
        )(x)


    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


model = BNN_TCN()
model.build(input_shape=(None, 28, 28, 1))
model.summary()
lq.models.summary(model)
