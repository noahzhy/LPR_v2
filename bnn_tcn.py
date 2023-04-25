import os
import larq as lq
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *


kwargs = dict(
    use_bias=False,
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip"
)

class BCNN(Model):
    def __init__(self, shape=(28, 28, 1)):
        super().__init__()
        self.qConv2d1 = lq.layers.QuantConv2D(
            32, (3, 3),
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False,
            input_shape=shape
        )
        self.maxpool1 = MaxPooling2D((2, 2))
        self.bn1 = BatchNormalization(scale=False)

        self.cnn_block1 = self.CNN_Block(64, (3, 3))
        self.cnn_block2 = self.CNN_Block(128, (3, 3))
        self.cnn_block3 = self.CNN_Block(256, (3, 3))

    def CNN_Block(self, filters, kernel_size, strides=1, name=None):
        return Sequential([
            lq.layers.QuantConv2D(filters, kernel_size, padding='same', strides=strides, **kwargs),
            MaxPooling2D((2, 2)),
            BatchNormalization(scale=False),
        ], name=name)

    def call(self, inputs):
        x = self.qConv2d1(inputs)
        x = self.bn1(self.maxpool1(x))
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        return x


class ResBlock(Model):
    def __init__(self, filters, factor, kernel_size=3, strides=1, name=None):
        super().__init__()
        dilation_rate = 2 ** factor
        self.conv1 = lq.layers.QuantConv1D(
            filters, kernel_size,
            padding='causal',
            strides=strides,
            dilation_rate=dilation_rate,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=False,
        )
        self.bn1 = BatchNormalization(scale=False)
        self.conv2 = lq.layers.QuantConv1D(
            filters, kernel_size,
            padding='causal',
            strides=strides,
            dilation_rate=dilation_rate,
            **kwargs
        )
        self.bn2 = BatchNormalization(scale=False)
        self.add = Add()
        self.relu = Activation('relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([x, inputs])
        x = self.relu(x)
        return x


class BTCN(Model):
    def __init__(self):
        super().__init__()
        self.dense_softmax = lq.layers.QuantDense(10, activation="softmax", **kwargs)

    def call(self, inputs):
        x = inputs
        # shape
        print(x.shape)
        
        for i in range(3):
            x = ResBlock(64, i, name=f"resblock{i}")(x)
            # shape
            print(x.shape)

        # softmax dense
        x = self.dense_softmax(x)

        return x

if __name__ == "__main__":
    # model = BTCN()
    model = ResBlock(128, 2)
    # model = BCNN()
    model.build(input_shape=(None, 28, 28, 1))
    model.summary()
    # save
    model.save_weights("bnn.h5")
    lq.models.summary(model)
