import os
import larq as lq
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *

from ctc import CTCLayer


kwargs = dict(
    use_bias=False,
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip",
)

MAX_LABEL_LEN = 8


class LearnableBias(Layer):
    def __init__(self, out_chn, name=None):
        super(LearnableBias, self).__init__()
        self.bias = self.add_weight(
            shape=(1, 1, 1, out_chn),
            initializer=tf.zeros_initializer(),
            trainable=True,
            name=name,
        )

    def call(self, x):
        out = x + self.bias
        return out


def conv3x3(filters, strides=1, name=None):
    return lq.layers.QuantConv2D(
        filters, (3, 3), padding="same", strides=strides, name=name, **kwargs
    )

def conv1x1(filters, strides=1, name=None):
    return lq.layers.QuantConv2D(
        filters, (1, 1), padding="same", strides=strides, name=name, **kwargs
    )


def conv3x3_bn(filters, kernel_size=(3, 3), strides=1, name=None):
    return Sequential(
        [
            lq.layers.QuantConv2D(
                filters, kernel_size, padding="same", strides=strides, name=name, **kwargs
            ),
            BatchNormalization(scale=False),
        ],
        name=name,
    )


def conv1x1_bn(filters, kernel_size=(3, 3), strides=1, name=None):
    return Sequential(
        [
            lq.layers.QuantConv2D(
                filters, kernel_size, padding="same", strides=strides, name=name, **kwargs
            ),
            BatchNormalization(scale=False),
        ],
        name=name,
    )


def conv3x3_pool_bn(filters, strides=1, name=None):
    return Sequential(
        [
            lq.layers.QuantConv2D(
                filters, (3, 3), padding="same", strides=strides, name=name, **kwargs
            ),
            MaxPooling2D((2, 2), strides=(2, 2)),
            BatchNormalization(scale=False),
        ],
        name=name,
    )


def depthwise_conv3x3_bn(strides=1, name=None):
    return Sequential(
        [
            lq.layers.QuantDepthwiseConv2D(
                (3, 3), padding="same", strides=strides, name=name,
            ),
            BatchNormalization(scale=False),
        ],
        name=name,
    )


def lb_activation(filters, activation="ste_sign", name=None):
    return Sequential(
        [
            LearnableBias(filters, name=name + "_bias"),
            Activation(activation, name=name + "_activation"),
        ],
        name=name + "_lb_activation",
    )

def basic_block(in_channels, out_channels, strides=1, name=None):
    return Sequential(
        [
            # 3x3 conv
            lb_activation(filters=in_channels, name="lb_activation_1"),
            conv3x3_bn(filters=out_channels, strides=strides),
            # relu
            lb_activation(filters=out_channels, activation="relu", name="lb_activation_2"),
            # 1x1 conv
            lb_activation(filters=out_channels, name="lb_activation_3"),
            conv1x1_bn(filters=out_channels),
            # relu
            lb_activation(filters=out_channels, activation="relu", name="lb_activation_4"),
        ],
        name=name,
    )


class BasicBlock(Model):
    def __init__(self, in_channels, out_channels, strides=1, name=None):
        super().__init__()
        self.strides = strides
        self.conv1 = Sequential([
            lb_activation(filters=in_channels, name="lb_activation_1"),
            conv3x3_bn(filters=out_channels, strides=strides),
        ])
        self.avgpool = AveragePooling2D((2, 2), strides=(2, 2))
        self.add = Add()
        self.lb_activation = lb_activation(filters=out_channels, activation="relu", name="lb_activation_2")
        # self.conv2 = Sequential([
        #     lb_activation(filters=out_channels, name="lb_activation_3"),
        #     conv1x1_bn(filters=out_channels),
        # ])
        # self.lb_activation2 = lb_activation(filters=out_channels, activation="relu", name="lb_activation_4")


    def call(self, inputs):
        if self.strides == 1:
            out = self.conv1(inputs)
        else:
            x = self.conv1(inputs)
            avg = self.avgpool(inputs)
            out = self.add([x, avg])

        out = self.lb_activation(out)
        # out = self.conv2(out)
        # out = self.lb_activation2(out)
        return out


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

        # depthwise
        self.depthwise_block1 = self.Depthwise_Block(32, 3, name="depthwise_block1")
        self.depthwise_block2 = self.Depthwise_Block(64, 3, name="depthwise_block2")
        self.depthwise_block3 = self.Depthwise_Block(128, 3, name="depthwise_block3")

        # self.cnn_block1 = self.CNN_Block(64, (3, 3))
        # self.cnn_block2 = self.CNN_Block(128, (3, 3))
        # self.cnn_block3 = self.CNN_Block(256, (3, 3))

        # dropout
        self.dropout = Dropout(0.5)

    def CNN_Block(self, filters, kernel_size, strides=1, name=None):
        return Sequential([
            lq.layers.QuantConv2D(filters, kernel_size, padding='same', strides=strides, **kwargs),
            MaxPooling2D((2, 2)),
            BatchNormalization(scale=False),
        ], name=name)

    def Depthwise_Block(self, filters, kernel_size, strides=1, name=None):
        # dw + pw
        return Sequential([
            lq.layers.QuantDepthwiseConv2D(kernel_size, padding='same', strides=strides),
            MaxPooling2D((2, 2)),
            BatchNormalization(scale=False),
            lq.layers.QuantConv2D(filters, (1, 1), padding='same', strides=strides, **kwargs),
            BatchNormalization(scale=False),
        ], name=name)

    def call(self, inputs):
        x = self.qConv2d1(inputs)
        x = self.bn1(self.maxpool1(x))
        # x = self.cnn_block1(x)
        # x = self.cnn_block2(x)
        # x = self.cnn_block3(x)
        x = self.depthwise_block1(x)
        x = self.depthwise_block2(x)
        x = self.depthwise_block3(x)
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
        self.shortcut = lq.layers.QuantConv1D(
            filters, 1,
            padding='same',
            name=f"shortcut_{factor}",
            **kwargs
        )
        self.bn2 = BatchNormalization(scale=False)
        self.relu = Activation('relu')
        self.add = Add()

    def call(self, inputs):
        shortcut = self.shortcut(inputs)

        x = self.conv1(inputs)
        x = self.relu(self.bn1(x))
        x = self.conv2(x)
        x = self.relu(self.bn2(x))

        x = self.add([x, shortcut])
        x = self.relu(x)
        return x


class BTCN(Model):
    def __init__(self, features=128, output_dim=10, **kwargs):
        super().__init__()
        self.features = features
        self.output_dim = output_dim
        self.dense_softmax = lq.layers.QuantDense(self.output_dim, activation="softmax", **kwargs)
        # resblock
        for i in range(3):
            setattr(self, f"resblock{i}", ResBlock(features, i))

    def call(self, inputs):
        x = inputs

        for i in range(3):
            x = getattr(self, f"resblock{i}")(x)

        # softmax dense
        x = self.dense_softmax(x)
        return x


class TinyLPR(Model):
    def __init__(self, features=128, output_dim=85, tcn_blocks=3, train=False, **kwargs):
        super().__init__()
        self.features = features
        self.output_dim = output_dim
        self.tcn_blocks = tcn_blocks
        self.train = train

        self.qConv2d1 = lq.layers.QuantConv2D(
            64, (3, 3), strides=1,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            padding='same',
            use_bias=False,
        )
        self.maxpool1 = MaxPooling2D((2, 2))
        self.bn1 = BatchNormalization(momentum=0.9)

        self.basic_block1 = BasicBlock(64, 64, strides=2, name="basic_block_1")
        self.basic_block2 = BasicBlock(64, 128, strides=1, name="basic_block_2")
        self.basic_block3 = BasicBlock(128, 128, strides=2, name="basic_block_3")
        # self.basic_block4 = BasicBlock(128, 128, strides=1, name="basic_block_4")
        self.conv2d_last = Sequential([
            lb_activation(filters=128, name="lb_activation_1"),
            conv3x3_bn(filters=128, kernel_size=(1, 6), name="conv2d_1"),
        ], name="conv2d_last")

        # dense
        self.dense1 = lq.layers.QuantDense(features, activation="relu", **kwargs)

        # resblock
        for i in range(self.tcn_blocks):
            setattr(self, f"resblock{i}", ResBlock(features, i))

        self.dense_softmax = lq.layers.QuantDense(self.output_dim, activation="softmax", **kwargs)
        self.ctc = CTCLayer(name="ctc_loss")

    def Conv2D_Block(self, filters, kernel_size, strides=1, name=None):
        return Sequential([
            lq.layers.QuantConv2D(filters, kernel_size, padding='same', strides=strides, **kwargs),
            BatchNormalization(momentum=0.999, scale=False),
        ], name=name)

    def Depthwise_Block(self, filters, kernel_size, strides=1, name=None):
        # dw + pw
        return Sequential([
            lq.layers.QuantDepthwiseConv2D(
                kernel_size,
                strides=strides,
                padding='same',
                kernel_constraint="weight_clip",
                use_bias=False,
            ),
            BatchNormalization(momentum=0.9),
            Activation("relu"),
            lq.layers.QuantConv2D(filters, (1, 1), padding='same', strides=strides, **kwargs),
            BatchNormalization(momentum=0.9),
            Activation("relu"),
        ], name=name)

    def call(self, inputs):
        x = inputs

        x = self.qConv2d1(x)
        x = self.maxpool1(x)
        x = self.bn1(x)

        x = self.basic_block1(x)
        x = self.basic_block2(x)
        x = self.basic_block3(x)
        # x = self.basic_block4(x)

        x = self.conv2d_last(x)

        n, h, w, c = x.shape
        x = tf.split(x, num_or_size_splits=2, axis=2)
        x = Concatenate(axis=1)([x[0], x[1]])
        x = Reshape((h, w*c))(x)
        
        # dense
        x = self.dense1(x)

        for i in range(self.tcn_blocks):
            x = getattr(self, f"resblock{i}")(x)

        # softmax dense
        x = self.dense_softmax(x)

        if self.train:
            # input shape: [batch_size, max_label_length]
            labels = Input(name='labels', shape=(MAX_LABEL_LEN,), dtype='int64')
            # ctc
            x = self.ctc(labels, x)
            return x
        else:
            return x


if __name__ == "__main__":
    model = TinyLPR(train=False)
    # model = BTCN()
    # model = ResBlock(128, 1)
    # model = BCNN()
    model.build(input_shape=(None, 96, 48, 1))

    labels = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8]], dtype="int32")
    x = tf.random.normal((1, 96, 48, 1), dtype=tf.float32)
    y = model(x)
    print(y.shape)

    model.summary()
    # save
    model.save_weights("bnn.h5")
    lq.models.summary(model)
