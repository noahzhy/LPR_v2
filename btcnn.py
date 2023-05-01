import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

import keras
import larq as lq
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.datasets import mnist



kwargs = dict(
    use_bias=False,
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip",
)


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


def conv3x3_bn(filters, kernel_size=(3, 3), strides=1, padding="same", name=None):
    return Sequential(
        [
            lq.layers.QuantConv2D(
                filters, kernel_size, padding=padding, strides=strides, name=name, **kwargs
            ),
            BatchNormalization(momentum=0.999, scale=False),
        ],
        name=name,
    )


def conv1x1_bn(filters, kernel_size=(3, 3), strides=1, padding="same", name=None):
    return Sequential(
        [
            lq.layers.QuantConv2D(
                filters, kernel_size, padding=padding, strides=strides, name=name, **kwargs
            ),
            BatchNormalization(momentum=0.999, scale=False),
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
            BatchNormalization(momentum=0.999, scale=False),
        ],
        name=name,
    )


def depthwise_conv3x3_bn(strides=1, name=None):
    return Sequential(
        [
            lq.layers.QuantDepthwiseConv2D(
                (3, 3), padding="same", strides=strides, name=name,
            ),
            BatchNormalization(momentum=0.999, scale=False),
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


def lba_conv3x3_lba(
        in_channels, 
        out_channels,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        name=None,
    ):
    return Sequential(
        [
            # 3x3 conv
            lb_activation(filters=in_channels, name="lb_activation_1"),
            conv3x3_bn(filters=out_channels, kernel_size=kernel_size, strides=strides, padding=padding, name="conv3x3_bn_1"),
            # relu
            lb_activation(filters=out_channels, activation="relu", name="lb_activation_2"),
        ],
        name=name,
    )


class BasicBlock(Model):
    def __init__(self, in_channels, out_channels, strides=1, name=None):
        super().__init__()
        self.strides = strides
        self.conv1 = Sequential([
            lb_activation(filters=in_channels, name="lb_activation_1"),
            conv3x3_bn(filters=in_channels, strides=strides),
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
            out = self.add([out, inputs])
        else:
            x = self.conv1(inputs)
            avg = self.avgpool(inputs)
            out = self.add([x, avg])

        out = self.lb_activation(out)
        # out = self.conv2(out)
        # out = self.lb_activation2(out)
        return out


class ResBlock(Model):
    def __init__(self, filters, factor, kernel_size=2, strides=1, dropout=.2, name=None):
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
        self.bn1 = BatchNormalization(momentum=0.999, scale=False)
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
        self.bn2 = BatchNormalization(momentum=0.999, scale=False)
        self.relu = Activation('relu')
        self.add = Add()
        # dropout 0.2
        self.dropout = Dropout(dropout)

    def call(self, inputs):
        shortcut = self.shortcut(inputs)

        x = self.conv1(inputs)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)

        x = self.add([x, shortcut])
        x = self.relu(x)
        return x


class BTCN(Model):
    def __init__(self,
        seq_len=20,
        num_feat=1,
        num_filters=128,
        num_classes=10,
        kernel_size=2,
        blocks=3,
        dropout=0.2,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_feat = num_feat
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.dropout = dropout

        # resblock
        for i in range(self.blocks):
            setattr(self, f"resblock{i}", ResBlock(
                num_filters, i,
                kernel_size=kernel_size,
                dropout=dropout, 
                name=f"resblock{i}"
            ))

        # softmax dense
        self.dense_softmax = lq.layers.QuantDense(self.num_classes, **kwargs)
        # bn
        self.bn = BatchNormalization(momentum=0.999, scale=False)
        # softmax
        self.softmax = Activation("softmax")

    def build(self):
        x = inputs = Input(shape=(self.seq_len, self.num_feat))

        for i in range(self.blocks):
            x = getattr(self, f"resblock{i}")(x)

        x = Flatten()(x)
        # softmax dense
        x = self.dense_softmax(x)
        x = self.softmax(self.bn(x))

        return Model(inputs=inputs, outputs=x)

    def call(self, inputs):
        return self.build(input_shape=inputs.shape)(inputs)


if __name__ == "__main__":

    input_shape = (28, 28)
    num_classes = 10


    (train_x, train_y), (test_x, test_y) = data_generator()
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


    # import matplotlib.pyplot as plt
    
    # # random choose one sample, and plot it
    # idx = np.random.randint(0, len(train_x))
    # img = train_x[idx]
    # label = train_y[idx]
    # plt.imshow(img, cmap="gray")
    # plt.title(f"Label: {label}")
    # plt.show()

    # quit()

    model = BTCN(
        num_classes=num_classes,
        seq_len=28,
        num_feat=28,
        num_filters=32,
        kernel_size=3,
        blocks=3,
        dropout=0.05,
    ).build()

    x = np.random.randn(1, *input_shape)
    y = model(x)
    print(y)

    model.summary()
    # save
    model.save_weights("btcn.h5")
    lq.models.summary(model)


    # Compile model 编译模型
    model.compile(
        optimizer=Adam(lr=0.003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    # Training model 训练模型
    model.fit(
        train_x, train_y,
        batch_size=128,
        epochs=10,
        verbose=2,
        validation_data=(test_x, test_y)
    )
    # Assessment model 评估模型
    pre = model.evaluate(test_x, test_y, batch_size=64, verbose=2)
    print('test_loss:', pre[0], '- test_acc:', pre[1])
