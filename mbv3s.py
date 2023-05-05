import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *
# from keras_flops import get_flops

from loss import CTCLayer, ACELayer


MAX_LABEL_LENGTH = 8


class BottleNeck(Layer):
    def __init__(self, in_size, exp_size, out_size, s, k=3):
        super(BottleNeck, self).__init__()
        self.stride = s
        self.in_size = in_size
        self.out_size = out_size
        self.conv1 = Conv2D(
            filters=exp_size,
            kernel_size=(1, 1),
            strides=1,
            padding="same")
        self.bn1 = BatchNormalization()
        self.dwconv = DepthwiseConv2D(
            kernel_size=(k, k),
            strides=s,
            padding="same")
        self.bn2 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=out_size,
            kernel_size=(1, 1),
            strides=1,
            padding="same")
        self.bn3 = BatchNormalization()
        self.linear = Activation(tf.keras.activations.linear)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu6(x)

        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu6(x)

        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = self.linear(x)

        if self.stride == 1 and self.in_size == self.out_size:
            x = add([x, inputs])

        return x


class MobileNetV3Small(tf.keras.Model):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=2,
            padding="same")
        self.act1 = Activation(tf.nn.relu6)
        self.bn1 = BatchNormalization()
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=2, k=3)
        self.bneck2 = BottleNeck(in_size=16, exp_size=72, out_size=24, s=2, k=3)
        self.bneck3 = BottleNeck(in_size=24, exp_size=88, out_size=24, s=1, k=3)

        self.bneck4 = BottleNeck(in_size=24, exp_size=96, out_size=40, s=2, k=5)
        self.bneck5 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, k=5)
        self.bneck6 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, k=5)
        self.bneck7 = BottleNeck(in_size=40, exp_size=120, out_size=48, s=1, k=5)
        self.bneck8 = BottleNeck(in_size=48, exp_size=144, out_size=48, s=1, k=5)

        self.bneck9 = BottleNeck(in_size=48, exp_size=288, out_size=96, s=2, k=5)
        self.bneck10 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, k=5)
        self.bneck11 = BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, k=5)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.bneck1(x, training=training)
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        c1 = x

        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        x = self.bneck7(x, training=training)
        x = self.bneck8(x, training=training)
        c2 = x

        x = self.bneck9(x, training=training)
        x = self.bneck10(x, training=training)
        x = self.bneck11(x, training=training)
        c3 = x

        return c1, c2, c3


class TCN(tf.keras.Model):
    def __init__(self, seq_len, filters, kernel_size=2, blocks=2):
        super(TCN, self).__init__()
        self.seq_len = seq_len
        self.filters = filters
        self.kernel_size = kernel_size
        self.blocks = blocks

    def ResBlock(self, factor, inputs):
        dilation_rate = 2 ** factor

        x = Conv1D(self.filters, self.kernel_size, padding='same', dilation_rate=dilation_rate)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(0.05)(x)

        x = Conv1D(self.filters, self.kernel_size, padding='same', dilation_rate=dilation_rate)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(0.05)(x)

        shortcut = Conv1D(self.filters, 1, padding='same')(inputs)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def call(self, inputs):
        x = inputs
        for i in range(self.blocks):
            x = self.ResBlock(i, x)
        return x


class Head(tf.keras.Model):
    def __init__(self):
        super(Head, self).__init__()
        self.ctc = CTCLayer()
        self.ace = ACELayer()

    def call(self,
        ctc, ctc_labels,
        ace, ace_labels):
        ctc_loss = self.ctc(ctc_labels, ctc)
        ace_loss = self.ace(ace_labels, ace)
        return ctc_loss, ace_loss


class TinyLPR(tf.keras.Model):
    def __init__(self, bs, shape, output_dim=86, tcn_ksize=2, tcn_blocks=2, train=True):
        super(TinyLPR, self).__init__()
        self.bs = bs
        self.shape = shape
        self.output_dim = output_dim
        self.train = train
        # backbone and merge layer
        self.model = MobileNetV3Small()
        self.upsample = UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.pooling = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
        # # tcn
        # self.tcn = TCN(seq_len=16, filters=128, kernel_size=tcn_ksize, blocks=tcn_blocks)
        # out
        self.dense_softmax = Dense(self.output_dim, activation='softmax', name='softmax0')
        self.loss = Head()
        # Define inputs here
        self.input_tensor = Input(shape=self.shape, dtype=tf.float32, batch_size=self.bs)
        self.ace_label_tensor = Input(shape=(self.output_dim,), dtype=tf.int64, batch_size=self.bs)
        self.ctc_label_tensor = Input(shape=(MAX_LABEL_LENGTH,), dtype=tf.int64, batch_size=self.bs)

    def build(self, input_shape):

        c1, c2, c3 = self.model(self.input_tensor, training=self.train)
        c1 = self.pooling(c1)
        c3 = self.upsample(c3)
        f_map = tf.concat([c1, c2, c3], axis=-1)
        # ACE loss
        ace = self.dense_softmax(f_map)

        x = tf.split(f_map, num_or_size_splits=2, axis=1)
        x = tf.concat(x, axis=2)
        out = tf.reduce_mean(x, axis=1)
        # x = self.tcn(x)
        # CTC loss
        ctc = self.dense_softmax(out)

        if self.train:
            loss = self.loss(
                ctc, self.ctc_label_tensor,
                ace, self.ace_label_tensor,
            )
            return Model(
                inputs=[
                    self.input_tensor,
                    self.ctc_label_tensor,
                    self.ace_label_tensor,
                ],
                outputs=loss
            )
        else:
            return Model(inputs=self.input_tensor, outputs=out)

    def call(self, inputs):
        return self.build(inputs.shape)(inputs)


if __name__ == '__main__':
    input_shape = (64, 128, 1)
    bs = 1

    model = TinyLPR(
        bs=bs,
        shape=input_shape,
        train=True,
    ).build(input_shape=[
        (bs, *input_shape),
        (bs, MAX_LABEL_LENGTH),
        (bs, 86),
    ])
    model.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
        loss=lambda y_true, y_pred: y_pred
    )
    model.summary()
    # save
    model.save("tinyLPR.h5")

    inputs = tf.random.normal(shape=(1, *input_shape))
    ctc_labels = tf.random.uniform(shape=(1, MAX_LABEL_LENGTH), minval=0, maxval=85, dtype=tf.int64)
    ace_labels = tf.random.uniform(shape=(1, 86), minval=0, maxval=85, dtype=tf.int64)
    loss = model(
        [inputs, ctc_labels, ace_labels],
        training=True,
    )
    print(loss)
