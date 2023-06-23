from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *
from keras_flops import get_flops

from loss import *


MAX_LABEL_LEN = 10


class BCE_loss(Layer):
    def __init__(self, name="bce_loss"):
        super(BCE_loss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
        return loss


# CBA, return Sequential
def CBA(filters, kernel_size=3, strides=1, padding='same', dilation_rate=1, activation='relu', use_bias=False, name=None, train=False):
    return Sequential([
        Conv2D(filters, kernel_size, strides=strides, padding=padding, dilation_rate=dilation_rate, use_bias=use_bias),
        BatchNormalization(trainable=train),
        Activation(activation),
    ], name=name)


class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(3,input_shape[2], self.output_dim),
            initializer='uniform',
            trainable=True
        )
        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (self.output_dim**0.5)
        QK = K.softmax(QK)
        V = K.batch_dot(QK,WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)


def Attn(x_local, x_global, out_dim, train=False):
    _local = Sequential([
        # dw
        DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding="same", kernel_initializer='he_normal', use_bias=False),
        BatchNormalization(trainable=train),
        Activation(tf.nn.relu6),
        # pw
        Conv2D(filters=out_dim, kernel_size=(1, 1), strides=1, padding="same", kernel_initializer='he_normal', use_bias=False),
        BatchNormalization(trainable=train),
        Activation(tf.nn.relu6),
    ], name="local")
    _global = Sequential([
        GlobalAveragePooling2D(keepdims=True),
        Conv2D(out_dim//4, (1, 1), padding='same', activation='relu'),
        Conv2D(out_dim, (1, 1), padding='same', activation='sigmoid'),
    ], name="global")
    # multi
    x = tf.multiply(_global(x_global), _local(x_local))
    return x


class SEBlock(Layer):
    def __init__(self, input_channels, r=4):
        super(SEBlock, self).__init__()
        self.pool = GlobalAveragePooling2D(keepdims=True)
        self.fc1 = Dense(units=input_channels // r)
        self.fc2 = Dense(units=input_channels)

    def call(self, inputs, **kwargs):
        x = self.pool(inputs)
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.sigmoid(x)
        output = inputs * x
        return output


class BottleNeck(Layer):
    def __init__(self, in_size, exp_size, out_size, s, k=3, se=False, name="block"):
        super(BottleNeck, self).__init__()
        self.stride = s
        self.in_size = in_size
        self.out_size = out_size
        self.se = se

        self.pw_in = Sequential([
            Conv2D(filters=exp_size, kernel_size=(1, 1), strides=1, padding="same", kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            Activation(tf.nn.relu6)
        ], name="{}/pw_in".format(name))
        
        self.dw = Sequential([
            DepthwiseConv2D(kernel_size=(k, k), strides=s, padding="same", kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            Activation(tf.nn.relu6),
        ], name="{}/dw".format(name))

        self.se_block = SEBlock(exp_size)

        self.pw_out = Sequential([
            Conv2D(filters=out_size, kernel_size=(1, 1), strides=1, padding="same", kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
        ], name="{}/pw_out".format(name))

    def call(self, inputs, training=None, **kwargs):
        # shortcut
        x = self.pw_in(inputs)
        x = self.dw(x)

        if self.se:
            x = self.se_block(x)

        x = self.pw_out(x)

        if self.stride == 1 and self.in_size == self.out_size:
            x = add([x, inputs])

        return x


class MobileNetV3Small(tf.keras.Model):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name="conv1")
        self.bn1 = BatchNormalization(name="bn1")
        self.act1 = Activation(tf.nn.relu6, name="act1")
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=2, k=3)
        self.bneck2 = BottleNeck(in_size=16, exp_size=72, out_size=24, s=1, k=3)
        self.bneck3 = BottleNeck(in_size=24, exp_size=88, out_size=24, s=1, k=3)

        self.bneck4 = BottleNeck(in_size=24, exp_size=96, out_size=40, s=2, k=5)
        self.bneck5 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, k=5)
        self.bneck6 = BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, k=5)
        self.bneck7 = BottleNeck(in_size=40, exp_size=120, out_size=48, s=1, k=5)
        self.bneck8 = BottleNeck(in_size=48, exp_size=144, out_size=48, s=1, k=5)

        self.bneck9 = BottleNeck(in_size=48, exp_size=288, out_size=96, s=2, k=5)

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
        c3 = x

        return c1, c2, c3


class TinyLPR(tf.keras.Model):
    def __init__(self, shape, output_dim=86, train=True, with_mask=False, **kwargs):
        super(TinyLPR, self).__init__()
        self.shape = shape
        self.output_dim = output_dim
        self.train = train
        self.with_mask = with_mask
        # backbone and merge layer
        self.model = MobileNetV3Small()
        self.upsample = UpSampling2D(size=(2, 2), interpolation="bilinear", name='upsample')

        # skip connection
        self.skip = Conv2D(128, 1, strides=1, padding='same', kernel_initializer='he_normal', name='c1_conv')

        self.mask_head = Sequential([
            UpSampling2D(size=(2, 2), interpolation="bilinear", name='seg_upsample1'),
            Conv2D(1, 1, padding='same', kernel_initializer='he_normal', name='seg_mask', activation='sigmoid'),
            UpSampling2D(size=(4, 4), interpolation="bilinear", name='seg_upsample2'),
        ], name='mask_head')

        # out
        self.dense = Dense(self.output_dim, kernel_initializer='he_normal', name='dense_last')
        self.softmax = Activation('softmax', dtype='float32', name='act_softmax')
        
        # ctc loss
        self.ctc_loss = FocalLossLayer(name="ctc")
        # bce loss
        self.mask_loss = BCE_loss(name="bce")
        # self.mask_loss = DiceLoss(name="dice")

        # inputs
        self.input_tensor = Input(shape=self.shape, dtype=tf.float32, name='input0')
        self.ctc_label_tensor = Input(shape=(MAX_LABEL_LEN,), dtype=tf.int64, name='input_ctc')
        self.mask_label_tensor = Input(shape=self.shape, dtype=tf.int64, name='input_mask')

        # dropout
        self.dropout = Dropout(0.5, trainable=self.train, name='dropout')

    def build(self, input_shape):
        # backbone
        c1, c2, c3 = self.model(self.input_tensor, training=self.train)
        f_map = c3 # (bs, 4, 8, 96)

        attn = Attn(x_global=f_map, x_local=c2, out_dim=128, train=self.train)
        concat = attn
        # attn = self.upsample(attn)

        # skip = self.skip(c2)
        # concat = tf.multiply(skip, attn, name='multiply')

        bs, h, w, c = tf.shape(concat)[0], tf.shape(concat)[1], tf.shape(concat)[2], tf.shape(concat)[3]

        # flatten
        x = tf.reshape(concat, (bs, h*w, c), name='flatten_reshape')
        # # layer norm
        # x = LayerNormalization(axis=-1, name='layer_norm')(x)

        if self.train:
            x = self.dropout(x)

        reshaped = x
        # softmax
        x = self.dense(x)
        ctc = self.softmax(x)

        if self.train:
            ctc_loss = self.ctc_loss(self.ctc_label_tensor, ctc)

            if self.with_mask:
                mask = self.mask_head(concat) # (bs, 64, 128, 86)
                mask_loss = self.mask_loss(self.mask_label_tensor, mask)
                loss = ctc_loss, mask_loss
            else:
                loss = ctc_loss

            return Model(
                inputs=[self.input_tensor, self.ctc_label_tensor, self.mask_label_tensor],
                outputs=loss,
                name='tinyLPR',)
        else:
            return Model(inputs=self.input_tensor, outputs=ctc, name='tinyLPR')

    def call(self, inputs):
        return self.build(inputs.shape)(inputs)


if __name__ == '__main__':
    input_shape = (64, 128, 1)

    model = TinyLPR(
        shape=input_shape,
        train=True,
        with_mask=True,
    ).build(input_shape=[
        (None, *input_shape), (None, MAX_LABEL_LEN), (None, *input_shape)
    ])
    model.save("tinyLPR.h5")

    model = TinyLPR(
        shape=input_shape,
        train=False,
        with_mask=False,
    ).build(input_shape=(None, *input_shape))
    model.summary()
    # model.save("tinyLPR_deploy.h5")
    model.load_weights("best_model.h5")

    # get flops
    flops = get_flops(model, 1)
    # to mflops and keep 2 decimal places
    flops = round(flops / 10.0 ** 6, 2)
    print(f"FLOPS: {flops} MFLOPS")
