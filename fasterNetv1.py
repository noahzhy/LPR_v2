import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *
from keras_flops import get_flops
from keras_cv_attention_models import fasternet

from loss import CTCLayer, ACELayer


MAX_LABEL_LENGTH = 9


class TCN(tf.keras.Model):
    def __init__(self, seq_len, filters, kernel_size=2, blocks=2, train=False, **kwargs):
        super(TCN, self).__init__()
        self.seq_len = seq_len
        self.filters = filters
        self.kernel_size = kernel_size
        self.blocks = blocks
        self.train = train

        # set up layers via blocks
        for i in range(self.blocks):
            setattr(self, f'block_{i}', self.ResBlock(i))

    def ResBlock(self, factor):
        dilation_rate = 2 ** factor

        # input
        inputs = Input(shape=(self.seq_len, self.filters))

        shortcut = Conv1D(self.filters, 1, padding='same', kernel_initializer='he_normal')(inputs)
        x = Conv1D(self.filters, self.kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(0.1, trainable=self.train)(x)

        x = Conv1D(self.filters, self.kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(0.1, trainable=self.train)(x)

        x = add([x, shortcut])
        x = Activation('relu')(x)
        return Model(inputs, x, name=f'resblock_{factor}')

    def call(self, inputs):
        x = inputs
        for i in range(self.blocks):
            x = getattr(self, f'block_{i}')(x)
        return x


class TinyLPR(tf.keras.Model):
    def __init__(self, bs, shape, output_dim=86, tcn_ksize=2, tcn_blocks=2, train=True):
        super(TinyLPR, self).__init__()
        self.bs = bs
        self.shape = shape
        self.output_dim = output_dim
        self.train = train
        # backbone and merge layer
        nn = fasternet.FasterNet(
            input_shape=(64, 128, 1),
            activation='relu',
        )
        # get layers
        c1 = nn.get_layer('stack1_block1_output').output
        c2 = nn.get_layer('stack2_block2_output').output
        c3 = nn.get_layer('stack3_block7_output').output
        self.model = Model(inputs=nn.input, outputs=[c1, c2, c3], name='fasterNet')
        self.upsample = UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        # tcn
        self.tcn = TCN(
            seq_len=128,
            filters=280,
            kernel_size=tcn_ksize,
            blocks=tcn_blocks,
            train=self.train)
        # out
        self.dense_softmax = Sequential([
            Dense(self.output_dim),
            BatchNormalization(trainable=self.train),
            Activation('softmax'),
        ], name='dense_softmax')
        self.ctc_loss = CTCLayer(name="ctc_loss")
        self.ace_loss = ACELayer(name="ace_loss")
        # inputs
        self.input_tensor = Input(shape=self.shape, dtype=tf.float32, batch_size=self.bs, name='input0')
        self.ace_label_tensor = Input(shape=(self.output_dim,), dtype=tf.int64, batch_size=self.bs, name='input_ace')
        self.ctc_label_tensor = Input(shape=(MAX_LABEL_LENGTH,), dtype=tf.int64, batch_size=self.bs, name='input_ctc')
        # dropout
        self.dropout = SpatialDropout1D(0.2, trainable=self.train)

    def build(self, input_shape):
        # backbone
        c1, c2, c3 = self.model(self.input_tensor)
        # c1
        c1 = self.pooling(c1)
        c3 = self.upsample(c3)

        # cat
        f_map = concatenate([c1, c2, c3], axis=-1)

        # bn
        x = BatchNormalization(trainable=self.train)(f_map)

        x = tf.reshape(x, (-1, 128, 280))

        shortcut = x
        x = self.tcn(x)
        x = self.dropout(x)
        x = add([x, shortcut])

        # CTC loss
        ctc = self.dense_softmax(x)
        ace = self.dense_softmax(x)

        if self.train:
            ctc_loss = self.ctc_loss(self.ctc_label_tensor, ctc)
            ace_loss = self.ace_loss(self.ace_label_tensor, ace)

            loss = ctc_loss + ace_loss*.1
            return Model(
                inputs=[self.input_tensor, self.ctc_label_tensor, self.ace_label_tensor],
                outputs=loss
            )
        else:
            return Model(inputs=self.input_tensor, outputs=ctc)

    def call(self, inputs):
        return self.build(inputs.shape)(inputs)


if __name__ == '__main__':
    input_shape = (64, 128, 1)
    bs = 1

    model = TinyLPR(
        bs=bs,
        shape=input_shape,
        train=False,
    ).build(input_shape=(bs, *input_shape))
    model.summary()
    # save
    model.save("tinyLPR_deploy.h5")

    flops = get_flops(model, 1)
    # to mflops and keep 2 decimal places
    flops = round(flops / 10.0 ** 6, 2)
    print(f"FLOPS: {flops} MFLOPS")

    # fasternet = fasternet.FasterNet(
    #     input_shape=(64, 128, 1),
    #     activation='relu',
    # )
    # # get layers
    # output = fasternet.get_layer('stack3_block7_output').output
    # model = Model(inputs=fasternet.input, outputs=output)
    # model.summary()
    # # save
    # model.save("fasternet_deploy.h5")

