import numpy as np
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K

from ctc import CTCLayer

from net_flops import net_flops

# from keras_flops import get_flops


MAX_LABEL_LEN = 8

class TinyLPR(Model):
    def __init__(self,
        shape=(96, 48, 1),
        seq_len=12,
        filters=128,
        blocks=3,
        kernel_size=3,
        output_dim=85,
        train=False,
        **kwargs
    ):
        super(TinyLPR, self).__init__()
        self.shape = shape
        self.seq_len = seq_len
        self.filters = filters
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.train = train

    def CnnBlock(self):
        inputs = Input(shape=self.shape, name='input')
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)

        n, h, w, c = x.shape
        # split the image into 2 parts via width and concat them via height
        x = tf.split(x, num_or_size_splits=2, axis=2)
        x = Concatenate(axis=1)([x[0], x[1]])
        # reshape to (n, h, w*c)
        x = Reshape((h, w*c))(x)
        x = Dense(self.filters, activation='relu')(x)
        return Model(inputs=inputs, outputs=x, name='CNN')

    def ResBlock(self, factor):
        dilation_rate = 2 ** factor

        inputs = Input(shape=(self.seq_len, self.filters), name='input_tcn')
        r = Conv1D(self.filters, self.kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(inputs)
        r = Conv1D(self.filters, self.kernel_size, padding='causal', dilation_rate=dilation_rate)(r)

        if inputs.shape[-1] == self.filters:
            shortcut = inputs
        else:
            shortcut = Conv1D(self.filters, self.kernel_size, padding='same')(inputs)

        o = add([r, shortcut])
        o = Activation('relu')(o)
        return Model(inputs=inputs, outputs=o, name='ResBlock_{}'.format(factor))

    def build(self, input_shape):
        inputs = Input(shape=input_shape)
        x = self.CnnBlock()(inputs)

        for i in range(self.blocks):
            block = self.ResBlock(i)
            x = block(x)

        # softmax, dense
        x = Dense(self.output_dim, activation='softmax', name='softmax')(x)

        # ctc
        if self.train:
            labels = Input(name='labels', shape=(MAX_LABEL_LEN,), dtype='int64')
            ctc = CTCLayer(name='ctc_loss')(labels, x)
            return Model(inputs=[inputs, labels], outputs=ctc, name='TCN')            

        return Model(inputs=inputs, outputs=x, name='TCN')

    def call(self, inputs):
        return self.build(inputs.shape)(inputs)


if __name__ == "__main__":
    features = 128
    input_shape = (96, 48, 1)
    labels = Input(name='labels', shape=[MAX_LABEL_LEN], dtype='int64')

    model = TinyLPR(
        shape=input_shape,
        output_dim=85,
        train=True
    ).build(input_shape)
    model.summary()
    model.compile(
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )

    # flops = get_flops(model, batch_size=1)
    # # to M FLOPS
    # print('FLOPS: %.3f M' % (flops/1e6))

    # # x = np.random.randn(1, 96, 48, 1)

    # model.compile(
    #     optimizer=Adam(lr=0.001),
    #     metrics=['accuracy'],
    #     loss={'ctc_loss': lambda y_true, y_pred: y_pred}
    # )

    # from utils import *
    # dataLoader = LPGenerate(5, shuffle=True)
    # for i in range(10):
    #     x, y = dataLoader.__getitem__(i)
    #     model.fit(x=x, y=y, batch_size=1, epochs=1, verbose=1)
    #     model.save(filepath='tinyLPR.h5')

