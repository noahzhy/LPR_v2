import numpy as np
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K
from tensorflow.keras.applications import MobileNetV3Small

from keras_flops import get_flops

from net_flops import net_flops
from ctc import CTCLayer, ACELayer


MAX_LABEL_LEN = 8

class TinyLPR(Model):
    def __init__(self,
        shape=(96, 48, 1),
        seq_len=24,
        filters=128,
        blocks=2,
        kernel_size=2,
        output_dim=86,
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

    def CNN_Block(self):
        # inputs = Input(shape=self.shape, name='image')
        nn = MobileNetV3Small(
            input_shape=self.shape,
            weights=None,
            include_top=True
        )

        print(nn.summary())
        # save
        nn.save('mobilenetv3small.h5')
        quit()
        
        #! Default batch norm is configured for huge networks, let's speed it up
        for layer in nn.layers:
            if type(layer) == BatchNormalization:
                layer.momentum = 0.9
        #! Cut MobileNet where it hits 1/8th input resolution; i.e. (HW/8, HW/8, C)
        x = nn.get_layer('block_6_expand_relu').output

        x = tf.split(x, num_or_size_splits=2, axis=2)
        x = tf.concat(x, axis=1)
        x = Conv2D(filters=self.filters, kernel_size=(1, 3), padding='valid', activation='relu')(x)
        x = tf.squeeze(x, axis=2)

        return Model(inputs=nn.input, outputs=x, name='CNN')

    def ResBlock(self, factor):
        dilation_rate = 2 ** factor

        inputs = Input(shape=(self.seq_len, self.filters), name='input_tcn')
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
        return Model(inputs=inputs, outputs=x, name='ResBlock_{}'.format(factor))

    def build(self, input_shape):
        inputs = Input(shape=input_shape)
        x = self.CNN_Block()(inputs)

        # shortcut = x

        # for i in range(self.blocks):
        #     block = self.ResBlock(i)
        #     x = block(x)

        # x = add([x, shortcut])
        # softmax, dense
        x = Dense(self.output_dim, name='softmax0')(x)
        x = tf.nn.softmax(x, axis=-1)

        # ctc
        if self.train:
            labels = Input(name='labels', shape=(MAX_LABEL_LEN,), dtype='int64')
            ctc = CTCLayer(name='ctc_loss')(labels, x)
            # ace = ACELayer(name='ace_loss')(labels, x)
            return Model(inputs=[inputs, labels], outputs=ctc, name='TinyLPR')

        return Model(inputs=inputs, outputs=x, name='TinyLPR')

    def call(self, inputs):
        return self.build(inputs.shape)(inputs)


if __name__ == "__main__":
    w = 48
    h = 2 * w
    features = 128
    input_shape = (h, w, 1)
    model = TinyLPR(
        shape=input_shape,
        output_dim=86,
        seq_len=w//2,
        blocks=2,
        filters=features,
        train=False
    ).build(input_shape)
    # model.summary()
    # model.compile(
    #     optimizer=Adam(lr=0.001),
    #     metrics=['accuracy']
    # )

    model.predict(np.random.rand(1, h, w, 1))

    flops = get_flops(model, batch_size=1)
    # to M FLOPS
    print('FLOPS: %.3f M' % (flops/1e6))
