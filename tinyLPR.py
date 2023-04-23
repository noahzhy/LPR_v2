import numpy as np
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K

from net_flops import net_flops


class TCN:
    def __init__(self, seq_len=10, features=64, num_layers=6, kernel_size=3):
        self.seq_len = seq_len
        self.features = features
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.model = self.build_model()

    def build_model(self):        
        inputs = Input(shape=(self.seq_len, self.features), name='the_input')
        x = inputs

        for i in range(self.num_layers):
            x = self.ResBlock(x, self.features, self.kernel_size, 2 ** i)

        # softmax, dense
        x = Dense(85, activation='softmax', name='softmax')(x)
        return Model(inputs=inputs, outputs=x)

    def ResBlock(self, x, features, kernel_size, dilation_rate):
        r = Conv1D(features, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
        r = Conv1D(features, kernel_size, padding='same', dilation_rate=dilation_rate)(r)

        if x.shape[-1] == features:
            shortcut = x
        else:
            shortcut = Conv1D(features, kernel_size, padding='same')(x)
        o = add([r, shortcut])
        o = Activation('relu')(o)
        return o

    def __call__(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    features = 128

    model = TCN(20, features, 4, 3)
    # x = np.random.random((1, 20, features))
    # x = np.ones(shape=(1, 20, features))
    # gererate data shape=(batch_size, look_back, features) value=[0, 1] linear
    x = np.linspace(0, 1, 20 * features).reshape((1, 20, features))
    # predict
    y = model(x)
    tcn_max = tf.argmax(y, axis=-1)
    print(tcn_max)
    # summary
    model.model.summary()

    # y.summary()
    # model.model.save('TCN.h5')
    # # summary
    # model.model.summary()
    # # flops
    # flops = net_flops(model.model)
