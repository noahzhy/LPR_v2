import numpy as np
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K

from cnn import *
from tcn import *


class CNN_TCN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape, name='the_input')

        cnn = CNN()(inputs)
        tcn = TCN(256, 6, 64)(cnn)

        # softmax, dense
        y_pred = Dense(self.num_classes, activation='softmax', name='softmax')(tcn)

        return Model(inputs=inputs, outputs=y_pred)

    def __call__(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    x = np.random.random((1, 96, 192, 1))
    model = CNN_TCN((96, 192, 1), 85)
    y = model(x)
    print(y.shape)
    model.model.summary()
    model.model.save('CNN_TCN.h5')
