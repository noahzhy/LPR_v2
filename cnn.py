import numpy as np
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K

from keras_flops import get_flops


## DepthSepCov
class DepthSepCov(Layer):
    def __init__(self, feature, kernel_size, strides, padding='same'):
        super(DepthSepCov, self).__init__()
        self.dw = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', activation='relu')
        self.pw = Conv2D(feature, kernel_size=(1, 1), strides=(1, 1), activation='relu')

    def call(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class ResidualBlock(Layer):
    def __init__(self, feature, kernel_size, strides, padding='same'):
        super(ResidualBlock, self).__init__()
        self.pw = DepthSepCov(feature, kernel_size, strides, padding)

    def call(self, x):
        pw = self.conv1(x)
        out = add([x, pw])
        return out


class RepDepthSepCov(Layer):
    def __init__(self, feature, strides, padding='same'):
        super(RepDepthSepCov, self).__init__()
        self.dw_5x5 = DepthwiseConv2D(kernel_size=5, strides=strides, padding='same', activation='relu')
        self.dw_3x3 = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', activation='relu')
        self.dw_1x1 = DepthwiseConv2D(kernel_size=1, strides=strides, padding='same', activation='relu')

    def call(self, x):
        dw5x5 = self.dw_5x5(x)
        bn5x5 = BatchNormalization()(dw5x5)
        dw3x3 = self.dw_3x3(x)
        bn3x3 = BatchNormalization()(dw3x3)
        dw1x1 = self.dw_1x1(x)
        bn1x1 = BatchNormalization()(dw1x1)
        # add
        out = add([bn5x5, bn3x3, bn1x1])
        return out


class SteamConv(Layer):
    def __init__(self):
        super(SteamConv, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), activation='relu')
    
    def call(self, x):
        y = self.conv1(x)
        return y
    

class CNN:
    def __init__(self, input_shape=(48, 192, 1)):
        ''' simple version
        input                 28^2x1
        Conv2d          3x3 2 15^2x16
        depthSepConv    3x3 1 15^2x32
        depthSepConv    3x3 2 8^2x64
        depthSepConv*2  5x5 1 8^2x128
        depthSepConv    5x5 1 8^2x256(SE)
        depthSepConv    5x5 1 8^2x256(SE)
        GAP             7x7 1 1^2x256
        Conv2d, NBN     1x1 1 1^2x512
        
        '''
        self.input_shape = input_shape
        self.model = self.build()

    def build(self):
        x = Input(shape=self.input_shape)

        y = SteamConv()(x)
    
        y = DepthSepCov(32, (3, 3), (1, 1))(y)
        y = DepthSepCov(64, (3, 3), (2, 2))(y)
    
        y = DepthSepCov(128, (5, 5), (1, 1))(y)
        y = DepthSepCov(128, (5, 5), (1, 1))(y)
    
        y = DepthSepCov(256, (5, 5), (1, 1))(y)
        y = DepthSepCov(256, (5, 5), (1, 1))(y)

        y = GlobalAveragePooling2D()(y)
        
        return Model(inputs=x, outputs=y, name='CNN')


if __name__ == "__main__":
    x = np.random.random((1, 48, 192, 1))
    x = tf.convert_to_tensor(x)
    model = CNN(input_shape=(48, 192, 1))
