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
    

class PP_LCNet(Layer):
    def __init__(self, feature):
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
        super(PP_LCNet, self).__init__()
        self.steam = SteamConv()
        # 3x3
        self.dep1 = DepthSepCov(32, (3, 3), (1, 1))
        self.dep2 = DepthSepCov(64, (3, 3), (2, 2))
        
        # 5x5
        self.dep3 = DepthSepCov(128, (5, 5), (1, 1))
        self.dep4 = DepthSepCov(128, (5, 5), (1, 1))
        
        self.dep5 = DepthSepCov(256, (5, 5), (1, 1))
        self.dep6 = DepthSepCov(256, (5, 5), (1, 1))
        
        self.gap = GlobalAveragePooling2D()
        self.conv1 = Conv2D(512, kernel_size=(1, 1), activation='relu', padding='same')
        self.fl = Flatten()
        self.fc1 = Dense(feature, activation='softmax')
        
    def call(self, x):
        y = self.steam(x)
        
        y = self.dep2(self.dep1(y))
        y = self.dep4(self.dep3(y))
        
        y = self.dep6(self.dep5(y))
        y = self.gap(y)
        print(y.shape)
        y = tf.expand_dims(tf.expand_dims(y, axis=1), axis=1)
        y = self.fl(self.conv1(y))
        y = self.fc1(y)
        
        return y


if __name__ == "__main__":
    x = np.random.random((1, 48, 192, 3))
    x = tf.convert_to_tensor(x)
    model = PP_LCNet(100)
    y = model(x)
    print(y.shape)
