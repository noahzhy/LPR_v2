# TCN for minst data
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from keras.datasets import mnist
input_data = tf.keras.datasets.mnist
from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense


# Load data 载入数据
def read_data(path):
    # mnist = input_data.read_data_sets(path, one_hot=True)
    mnist = input_data.load_data(path)
    train_x, train_y = mnist.train.images.reshape(-1, 28, 28), mnist.train.labels,
    valid_x, valid_y = mnist.validation.images.reshape(-1, 28, 28), mnist.validation.labels,
    test_x, test_y = mnist.test.images.reshape(-1, 28, 28), mnist.test.labels
    return train_x, train_y, valid_x, valid_y, test_x, test_y


# Residual block 残差块
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(
        x)  # first convolution
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # Second convolution

    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  	# shortcut (shortcut)
    o = add([r, shortcut])
    # Activation function
    o = Activation('relu')(o)  
    return o


# Sequence Model 时序模型
def TCN(classes=85, epoch=20):
    inputs = Input(shape=(28, 28))
    x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=16, kernel_size=3, dilation_rate=4)
    # x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    # View network structure 查看网络结构
    model.summary()
    # Compile model 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Training model 训练模型
    model.fit(train_x, train_y, batch_size=500, nb_epoch=epoch, verbose=2, validation_data=(valid_x, valid_y))
    # Assessment model 评估模型
    pre = model.evaluate(test_x, test_y, batch_size=500, verbose=2)
    print('test_loss:', pre[0], '- test_acc:', pre[1])

# # MINST数字从0-9共10个，即10个类别
classes = 85
epoch = 30
# train_x, train_y, valid_x, valid_y, test_x, test_y = read_data('MNIST_data')
# #print(train_x, train_y)

TCN(classes, epoch)
