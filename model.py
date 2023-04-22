import numpy as np
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K

from keras_flops import get_flops

from ctc import CTCLayer



MAX_LABEL_LEN = 8


class TemporalConvNet:
    def __init__(self, seq_length, blocks=6, filters=64):
        self.seq_length = seq_length
        self.n_filters = filters

        self.model = self.build(blocks)

    def residual_block(self, factor):
        dilation = 2 ** factor
        inputs = Input(shape=(self.seq_length, 1))

        # Residual block
        c1 = Conv1D(self.n_filters, kernel_size=4, strides=1, padding='causal', dilation_rate=dilation)(inputs)
        at1 = Activation('tanh')(c1)
        as1 = Activation('sigmoid')(c1)
        a1 = multiply([at1, as1])
        n1 = BatchNormalization(momentum=0.6)(a1)

        c2 = Conv1D(self.n_filters, kernel_size=4, strides=1, padding='causal', dilation_rate=dilation)(n1)
        at2 = Activation('tanh')(c2)
        as2 = Activation('sigmoid')(c2)
        a2 = multiply([at2, as2])
        n2 = BatchNormalization(momentum=0.6)(a2)

        # Residual connection
        residual = Conv1D(1, kernel_size=1, padding='same')(n2)
        outputs = add([inputs, residual])

        return Model(inputs=inputs, outputs=outputs, name='residual_block_{}'.format(factor))

    def build(self, dilations, train=True):
        model = Sequential()

        for dilation in range(dilations):
            block = self.residual_block(dilation)
            model.add(block)

        return Model(inputs=model.input, outputs=model.output, name='TCN')


    def train(self, train_x, train_y, epochs, verbose, x_test):
        return self.model.fit(
            x=train_x,
            y=train_y,
            batch_size=self.seq_length,
            epochs=epochs,
            verbose=verbose
        )

    # def predict(self, data, n_ahead):
    #     predicted = []

    #     for _ in range(n_ahead):
    #         pred = self.model.predict(data)
    #         new = pred[:, -1, :]
    #         predicted.append(new)
    #         data = np.append(data, new).reshape(1, -1, 1)
    #         data = data[:, 1:, :]

    #     return np.array(predicted).flatten()

    def save_model(self, name):
        self.model.save('{}.h5'.format(name))
        # summary
        self.model.summary()
        print('Model has been saved as {}.h5'.format(name))


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = K.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :MAX_LABEL_LEN
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


if __name__ == "__main__":
    model = TemporalConvNet(128, 6, 64)


    model.save_model('tcn_lpr')

    # model.model.summary()

    # x = np.random.rand(1, 128, 1)
    # y_pred = model.model.predict(x)
    # # y = decode_batch_predictions(pred=model.model.predict(x))
    # print(y_pred)
    

    # flops = get_flops(model.model, batch_size=1)
    # # to Gflops
    # print(f"FLOPS: {flops / 1e9:.03} G")
