from itertools import groupby

import numpy as np
import tensorflow as tf
from keras.layers import Layer


def best_path(mat: np.ndarray, chars: str) -> str:
    # get char indices along best path
    best_path_indices = np.argmax(mat, axis=1)

    # collapse best path (using itertools.groupby), map to chars, join char list to string
    blank_idx = len(chars)
    best_chars_collapsed = [chars[k] for k, _ in groupby(best_path_indices) if k != blank_idx]
    res = ''.join(best_chars_collapsed)
    return res


# CTC Decoder using Greedy Search, in keras operations
class CTCDecoder(Layer):
    def __init__(self, name="CTCDecoder", num_classes=85, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def call(self, mat, chars):
        best_path_indices = tf.argmax(mat, axis=2, output_type=tf.int32)

        # collapse best path (using tf.unique), map to chars, join char list to string
        blank_idx = len(chars)
        best_path_unique, _ = tf.unique(tf.squeeze(best_path_indices))
        best_chars_collapsed = tf.gather(tf.constant(list(chars)), best_path_unique)
        best_chars_collapsed = tf.boolean_mask(best_chars_collapsed, tf.not_equal(best_path_unique, blank_idx))
        res = tf.strings.reduce_join(best_chars_collapsed)
        return res.numpy().decode('utf-8')


if __name__ == "__main__":
    mat = np.array([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]])
    chars = 'ab'

    print(f'Best path: "{best_path(mat, chars)}"')
    
    # ctc_decoder = CTCDecoder(num_classes=len(chars) + 1)
    # # convert to tensor
    # mat = tf.convert_to_tensor([mat])
    # # print shape
    # print(f'Input shape: {mat.shape}')
    # # call decoder
    # print(f'Best path: "{ctc_decoder(mat, chars)}"')
