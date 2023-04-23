import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


class CTCDecoder(tf.keras.layers.Layer):
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
        return res


# Define input tensor
input_tensor = tf.constant(np.array([[[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]]]), dtype=tf.float32)
# Define character set
char_set = 'ab'

# Create decoder
ctc_decoder = CTCDecoder(num_classes=len(char_set) + 1)
decoded_text = ctc_decoder(input_tensor, char_set)

# Print decoded text
print(decoded_text)
