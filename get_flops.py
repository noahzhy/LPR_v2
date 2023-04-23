import os
import sys
from pathlib import Path

import tensorflow as tf
from keras.layers import *
import keras.backend as K

from cnn import *
from model import *


import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the Keras model from the HDF5 file
model_h5_path = "TCN.h5"
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils import layer_utils
from tensorflow.keras.utils import custom_object_scope

# Load the Keras model
model = load_model(model_h5_path)

# Get the total number of parameters in the model
total_params = layer_utils.count_params(model.trainable_weights)

# Calculate the FLOPs based on the number of MAC operations
flops = (
    # Total number of MAC operations in all layers
    sum([K.count_params(p) * (p.input_shape[1:].num_elements() or 1) for p in model.trainable_weights]) +
    # Total number of bias additions in all layers
    sum([K.count_params(p) for p in model.trainable_weights if p.shape[-1] != 1])
) * 2 # Multiply by 2 because each MAC operation involves 2 floating-point multiplications

print(f'Total number of FLOPs: {flops:.2f}')



# if __name__ == "__main__":
#     flops = get_flops(model_h5_path)
#     print(f"FLOPS: {flops}")
