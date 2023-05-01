import larq as lq
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))
    # Normalize pixel values to be between -1 and 1
    train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
    return train_images, train_labels, test_images, test_labels


# All quantized layers except the first will use the same options
kwargs = dict(
    use_bias=False,
    input_quantizer="ste_sign",
    kernel_quantizer="ste_sign",
    kernel_constraint="weight_clip"
)


def build_model():
    model = tf.keras.models.Sequential()
    # In the first layer we only quantize the weights and not the input
    model.add(lq.layers.QuantConv2D(
        32, (3, 3),
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        use_bias=False,
        input_shape=(28, 28, 1))
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(64, (3, 3), **kwargs))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.BatchNormalization(scale=False))

    model.add(lq.layers.QuantConv2D(64, (3, 3), **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(tf.keras.layers.Flatten())

    model.add(lq.layers.QuantDense(64, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    # dropout
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(lq.layers.QuantDense(10, **kwargs))
    model.add(tf.keras.layers.BatchNormalization(scale=False))
    model.add(tf.keras.layers.Activation("softmax"))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train(model):
    model.summary()
    lq.models.summary(model)

    model.fit(
        train_images, train_labels,
        batch_size=64,
        epochs=10,
        validation_data=(test_images, test_labels),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir="./logs"),
            # save best model
            tf.keras.callbacks.ModelCheckpoint(
                filepath="mnist.h5",
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ],
    )

    model.load_weights("mnist.h5")

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print(f"Test accuracy {test_acc * 100:.2f} %")

    import larq as lq
    import larq_compute_engine as lce
    import tensorflow as tf

    m = model
    lq.models.summary(m)
    with open("mnist.tflite", "wb") as flatbuffer_file:
        flatbuffer_bytes = lce.convert_keras_model(m)
        flatbuffer_file.write(flatbuffer_bytes)


if __name__ == '__main__':
    model = build_model()
    model.summary()
    lq.models.summary(model)
    # train(model)
