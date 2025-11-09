"""Training entrypoint."""

import os
import time
import idx2numpy
import keras
import numpy as np
from data_labels import data_labels
from config import model_save_name, epochs, batch_size, min_lr, factor, patience


def train_model(model, k):
    """
    Train the model using the dataset.

    Args:
        model: The Keras Sequential model to be trained.
        k: the part of the dataset (1/k)

    Returns:
        None
    """
    t_start = time.time()

    path = os.getcwd() + "/data/"

    x_train = idx2numpy.convert_from_file(
        path + "emnist-byclass-train-images-idx3-ubyte"
    )
    y_train = idx2numpy.convert_from_file(
        path + "emnist-byclass-train-labels-idx1-ubyte"
    )
    x_test = idx2numpy.convert_from_file(path + "emnist-byclass-test-images-idx3-ubyte")
    y_test = idx2numpy.convert_from_file(path + "emnist-byclass-test-labels-idx1-ubyte")

    # Reshape the dataset to have a shape of (num_samples, 28, 28, 1),
    # where 28x28 is the image size and 1 is the grayscale channel.
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))  # type: ignore
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))  # type: ignore

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, len(data_labels))

    # Use only 1/10th of the dataset to speed up local training.
    x_train = x_train[: x_train.shape[0] // k]
    y_train = y_train[: y_train.shape[0] // k]  # type: ignore
    x_test = x_test[: x_test.shape[0] // k]
    y_test = y_test[: y_test.shape[0] // k]  # type: ignore

    # Normalize the pixel values of the images to the range [0, 1].
    x_train = x_train.astype(np.float32)
    x_train /= 255.0
    x_test = x_test.astype(np.float32)
    x_test /= 255.0

    # Convert the labels to one-hot encoded format for multi-class classification.
    x_train_cat = keras.utils.to_categorical(y_train, len(data_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(data_labels))

    # Set up a learning rate reduction callback to reduce the learning rate
    # when the validation accuracy plateaus.
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",  # Monitor the validation accuracy.
        patience=patience,
        verbose=1,  # Print messages when the learning rate is reduced.
        factor=factor,
        min_lr=min_lr,
    )

    model.fit(
        x_train,
        x_train_cat,
        validation_data=(x_test, y_test_cat),  # Validation dataset.
        callbacks=[
            learning_rate_reduction
        ],  # Apply the learning rate reduction callback.
        batch_size=batch_size,  # Number of samples per gradient update.
        epochs=epochs,  # Number of epochs to train the model.
    )

    model.save(model_save_name)

    print("Training done, dT:", time.time() - t_start)
