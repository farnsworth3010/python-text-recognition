"""Model creation, training and prediction."""

from keras import Input
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from data_labels import data_labels


def get_model():
    """
    Create and compile the model.

    Returns:
        A compiled Keras Sequential model for classification.
    """
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    # 28 x 28, grayscale (1 channel)

    # First convolutional layer: Applies 32 filters of size 3x3 with ReLU activation.
    # This layer extracts features from the input image.
    model.add(
        Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            padding="valid",  # No padding; the output size will be reduced.
            activation="relu",
        )
    )

    # Second convolutional layer: Applies 64 filters of size 3x3 with ReLU activation.
    # This layer extracts more complex features from the image.
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation="relu"))

    # MaxPooling layer: Reduces the spatial dimensions of the feature maps by taking the maximum value in a 2x2 window.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout layer: Randomly sets 25% of the weights to zero during training to prevent overfitting.
    model.add(Dropout(0.25))

    # Flatten layer: Converts the 2D feature maps into a 1D vector for the fully connected layers.
    model.add(Flatten())

    # Fully connected (dense) layer: Applies 512 neurons with ReLU activation.
    # This layer learns high-level representations of the data.
    model.add(Dense(512, activation="relu"))

    # Dropout layer: Randomly sets 50% of the weights to zero during training to further prevent overfitting.
    model.add(Dropout(0.5))

    # Output layer: Applies a softmax activation to produce probabilities for each class.
    # The number of neurons corresponds to the number of classes (len(data_labels)).
    model.add(Dense(len(data_labels), activation="softmax"))

    # Compile the model: Specifies the loss function, optimizer, and evaluation metrics.
    model.compile(
        loss="categorical_crossentropy",  # Loss function for multi-class classification.
        optimizer="adadelta",  # Optimizer for adjusting weights during training.
        metrics=["accuracy"],  # Metric to evaluate the model's performance.
    )

    return model
