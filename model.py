"""Model creation, training and prediction."""

import os
import time
from typing import Any, List
import keras
import idx2numpy
import numpy as np
from keras import Input
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
import cv2
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


def train_model(model, k):
    """
    Train the model using the dataset.

    Args:
        model: The Keras Sequential model to be trained.

    Returns:
        None
    """
    t_start = time.time()  # Record the start time of the training process.

    # Define the path to the dataset files.
    path = os.getcwd() + "/data/"

    # Load the training and testing datasets using idx2numpy.
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
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

    # Print the shapes of the datasets and the number of classes.
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, len(data_labels))

    # Use only 1/10th of the dataset to speed up local training.
    x_train = x_train[: x_train.shape[0] // k]
    y_train = y_train[: y_train.shape[0] // k]
    x_test = x_test[: x_test.shape[0] // k]
    y_test = y_test[: y_test.shape[0] // k]

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
        patience=3,  # Number of epochs with no improvement before reducing the learning rate.
        verbose=1,  # Print messages when the learning rate is reduced.
        factor=0.5,  # Reduce the learning rate by a factor of 0.5.
        min_lr=0.00001,  # Minimum learning rate.
    )

    # Train the model using the training dataset and validate on the testing dataset.
    model.fit(
        x_train,
        x_train_cat,
        validation_data=(x_test, y_test_cat),  # Validation dataset.
        callbacks=[
            learning_rate_reduction
        ],  # Apply the learning rate reduction callback.
        batch_size=64,  # Number of samples per gradient update.
        epochs=30,  # Number of epochs to train the model.
    )

    model.save("model.h5")
    print("Training done, dT:", time.time() - t_start)


def predict(model, img):
    """
    Predict the character represented by an image using the model.

    Args:
        model: The trained model.
        img: The input image as a NumPy array.

    Returns:
        The predicted character as a string.
    """
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)

    return chr(data_labels[result[0]])


def extract_letters(image_file: str, out_size=28) -> List[Any]:
    """Extract individual letter images from an input image.

    Args:
        image_file (str): Path to the input image file.
        out_size (int, optional): Size to which each letter image is resized. Defaults to 28.

    Returns:
        List[Any]: A list of extracted letter images.
    """
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    countours, hierarchy = cv2.findContours(
        img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    output = img.copy()

    letters = []
    for idx, countor in enumerate(countours):
        (x, y, w, h) = cv2.boundingRect(countor)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y : y + h, x : x + w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                # Enlarge image top-bottom
                # ------
                # ======
                # ------
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos : y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos : x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append(
                (
                    x,
                    w,
                    cv2.resize(
                        letter_square,
                        (out_size, out_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                )
            )

    letters.sort(key=lambda x: x[0], reverse=False)

    cv2.imshow("Text", output)

    if len(letters):
        cv2.waitKey(0)
        return letters
    else:
        print("Unable to recognize the text.")
        exit(1)


def img_to_str(model: Any, image_file: str) -> str:
    """Convert an image of letters to a string using the trained model.

    Args:
        model (Any): The trained Keras model for letter recognition.
        image_file (str): Path to the input image file.

    Returns:
        str: The recognized text from the image.
    """
    letters = extract_letters(image_file)
    s_out = ""

    def calculate_spacing(idx: int) -> int:
        """Calculate the spacing between letters."""
        if idx < len(letters) - 1:
            return letters[idx + 1][0] - letters[idx][0] - letters[idx][1]
        return 0

    for idx, letter in enumerate(letters):
        dn = calculate_spacing(idx)
        s_out += predict(model, letter[2])

        if dn > letter[1] / 4:
            s_out += " "

    return s_out
