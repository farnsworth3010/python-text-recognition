from typing import Any

import numpy as np
from data_labels import data_labels
from utils import calculate_spacing, extract_letters


def recognize_text(model: Any, image_file: str):
    """Convert an image of letters to a string using the trained model.

    Args:
        model (Any): The trained Keras model for letter recognition.
        image_file (str): Path to the input image file.

    Returns:
        str: The recognized text from the image.
    """
    letters = extract_letters(image_file)
    s_out = ""

    for idx, letter in enumerate(letters):
        dn = calculate_spacing(idx, letters)
        s_out += predict_letter(model, letter[2])

        if dn > letter[1] / 4:
            s_out += " "

    return s_out


def predict_letter(model, img):
    """
    Predict the letter represented by an image using the model.

    Args:
        model: The trained model.
        img: The input image as a NumPy array.

    Returns:
        The predicted character as a string.
    """
    # Expand the dimensions of the image to match the model's input shape (batch size of 1).
    img_arr = np.expand_dims(img, axis=0)

    # Normalize the image pixel values to the range [0, 1] and invert the colors (1 - pixel_value).
    img_arr = 1 - img_arr / 255.0

    # Rotate the image 270 degrees clockwise (equivalent to 90 degrees counterclockwise).
    img_arr[0] = np.rot90(img_arr[0], 3)

    # Flip the image horizontally to correct its orientation.
    img_arr[0] = np.fliplr(img_arr[0])

    # Reshape the image to ensure it has the correct dimensions (1, 28, 28, 1).
    img_arr = img_arr.reshape((1, 28, 28, 1))

    # Use the model to predict the class probabilities for the input image.
    predict_res = model.predict([img_arr])

    # Find the class index with the highest probability.
    result = np.argmax(predict_res, axis=1)

    # Convert the class index to the corresponding character using the data_labels mapping.
    return chr(data_labels[result[0]])
