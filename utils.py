"""Util function definitions."""

import argparse
from typing import Any, List

import cv2
import numpy as np


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Text recognition script.")
    parser.add_argument(
        "--train",
        nargs="?",
        type=int,
        const=10,
        help="Train the model. Optional integer k specifies the dataset fraction: k means use 1/k of the dataset (e.g. --train 10 uses 1/10). If used without value, k defaults to 10.",
    )
    parser.add_argument(
        "--predict", type=str, help="Path to the image file for prediction."
    )

    return parser.parse_args(), parser


def calculate_spacing(idx: int, letters) -> int:
    """
    Calculate the spacing between letters based on their bounding box positions.

    Args:
        idx (int): The index of the current letter.
        letters (list): A list of letter bounding box data.

    Returns:
        int: The spacing between the current letter and the next letter.
    """
    if idx < len(letters) - 1:
        return letters[idx + 1][0] - letters[idx][0] - letters[idx][1]

    return 0


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

    # Apply binary thresholding to create a binary image (black and white).
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Perform erosion to remove noise and enhance the letter shapes.
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(
        img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    output = img.copy()

    letters = []

    for idx, contor in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contor)

        # Check if the contour is a top-level contour (not nested inside another contour).
        if hierarchy[0][idx][3] == 0:
            # Draw a rectangle around the detected letter on the output image.
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)

            # Extract the letter region from the grayscale image.
            letter_crop = gray[y : y + h, x : x + w]

            size_max = max(w, h)

            # Create a square canvas filled with white pixels to center the letter.
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)

            if w > h:
                # Enlarge the letter vertically to fit the square canvas.
                y_pos = size_max // 2 - h // 2
                letter_square[y_pos : y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge the letter horizontally to fit the square canvas.
                x_pos = size_max // 2 - w // 2
                letter_square[0:h, x_pos : x_pos + w] = letter_crop
            else:
                # If the letter is already square, use it as is.
                letter_square = letter_crop

            # Resize the letter to the specified output size (e.g., 28x28 pixels).
            letters.append(
                (
                    x,  # x-coordinate of the letter.
                    w,  # Width of the letter.
                    cv2.resize(
                        letter_square,
                        (out_size, out_size),
                        interpolation=cv2.INTER_AREA,
                    ),  # Resized letter image.
                )
            )

    # Sort the letters from left to right based on their x-coordinates.
    letters.sort(key=lambda x: x[0], reverse=False)

    # Display the output image with bounding boxes around the letters.
    cv2.imshow("Text", output)

    if letters:
        cv2.waitKey(0)
        return letters
    else:
        print("Unable to extract letters.")
        exit(1)
