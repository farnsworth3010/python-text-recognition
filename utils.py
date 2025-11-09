"""Util function definitions."""

import argparse


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
