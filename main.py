"""Text recognition entrypoint."""

import keras
from model import get_model
from predict import recognize_text
from train import train_model
from utils import parse_args
from config import model_save_name


def main():
    """Main function to handle training and prediction based on arguments."""

    args, parser = parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.train:
        train_model(get_model(), args.train)

    if args.predict:
        model = keras.models.load_model(model_save_name)
        s_out = recognize_text(model, args.predict)
        print(s_out)


if __name__ == "__main__":
    main()
