import argparse
import logging
import os
import shutil
import warnings
from datetime import datetime

from cancelation_classifier.model.xgb import ClassifierXGB
from cancelation_classifier.preprocess.preprocess_data import GetParseData

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger("Train New Model")
logging.basicConfig(level=logging.INFO)

# this is just sample how it can look like for one case one carousel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", help="Full path or table end point. (e.g. data/train)", required=True)
    parser.add_argument("--output_model_path", help="Full output model path. (e.g. data/model)", required=True)
    parser.add_argument(
        "--improvement_threshold",
        help="Improvment threshold in %.",
        default=0.1,
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    if not os.path.exists(args.output_model_path):
        os.makedirs(args.output_model_path)
        logger.info(f"Directory '{args.output_model_path}' created.")
    else:
        logger.info(f"Directory '{args.output_model_path}' already exists.")

    current_datetime = datetime.now()
    formatted_integer = str(int(current_datetime.strftime("%Y%m%d%H%M%S")))
    current_model_path = os.path.join(args.output_model_path, formatted_integer)
    os.makedirs(current_model_path)
    logger.info(f"New model directory '{current_model_path}' created.")

    # add date in front of all model
    preprocessor = GetParseData(path_to_data=args.path_to_data, model_path=current_model_path)
    X_train, X_test, y_train, y_test = preprocessor.get_train_data()

    model = ClassifierXGB(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model_path=current_model_path)
    how_is_better = model.train()

    if how_is_better > args.improvement_threshold:
        shutil.rmtree(os.path.join(args.output_model_path, "best"), ignore_errors=True)
        shutil.copytree(current_model_path, os.path.join(args.output_model_path, "best"))
        logger.info("The best model has been replaced by a new model.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
