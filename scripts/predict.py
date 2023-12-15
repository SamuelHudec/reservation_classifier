import argparse
import logging
import os.path
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from cancelation_classifier.preprocess.preprocess_data import GetPredictData

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger("Predict using Best model")
logging.basicConfig(level=logging.INFO)

# this is just sample how it can look like for one case one carousel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", help="Full path or table end point.", required=True)
    parser.add_argument("--output_path", help="Full path or table end point.", required=True)
    parser.add_argument("--model_path", help="Full model path.", required=True)
    parser.add_argument("--chunk_size", help="Full path", default=100)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    best_model_path = os.path.join(args.model_path, "best")
    if not os.path.exists(best_model_path):
        raise ValueError("Best model directory doesnt exists.")

    if not os.path.exists(args.path_to_data):
        raise ValueError("Data doesnt exists.")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        logger.info(f"Directory '{args.output_path}' created.")
    else:
        logger.info(f"Directory '{args.output_path}' already exists.")

    with open(os.path.join(best_model_path, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(best_model_path, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    logger.info(f"Encoder and model has been loaded.")

    # pandas is not a right solution
    csv_reader = pd.read_csv(args.path_to_data, chunksize=args.chunk_size)
    preprocessor = GetPredictData(encoder)

    # Iterate over chunks
    predicted_values = list()
    for chunk in csv_reader:
        data = preprocessor.get_prediction_data(chunk)
        y_hat = model.predict(data)
        predicted_values.append(y_hat)

    current_datetime = datetime.now()
    formatted_integer = str(int(current_datetime.strftime("%Y%m%d%H%M%S")))
    np.save(os.path.join(args.output_path, formatted_integer), np.concatenate(predicted_values))


if __name__ == "__main__":
    args = parse_args()
    main(args)
