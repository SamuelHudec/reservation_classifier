import json
import logging
import os.path
import pickle

import numpy as np
import xgboost as xgb
from sklearn import metrics

from config.xgb import LEARNING_RATE, MAX_DEPTH, METRIC_NAME, N_ESTIMATORS
from cancelation_classifier.config.data import RANDOM_SEED

logger = logging.getLogger("train_and_check_model")


class ClassifierXGB:
    """
    TODO: read model hyperparamers by dict parsed using dataclass
    """

    def __init__(
            self, X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, model_path: str
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_path = model_path
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=RANDOM_SEED,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
        )

    @staticmethod
    def _get_dumped_model_stats(best_model_path: str) -> float:
        with open(os.path.join(best_model_path, "metadata.json"), "rb") as f:
            loaded_metadata = json.load(f)
        if loaded_metadata["metric"] == METRIC_NAME:
            return loaded_metadata[METRIC_NAME]
        else:
            logger.error("New and best model metrics doesent match. Create a new directory.")

    def _dump_model_and_metadata(self, new_metric: float) -> None:
        model_metadata = dict()
        model_metadata["parameters"] = self.model.get_params()
        model_metadata[METRIC_NAME] = new_metric
        model_metadata["metric"] = METRIC_NAME
        with open(os.path.join(self.model_path, "metadata.json"), "w") as f:
            json.dump(model_metadata, f)
        with open(os.path.join(self.model_path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"New Model and metadata dumped to {self.model_path}.")

    def train(self) -> int:
        # have to split into preprocess and model becase knn have no predict or other conventions to use in pipe
        logger.info("Training started")
        self.model.fit(self.X_train, self.y_train)
        logger.info(f"Trained model parameters {self.model.get_params()}")

        logger.info("***Model evaluation.***")
        y_pred = self.model.predict(self.X_test)
        metric_function = getattr(metrics, METRIC_NAME)
        new_metric = metric_function(self.y_test, y_pred)
        logger.info(f"Test data {METRIC_NAME}: {new_metric}")

        self._dump_model_and_metadata(new_metric=new_metric)

        desired_path = os.path.dirname(self.model_path)
        best_model_path = os.path.join(desired_path, "best")
        if os.path.exists(best_model_path):
            best_model_metric = self._get_dumped_model_stats(best_model_path=best_model_path)
            improvement = ((new_metric - best_model_metric) / best_model_metric) * 100
            logger.info(f"New model improvement: {improvement}.")
        else:
            improvement = 1

        logger.info(f"Training script DONE")
        return improvement
