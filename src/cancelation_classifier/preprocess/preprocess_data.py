import logging
import os
import pickle
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cancelation_classifier.config.data import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_IMPUTER_STRATEGY,
    NUMERIC_FEATURES,
    NUMERIC_IMPUTER_STRATEGY,
    RANDOM_SEED,
    RESPONSE_COLUMN,
    TEST_SIZE,
    TRAINING_COLUMNS,
)
from utils.data import data_fetcher

# Filter out FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger("preprocess_training_data")


class GetParseData:
    """
    TODO: write ABC base class with fixed commons, in and outs
    TODO: extreme values
    """

    def __init__(self, path_to_data: str, model_path: str) -> None:
        self.data = data_fetcher(path=path_to_data)
        self.model_path = model_path
        self._check_and_log_stats()
        self.encoder = self._data_encoder()

    def _check_and_log_stats(self):
        # for some stats are questionable if not rise an exception
        logger.info(f"Data shape: {self.data.shape}")
        duplicates = self.data.drop_duplicates().shape[0] - self.data.shape[0]
        if duplicates > 0:
            logger.info(f"Number of duplicates: {duplicates}")
            self.data = self.data.drop_duplicates()
        missed_columns = [i for i in (set(TRAINING_COLUMNS) - set(self.data.columns))]
        if len(missed_columns) > 0:
            raise ValueError(f"The following training columns are missing in dataset: {missed_columns}")
        logger.info(f"Total number of nans: {self.data.isna().sum()}")

    @staticmethod
    def _data_encoder() -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy=NUMERIC_IMPUTER_STRATEGY)), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=CATEGORICAL_IMPUTER_STRATEGY)),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ]
        )
        return preprocessor

    def _handle_stays_in(self):
        # TODO: parametrize
        self.data["stays_in_nights"] = self.data["stays_in_weekend_nights"] + self.data["stays_in_week_nights"]

    def _dump_encoder(self) -> None:
        trained_model_path = os.path.join(self.model_path, "encoder.pkl")
        with open(trained_model_path, "wb") as f:
            pickle.dump(self.encoder, f)
        logger.info(f"Preprocessor dumped to {self.model_path}.")

    def get_train_data(self) -> Tuple[np.array, np.array, np.array, np.array]:
        self._handle_stays_in()
        X = self.data.drop(columns=[RESPONSE_COLUMN])
        y = self.data[RESPONSE_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        X_train = self.encoder.fit_transform(X_train)
        X_test = self.encoder.transform(X_test)
        self._dump_encoder()
        logger.info(f"Exporting training data split by ratio {TEST_SIZE}.")
        return X_train, X_test, y_train, y_test


class GetPredictData:
    """
    TODO: unify GetPredictData with GetParseData or find an elegant way
    """

    def __init__(self, encoder: ColumnTransformer) -> None:
        self.encoder = encoder

    @staticmethod
    def _check_and_log_stats(data):
        # for some stats are questionable if not rise an exception
        missed_columns = [i for i in (set(TRAINING_COLUMNS) - set(data.columns))]
        if len(missed_columns) > 0:
            raise ValueError(f"The following training columns are missing in dataset: {missed_columns}")
        logger.info(f"Total number of nans: {data.isna().sum()}")

    @staticmethod
    def _handle_stays_in(data) -> pd.DataFrame:
        # TODO: parametrize
        data["stays_in_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
        return data

    def get_prediction_data(self, data: pd.DataFrame) -> np.array:
        self._check_and_log_stats(data)
        data = self._handle_stays_in(data)
        return self.encoder.transform(data)
