import pandas as pd
from typing import Any
import pickle


NUMERICAL_COLUMNS = ["accommodates", "bathrooms", "bedrooms", "beds", "price"]
CATEGORICAL_COLUMNS = ["property_type", "room_type", "bathrooms_text"]


class BaseModel:
    """
    Base model for predicting listing attributes.

    Attributes:
        data (pd.DataFrame): DataFrame containing the historical listing data.
    """

    def __init__(self, data: pd.DataFrame = None) -> None:
        self._data = data if data is not None else pd.DataFrame()

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def add_listing(self, listing: pd.Series) -> None:
        """
        Add a new listing to the model's data.

        Args:
            listing (pd.Series): A Series containing the listing data to be added.
        """

        self._data = self._data.append(listing, ignore_index=True)

    def make_prediction(self, host_id: int) -> tuple[dict[str, Any], str]:
        """
        Make predictions for a single listing.

        Args:
            host_id (int): The ID of the host for whom to make predictions.

        Returns:
            tuple: A tuple containing:
                - predictions (dict): Predicted values for numerical and categorical columns.
                - data_type (str): Indicates if the predictions are based on user data or global data.
        """

        user_data = self._get_user_listings(host_id)
        data = user_data if len(user_data) > 0 else self._data
        data_type = "user" if len(user_data) > 0 else "global"

        predictions = {}
        for col in NUMERICAL_COLUMNS:
            if col in data.columns:
                predictions[col] = self._predict_numerical(data[col])
            else:
                predictions[col] = None

        for col in CATEGORICAL_COLUMNS:
            if col in data.columns:
                predictions[col] = self._predict_categorical(data[col])
            else:
                predictions[col] = None

        return predictions, data_type

    def save(self, filepath: str) -> None:
        """
        Save the trained model to a file using pickle.

        Args:
            filepath (str): Path where the model should be saved.
        """

        with open(filepath, 'wb') as f:
            pickle.dump(self._data, f)

    def load(self, filepath: str) -> None:
        """
        Load a trained model from a file using pickle.

        Args:
            filepath (str): Path to the saved model file.
        """

        with open(filepath, 'rb') as f:
            self._data = pickle.load(f)

    def _get_user_listings(self, host_id: int) -> pd.DataFrame:
        """
        Get previous listings for a specific user.

        Args:
            host_id (int): The ID of the host.

        Returns:
            pd.DataFrame: DataFrame containing listings for the specified host.
        """

        return self._data[self._data["host_id"] == host_id]

    @staticmethod
    def _predict_numerical(values: pd.Series) -> Any:
        """
        Predict numerical value using median.

        Args:
            values (pd.Series): Series of numerical values to predict from.

        Returns:
            float: Predicted median value.
        """

        values = values.dropna()
        if len(values) > 0:
            return values.median()
        return None

    @staticmethod
    def _predict_categorical(values: pd.Series) -> Any:
        """
        Predict categorical value using most common.

        Args:
            values (pd.Series): Series of categorical values to predict from.

        Returns:
            str: Most common value in the series.
        """

        values = values.dropna()
        if len(values) > 0:
            return values.mode().iloc[0] if len(values.mode()) > 0 else None
        return None

    @staticmethod
    def evaluate_predictions(predictions: dict, true_values: dict) -> dict:
        """
        Evaluate predictions against true values.

        Args:
            predictions (dict): Predicted values for numerical and categorical columns.
            true_values (dict): Actual values for the same columns.

        Returns:
            dict: Evaluation results containing errors for numerical columns and matches for categorical columns.
        """

        results = {}

        for col in NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS:
            if col in true_values and col in predictions:
                pred = predictions[col]
                true = true_values[col]

                if pred is not None and pd.notna(true):
                    if col in CATEGORICAL_COLUMNS:
                        match = str(pred) == str(true)
                        results[col] = {
                            "predicted": pred,
                            "actual": true,
                            "match": match,
                            "type": "categorical",
                        }
                    else:
                        error = abs(float(pred) - float(true))
                        results[col] = {
                            "predicted": pred,
                            "actual": true,
                            "error": error,
                            "type": "numerical",
                        }

        return results
