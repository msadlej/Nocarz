from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

from nocarz.config import INPUT_COLUMNS, NUMERICAL_TARGETS, CATEGORICAL_TARGETS


class AdvancedModel:
    """
    Advanced model for predicting listing attributes.

    Attributes:
        vectorizer (TfidfVectorizer): Vectorizer for text features.
        regressor (MultiOutputRegressor): Regressor for numerical targets.
        classifier (MultiOutputClassifier): Classifier for categorical targets.
        label_encoders (dict): Dictionary of label encoders for categorical targets.
        fitted (bool): Indicates if the model has been fitted.
    """

    def __init__(self, max_features: int = 1000, n_estimators: int = 100) -> None:
        self._max_features = max_features
        self._n_estimators = n_estimators

        self._fitted = False

    @property
    def max_features(self) -> int:
        return self._max_features

    @property
    def n_estimators(self) -> int:
        return self._n_estimators

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the training data.

        Args:
            X (pd.DataFrame): The input features for the listings, containing text features.
            y (pd.DataFrame): The target values for the listings, containing numerical and categorical columns.
        """

        X_clean = X.copy()
        for col in X_clean.columns:
            X_clean[col] = X_clean[col].fillna("")

        self._vectorizer = TfidfVectorizer(max_features=self._max_features)
        X_vec = self._vectorizer.fit_transform(X_clean.agg(" ".join, axis=1))

        y_reg = y[NUMERICAL_TARGETS]
        y_cls = pd.DataFrame()

        self._label_encoders: dict = {}
        for col in CATEGORICAL_TARGETS:
            le = LabelEncoder()
            y_cls[col] = le.fit_transform(y[col].astype(str))
            self._label_encoders[col] = le

        self._regressor = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=self._n_estimators, random_state=42)
        )
        self._classifier = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=self._n_estimators, random_state=42)
        )

        self._regressor.fit(X_vec, y_reg)
        self._classifier.fit(X_vec, y_cls)
        self._fitted = True

    def predict(self, X: pd.Series) -> dict:
        """
        Make predictions for a single listing.

        Args:
            X (pd.Series): The input data for the listing, containing text features.

        Returns:
            dict: Predicted values for numerical and categorical columns.
        """

        if not self.fitted:
            raise RuntimeError("Model has not been trained! Call fit() first.")

        text_input = []
        for col in INPUT_COLUMNS:
            value = X.get(col, "")
            text_input.append(str(value))
        X_vec = self._vectorizer.transform([" ".join(text_input)])

        y_reg_pred = self._regressor.predict(X_vec)[0]
        y_cls_pred = self._classifier.predict(X_vec)[0]

        result = {}
        for i, col in enumerate(NUMERICAL_TARGETS):
            result[col] = float(y_reg_pred[i])

        for i, col in enumerate(CATEGORICAL_TARGETS):
            result[col] = self._label_encoders[col].inverse_transform([y_cls_pred[i]])[
                0
            ]

        return result

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "regressor": self._regressor,
                    "classifier": self._classifier,
                    "label_encoders": self._label_encoders,
                    "fitted": self._fitted,
                },
                f,
            )

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            self._vectorizer = state["vectorizer"]
            self._regressor = state["regressor"]
            self._classifier = state["classifier"]
            self._label_encoders = state["label_encoders"]
            self._fitted = state["fitted"]

    @staticmethod
    def evaluate_predictions(predictions: dict, true_values: dict) -> dict:
        results = {}

        for col in NUMERICAL_TARGETS + CATEGORICAL_TARGETS:
            if col in true_values and col in predictions:
                pred = predictions[col]
                true = true_values[col]

                if pred is not None and pd.notna(true):
                    if col in CATEGORICAL_TARGETS:
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
