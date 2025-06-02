from typing import Any
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder

NUMERICAL_COLUMNS = ["accommodates", "bathrooms", "bedrooms", "beds", "price"]
CATEGORICAL_COLUMNS = ["property_type", "room_type", "bathrooms_text"]


class AdvancedModel:
    def __init__(self) -> None:
        self.regressor = None
        self.classifier = None
        self.le_dict = {}
        self.features = NUMERICAL_COLUMNS.copy()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        df = df.dropna(subset=self.features + CATEGORICAL_COLUMNS)

        X = df[self.features]

        y_reg = df[NUMERICAL_COLUMNS]
        y_cls = pd.DataFrame()
        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            y_cls[col] = le.fit_transform(df[col])
            self.le_dict[col] = le

        self.regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        self.classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

        self.regressor.fit(X, y_reg)
        self.classifier.fit(X, y_cls)
        self._fitted = True

    def predict(self, listing: pd.Series) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        X = listing[self.features].to_frame().T

        y_reg_pred = self.regressor.predict(X)[0]
        y_cls_pred = self.classifier.predict(X)[0]

        result = {}
        for i, col in enumerate(NUMERICAL_COLUMNS):
            result[col] = round(y_reg_pred[i], 2) if pd.notna(y_reg_pred[i]) else None

        for i, col in enumerate(CATEGORICAL_COLUMNS):
            result[col] = self.le_dict[col].inverse_transform([y_cls_pred[i]])[0] if col in self.le_dict else None

        return result

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "regressor": self.regressor,
                    "classifier": self.classifier,
                    "le_dict": self.le_dict,
                    "features": self.features,
                    "fitted": self._fitted,
                },
                f,
            )

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            self.regressor = state["regressor"]
            self.classifier = state["classifier"]
            self.le_dict = state["le_dict"]
            self.features = state["features"]
            self._fitted = state["fitted"]

    @staticmethod
    def evaluate_predictions(predictions: dict, true_values: dict) -> dict:
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
