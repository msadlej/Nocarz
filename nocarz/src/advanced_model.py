# Importy i konfiguracja
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pickle

# Definicje kolumn
INPUT_COLUMNS = ["name", "description", "neighbourhood"]
NUMERICAL_TARGETS = ["accommodates", "bathrooms", "bedrooms", "beds", "price"]
CATEGORICAL_TARGETS = ["property_type", "room_type", "bathrooms_text"]


# Klasa AdvancedModel
class AdvancedModel:
    def __init__(self) -> None:
        self.regressor = None
        self.classifier = None
        self.le_dict = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        df = df.dropna(subset=INPUT_COLUMNS + NUMERICAL_TARGETS + CATEGORICAL_TARGETS)

        X = df[INPUT_COLUMNS].fillna("")

        y_reg = df[NUMERICAL_TARGETS]
        y_cls = pd.DataFrame()
        for col in CATEGORICAL_TARGETS:
            le = LabelEncoder()
            y_cls[col] = le.fit_transform(df[col])
            self.le_dict[col] = le

        text_transformer = ColumnTransformer(
            [
                ("name", TfidfVectorizer(max_features=100), "name"),
                ("desc", TfidfVectorizer(max_features=200), "description"),
                ("neigh", TfidfVectorizer(max_features=50), "neighbourhood"),
            ]
        )

        X_vec = text_transformer.fit_transform(X)

        self.text_transformer = text_transformer
        self.regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
        self.classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))

        self.regressor.fit(X_vec, y_reg)
        self.classifier.fit(X_vec, y_cls)
        self._fitted = True

    def predict(self, listing: pd.Series) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        X = listing[INPUT_COLUMNS].fillna("").to_frame().T
        X_vec = self.text_transformer.transform(X)

        y_reg_pred = self.regressor.predict(X_vec)[0]
        y_cls_pred = self.classifier.predict(X_vec)[0]

        result = {}
        for i, col in enumerate(NUMERICAL_TARGETS):
            result[col] = round(y_reg_pred[i], 2) if pd.notna(y_reg_pred[i]) else None

        for i, col in enumerate(CATEGORICAL_TARGETS):
            result[col] = self.le_dict[col].inverse_transform([y_cls_pred[i]])[0] if col in self.le_dict else None

        return result

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "regressor": self.regressor,
                    "classifier": self.classifier,
                    "le_dict": self.le_dict,
                    "text_transformer": self.text_transformer,
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
            self.text_transformer = state["text_transformer"]
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
                        results[col] = {"predicted": pred, "actual": true, "match": match, "type": "categorical"}
                    else:
                        error = abs(float(pred) - float(true))
                        results[col] = {"predicted": pred, "actual": true, "error": error, "type": "numerical"}

        return results
