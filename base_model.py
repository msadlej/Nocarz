import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv("listings/listings.csv")

df = df[
    [
        "host_id",
        "description",
        "property_type",
        "room_type",
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms",
        "price",
        "amenities",
    ]
].dropna()

df["price"] = df["price"].replace("[\\$,]", "", regex=True).astype(float)

le_property = LabelEncoder()
le_room = LabelEncoder()
df["property_type_enc"] = le_property.fit_transform(df["property_type"])
df["room_type_enc"] = le_room.fit_transform(df["room_type"])

X = df["description"]
y_reg = df[["accommodates", "bedrooms", "beds", "bathrooms", "price"]]
y_cls = df[["property_type_enc", "room_type_enc"]]
host_ids = df["host_id"]

X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, host_train, host_test = train_test_split(
    X, y_reg, y_cls, host_ids, test_size=0.2, random_state=42
)

reg_model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
        ("regressor", MultiOutputRegressor(LinearRegression())),
    ]
)
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

cls_model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
        ("classifier", MultiOutputClassifier(LogisticRegression(max_iter=1000))),
    ]
)
cls_model.fit(X_train, y_cls_train)
y_cls_pred = cls_model.predict(X_test)

reg_columns = y_reg.columns.tolist()
df_reg = pd.DataFrame(y_reg_pred, columns=reg_columns)

for col in reg_columns:
    df_reg[col] = df_reg[col].clip(lower=0)

for col in ["accommodates", "bedrooms", "beds", "bathrooms"]:
    df_reg[col] = df_reg[col].round().astype(int)
    df_reg[col] = df_reg[col].clip(lower=1)

df_reg["price"] = df_reg["price"].round(2)

df_cls = pd.DataFrame(
    {
        "property_type": le_property.inverse_transform(y_cls_pred[:, 0]),
        "room_type": le_room.inverse_transform(y_cls_pred[:, 1]),
    }
)

df_final_model = pd.concat([df_cls.reset_index(drop=True), df_reg.reset_index(drop=True)], axis=1)

most_common_cat = df.groupby("host_id")[["property_type", "room_type"]].agg(lambda x: x.mode()[0])
mean_num = df.groupby("host_id")[["accommodates", "bedrooms", "beds", "bathrooms", "price"]].mean()
host_baseline = most_common_cat.join(mean_num)

baseline_per_host = pd.DataFrame(index=X_test.index)
for col in host_baseline.columns:
    baseline_per_host[col] = host_test.map(
        lambda hid: host_baseline.loc[hid][col] if hid in host_baseline.index else np.nan
    )

for col in ["accommodates", "bedrooms", "beds", "bathrooms"]:
    baseline_per_host[col] = baseline_per_host[col].round().clip(lower=1).astype("Int64")

baseline_per_host["price"] = baseline_per_host["price"].round(2)
baseline_per_host["host_id"] = host_test.values

mask = (df_final_model["price"] <= 0) | (df_final_model["price"].isna())
df_final_model = df_final_model.reset_index(drop=True)
baseline_per_host = baseline_per_host.reset_index(drop=True)

mask = (df_final_model["price"] <= 0) | (df_final_model["price"].isna())
df_final_model.loc[mask, "price"] = baseline_per_host.loc[mask, "price"]
df_final_model["price"] = df_final_model["price"].round(2)

df_final_model["price"] = df_final_model["price"].round(2)

print("Model predictions:")
print(df_final_model.head())

print("\nBaseline predictions per host:")
print(baseline_per_host.reset_index(drop=True).head())
