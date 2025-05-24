import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

df = pd.read_csv("listings/listings.csv")

df = df[
    [
        "host_id",
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

y_reg = df[["accommodates", "bedrooms", "beds", "bathrooms", "price"]]
y_cls = df[["property_type_enc", "room_type_enc"]]
host_ids = df["host_id"]

_, _, y_reg_test, y_reg_test, y_cls_test, y_cls_test, host_train, host_test = train_test_split(
    df.index, y_reg, y_cls, host_ids, test_size=0.2, random_state=42
)

most_common_cat = df.groupby("host_id")[["property_type", "room_type"]].agg(lambda x: x.mode()[0])
mean_num = df.groupby("host_id")[["accommodates", "bedrooms", "beds", "bathrooms", "price"]].mean()
host_baseline = most_common_cat.join(mean_num)

baseline_per_host = pd.DataFrame(index=host_test.index)
for col in host_baseline.columns:
    baseline_per_host[col] = host_test.map(
        lambda hid: host_baseline.loc[hid][col] if hid in host_baseline.index else np.nan
    )

for col in ["accommodates", "bedrooms", "beds", "bathrooms"]:
    baseline_per_host[col] = baseline_per_host[col].round().clip(lower=1).astype("Int64")

baseline_per_host["price"] = baseline_per_host["price"].round(2)
baseline_per_host["host_id"] = host_test.values

print("Baseline predictions per host:")
print(baseline_per_host.reset_index(drop=True).head())

print("\nMAE - regresja (baseline vs test):")
for col in y_reg.columns:
    if col in baseline_per_host.columns:
        mae = mean_absolute_error(y_reg_test[col].values, baseline_per_host[col].values)
        print(f"{col}: {mae:.2f}")

print("\nAccuracy - klasyfikacja (baseline vs test):")
acc_property_base = accuracy_score(
    y_cls_test["property_type_enc"], le_property.transform(baseline_per_host["property_type"])
)
acc_room_base = accuracy_score(y_cls_test["room_type_enc"], le_room.transform(baseline_per_host["room_type"]))
print(f"property_type: {acc_property_base:.2%}")
print(f"room_type: {acc_room_base:.2%}")
