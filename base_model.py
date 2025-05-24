import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

# Load and preprocess the data
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

# Encode categorical features for evaluation purposes
le_property = LabelEncoder()
le_room = LabelEncoder()
df["property_type_enc"] = le_property.fit_transform(df["property_type"])
df["room_type_enc"] = le_room.fit_transform(df["room_type"])

# Split data by host_id (input) - train data is historical listings, test data is new listings to predict
X = df["host_id"]  # Changed from description to host_id
y_reg = df[["accommodates", "bedrooms", "beds", "bathrooms", "price"]]
y_cls = df[["property_type_enc", "room_type_enc"]]

# Notice: we now stratify by host_id to ensure each host has listings in both train and test
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=None
)

# Create a DataFrame for training data (historical listings)
train_df = pd.DataFrame(
    {
        "host_id": X_train.values,
        "property_type": df.loc[X_train.index, "property_type"].values,
        "room_type": df.loc[X_train.index, "room_type"].values,
        "accommodates": y_reg_train["accommodates"].values,
        "bedrooms": y_reg_train["bedrooms"].values,
        "beds": y_reg_train["beds"].values,
        "bathrooms": y_reg_train["bathrooms"].values,
        "price": y_reg_train["price"].values,
    }
)

# Calculate most common categorical values from TRAINING data only
most_common_cat = train_df.groupby("host_id")[["property_type", "room_type"]].agg(lambda x: x.mode()[0])

# Calculate mean numerical values from TRAINING data only
mean_num = train_df.groupby("host_id")[["accommodates", "bedrooms", "beds", "bathrooms", "price"]].mean()

# Join them to get the baseline per host
host_baseline = most_common_cat.join(mean_num)

# Calculate global defaults for hosts without history
global_defaults = {
    "property_type": train_df["property_type"].mode()[0],
    "room_type": train_df["room_type"].mode()[0],
    "accommodates": train_df["accommodates"].mean(),
    "bedrooms": train_df["bedrooms"].mean(),
    "beds": train_df["beds"].mean(),
    "bathrooms": train_df["bathrooms"].mean(),
    "price": train_df["price"].mean(),
}

# Generate autocomplete predictions for test data (new listings)
baseline_per_host = pd.DataFrame(index=X_test.index)
for col in host_baseline.columns:
    baseline_per_host[col] = X_test.map(
        lambda hid: host_baseline.loc[hid][col] if hid in host_baseline.index else global_defaults[col]
    )

# Round and clean up numerical predictions
for col in ["accommodates", "bedrooms", "beds"]:
    baseline_per_host[col] = baseline_per_host[col].round().clip(lower=1).astype("Int64")

# Special handling for bathrooms as they can be 0.5, 1.5, etc.
baseline_per_host["bathrooms"] = (baseline_per_host["bathrooms"] * 2).round() / 2
baseline_per_host["bathrooms"] = baseline_per_host["bathrooms"].clip(lower=0.5)

baseline_per_host["price"] = baseline_per_host["price"].round(2)
baseline_per_host["host_id"] = X_test.values

# For evaluation, create a test data frame with actual values
test_df = pd.DataFrame(
    {
        "host_id": X_test.values,
        "property_type": df.loc[X_test.index, "property_type"].values,
        "room_type": df.loc[X_test.index, "room_type"].values,
        "accommodates": y_reg_test["accommodates"].values,
        "bedrooms": y_reg_test["bedrooms"].values,
        "beds": y_reg_test["beds"].values,
        "bathrooms": y_reg_test["bathrooms"].values,
        "price": y_reg_test["price"].values,
        "property_type_enc": y_cls_test["property_type_enc"].values,
        "room_type_enc": y_cls_test["room_type_enc"].values,
    }
)

# Print statistics about host coverage
hosts_with_history = sum(1 for host_id in X_test.unique() if host_id in host_baseline.index)
total_hosts = len(X_test.unique())
print(f"Hosts with historical data: {hosts_with_history}/{total_hosts} ({hosts_with_history/total_hosts:.1%})")

print("\nBaseline autocomplete predictions (first 5 entries):")
print(baseline_per_host.reset_index(drop=True).head())

print("\nMAE – Regression (autocomplete vs actual values):")
for col in y_reg.columns:
    if col in baseline_per_host.columns:
        mae = mean_absolute_error(test_df[col].values, baseline_per_host[col].values)
        print(f"{col}: {mae:.2f}")

print("\nAccuracy – Classification (autocomplete vs actual values):")
acc_property_base = accuracy_score(
    test_df["property_type_enc"], le_property.transform(baseline_per_host["property_type"])
)
acc_room_base = accuracy_score(
    test_df["room_type_enc"], le_room.transform(baseline_per_host["room_type"])
)
print(f"property_type: {acc_property_base:.2%}")
print(f"room_type: {acc_room_base:.2%}")

print("\nBaseline autocomplete predictions vs. actual values (first 5 entries):")
comparison = pd.DataFrame()
comparison["host_id"] = baseline_per_host["host_id"].reset_index(drop=True).head()
for col in host_baseline.columns:
    if col != "host_id":
        comparison[f"pred_{col}"] = baseline_per_host[col].reset_index(drop=True).head()
        comparison[f"actual_{col}"] = test_df[col].reset_index(drop=True).head()
print(comparison)

print(baseline_per_host.reset_index(drop=True).head())
