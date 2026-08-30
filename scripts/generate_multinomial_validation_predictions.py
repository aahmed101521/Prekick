import pandas as pd

from prekick.preprocessing import (
    build_standardization_mask,
)


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(
    "data/processed/model_data.csv",
    parse_dates=["Date"],
)


# ============================================================
# 2. DEFINE MODEL FEATURES
# ============================================================

history_stats = [
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
]

lag_columns = []

for stat in history_stats:
    for lag in range(1, 3):
        lag_columns.append(
            f"{stat}_lag_{lag}"
        )

history_indicator_columns = [
    "has_lag_1_history",
    "has_lag_2_history",
]

feature_columns = (
    [f"home_{column}" for column in lag_columns]
    + [f"away_{column}" for column in lag_columns]
    + [
        f"home_{column}"
        for column in history_indicator_columns
    ]
    + [
        f"away_{column}"
        for column in history_indicator_columns
    ]
)

if len(feature_columns) != 28:
    raise ValueError(
        "Expected exactly 28 model features."
    )

missing_columns = [
    column
    for column in feature_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing model features: {missing_columns}"
    )

standardization_mask = (
    build_standardization_mask(
        feature_columns
    )
)

print(
    "Model features:",
    len(feature_columns),
)

print(
    "Features to standardize:",
    standardization_mask.sum(),
)

print(
    "Features left unchanged:",
    (~standardization_mask).sum(),
)


# ============================================================
# 3. DEFINE PRE-BACKTEST VALIDATION PERIOD
# ============================================================

validation_start = pd.Timestamp(
    "2022-03-01"
)

backtest_start = pd.Timestamp(
    "2022-08-05"
)

validation_data = data[
    (data["Date"] >= validation_start)
    & (data["Date"] < backtest_start)
].copy()

print(
    "Validation matches:",
    len(validation_data),
)

print(
    "First validation date:",
    validation_data["Date"].min(),
)

print(
    "Last validation date:",
    validation_data["Date"].max(),
)


# ============================================================
# 4. CREATE PREDICTION-DATE BLOCKS
# ============================================================

prediction_dates = (
    validation_data["Date"]
    .drop_duplicates()
    .sort_values()
)

print(
    "Prediction-date blocks:",
    len(prediction_dates),
)
