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

standardization_mask = (
    build_standardization_mask(
        feature_columns
    )
)


# ============================================================
# 3. FREEZE MODEL SETTINGS
# ============================================================

penalty_strength = 10.0

print(
    "Selected penalty strength:",
    penalty_strength,
)


# ============================================================
# 4. DEFINE HELD-OUT BACKTEST PERIOD
# ============================================================

backtest_start = pd.Timestamp(
    "2022-08-05"
)

backtest_data = data[
    data["Date"] >= backtest_start
].copy()

print(
    "Backtest matches:",
    len(backtest_data),
)

prediction_dates = (
    backtest_data["Date"]
    .drop_duplicates()
    .sort_values()
)

print(
    "Prediction-date blocks:",
    len(prediction_dates),
)

print(
    "First backtest date:",
    backtest_data["Date"].min(),
)

print(
    "Last backtest date:",
    backtest_data["Date"].max(),
)
