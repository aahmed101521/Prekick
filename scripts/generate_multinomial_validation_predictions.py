import numpy as np
import pandas as pd

from prekick.multinomial import (
    fit_multinomial_model,
    predict_probabilities,
)
from prekick.preprocessing import (
    build_standardization_mask,
    fit_standardization_parameters,
    transform_features,
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


# ============================================================
# 5. WALK-FORWARD MULTINOMIAL VALIDATION
# ============================================================

penalty_strength = 1.0

prediction_blocks = []

for prediction_date in prediction_dates:
    print(
        "Fitting validation date:",
        prediction_date.date(),
    )

    # Strictly historical training data.
    train_data = data[
        data["Date"] < prediction_date
    ].copy()

    # Predict all matches on the current date together.
    predict_data = data[
        data["Date"] == prediction_date
    ].copy()

    # --------------------------------------------------------
    # LEARN PREPROCESSING FROM TRAINING DATA ONLY
    # --------------------------------------------------------

    train_features_raw = train_data[
        feature_columns
    ].to_numpy(
        dtype=float,
    )

    predict_features_raw = predict_data[
        feature_columns
    ].to_numpy(
        dtype=float,
    )

    (
        means,
        standard_deviations,
    ) = fit_standardization_parameters(
        train_features_raw,
        standardization_mask,
    )

    train_features = transform_features(
        train_features_raw,
        means,
        standard_deviations,
    )

    predict_features = transform_features(
        predict_features_raw,
        means,
        standard_deviations,
    )

    if not np.isfinite(
        train_features
    ).all():
        raise ValueError(
            "Training features contain non-finite values."
        )

    if not np.isfinite(
        predict_features
    ).all():
        raise ValueError(
            "Prediction features contain non-finite values."
        )

    # --------------------------------------------------------
    # FIT MULTINOMIAL MODEL
    # --------------------------------------------------------

    outcomes = train_data[
        "FTR"
    ].tolist()

    (
        coefficients,
        intercepts,
    ) = fit_multinomial_model(
        train_features,
        outcomes,
        penalty_strength=penalty_strength,
    )

    # --------------------------------------------------------
    # PREDICT CURRENT DATE
    # --------------------------------------------------------

    multinomial_probabilities = [
        predict_probabilities(
            match_features,
            coefficients,
            intercepts,
        )
        for match_features in predict_features
    ]

    (
        predict_data["multinomial_home_prob"],
        predict_data["multinomial_draw_prob"],
        predict_data["multinomial_away_prob"],
    ) = zip(*multinomial_probabilities)

    prediction_blocks.append(
        predict_data
    )


# ============================================================
# 6. COMBINE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.concat(
    prediction_blocks,
    ignore_index=True,
)


# ============================================================
# 7. VALIDATE RESULTS
# ============================================================

if len(validation_predictions) != 124:
    raise ValueError(
        "Validation predictions do not contain the expected 124 matches."
    )

if validation_predictions[
    "match_id"
].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in validation predictions."
    )

probability_sum = (
    validation_predictions["multinomial_home_prob"]
    + validation_predictions["multinomial_draw_prob"]
    + validation_predictions["multinomial_away_prob"]
)

if (
    probability_sum - 1.0
).abs().gt(1e-12).any():
    raise ValueError(
        "Some multinomial probabilities do not sum to 1."
    )

print(
    "Completed validation predictions:",
    len(validation_predictions),
)

print(
    validation_predictions[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "multinomial_home_prob",
            "multinomial_draw_prob",
            "multinomial_away_prob",
        ]
    ].head()
)
