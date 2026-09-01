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

from prekick.scoring import score_prediction

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


# ============================================================
# 5. WALK-FORWARD MULTINOMIAL BACKTEST
# ============================================================

prediction_blocks = []

for prediction_date in prediction_dates:
    print(
        "Fitting backtest date:",
        prediction_date.date(),
    )

    # Strictly historical training data.
    train_data = data[
        data["Date"] < prediction_date
    ].copy()

    # Predict every match on the current date together.
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
# 6. COMBINE BACKTEST PREDICTIONS
# ============================================================

backtest_predictions = pd.concat(
    prediction_blocks,
    ignore_index=True,
)


# ============================================================
# 7. VALIDATE BACKTEST PREDICTIONS
# ============================================================

if len(backtest_predictions) != 1520:
    raise ValueError(
        "Backtest predictions do not contain the expected 1520 matches."
    )

if backtest_predictions[
    "match_id"
].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in multinomial backtest."
    )

probability_sum = (
    backtest_predictions["multinomial_home_prob"]
    + backtest_predictions["multinomial_draw_prob"]
    + backtest_predictions["multinomial_away_prob"]
)

if (
    probability_sum - 1.0
).abs().gt(1e-12).any():
    raise ValueError(
        "Some multinomial probabilities do not sum to 1."
    )

print()
print(
    "Completed backtest predictions:",
    len(backtest_predictions),
)

print(
    backtest_predictions[
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


# ============================================================
# 8. SCORE HELD-OUT BACKTEST
# ============================================================

backtest_scores = pd.DataFrame(
    [
        score_prediction(
            row.multinomial_home_prob,
            row.multinomial_draw_prob,
            row.multinomial_away_prob,
            row.FTR,
        )
        for row in backtest_predictions.itertuples()
    ]
)

print()
print(
    "Multinomial held-out RPS:",
    backtest_scores["rps"].mean(),
)

print(
    "Multinomial held-out log loss:",
    backtest_scores["log_loss"].mean(),
)

print(
    "Multinomial held-out Brier:",
    backtest_scores["brier"].mean(),
)
