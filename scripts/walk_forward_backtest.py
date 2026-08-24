import pandas as pd


# ============================================================
# 1. LOAD PROCESSED MATCH DATA
# ============================================================

data = pd.read_csv(
    "data/processed/model_data.csv",
    parse_dates=["Date"],
)


# ============================================================
# 2. DEFINE BACKTEST PERIOD
# ============================================================

backtest_start = pd.Timestamp("2022-08-05")

backtest_data = data[
    data["Date"] >= backtest_start
]

print("Backtest matches:", len(backtest_data))


# ============================================================
# 3. CREATE PREDICTION-DATE BLOCKS
# ============================================================

prediction_dates = (
    backtest_data["Date"]
    .drop_duplicates()
    .sort_values()
)

print("Prediction-date blocks:", len(prediction_dates))


# ============================================================
# 4. WALK-FORWARD BASE-RATE PREDICTIONS
# ============================================================

prediction_blocks = []

for prediction_date in prediction_dates:
    train_data = data[
        data["Date"] < prediction_date
    ]

    predict_data = data[
        data["Date"] == prediction_date
    ]

    home_rate = (
        (train_data["FTR"] == "H").mean()
    )

    draw_rate = (
        (train_data["FTR"] == "D").mean()
    )

    away_rate = (
        (train_data["FTR"] == "A").mean()
    )

    predict_data = predict_data.copy()

    predict_data["base_rate_home_prob"] = home_rate
    predict_data["base_rate_draw_prob"] = draw_rate
    predict_data["base_rate_away_prob"] = away_rate

    prediction_blocks.append(predict_data)


# ============================================================
# 5. COMBINE ALL WALK-FORWARD PREDICTIONS
# ============================================================

backtest_predictions = pd.concat(
    prediction_blocks,
    ignore_index=True,
)


# ============================================================
# 6. CREATE CLEAN BACKTEST OUTPUT
# ============================================================

output_columns = [
    "match_id",
    "Date",
    "season",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "base_rate_home_prob",
    "base_rate_draw_prob",
    "base_rate_away_prob",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "market_odds_source",
    "market_overround",
]

backtest_output = backtest_predictions[
    output_columns
].copy()


# ============================================================
# 7. VALIDATE MARKET PROBABILITIES
# ============================================================

market_probability_columns = [
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
]

if backtest_output[
    market_probability_columns
].isna().any().any():
    raise ValueError(
        "Some backtest matches have missing market probabilities."
    )

market_prob_sum = (
    backtest_output["market_home_prob"]
    + backtest_output["market_draw_prob"]
    + backtest_output["market_away_prob"]
)

if (market_prob_sum - 1).abs().gt(1e-12).any():
    raise ValueError(
        "Some market probabilities do not sum to 1."
    )


# ============================================================
# 8. VALIDATE BACKTEST ROWS
# ============================================================

if len(backtest_output) != 1520:
    raise ValueError(
        "Backtest does not contain the expected 1520 matches."
    )

if backtest_output["match_id"].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in backtest predictions."
    )


# ============================================================
# 9. VALIDATE BASE-RATE PROBABILITIES
# ============================================================

base_rate_prob_sum = (
    backtest_output["base_rate_home_prob"]
    + backtest_output["base_rate_draw_prob"]
    + backtest_output["base_rate_away_prob"]
)

if (base_rate_prob_sum - 1).abs().gt(1e-12).any():
    raise ValueError(
        "Some base-rate probabilities do not sum to 1."
    )

print(
    "Market source counts:",
    backtest_output[
        "market_odds_source"
    ].value_counts(),
)

# ============================================================
# 10. SAVE BACKTEST PREDICTIONS
# ============================================================

backtest_output.to_csv(
    "predictions/backtest_predictions.csv",
    index=False,
)

print(
    "Saved backtest predictions:",
    backtest_output.shape,
)