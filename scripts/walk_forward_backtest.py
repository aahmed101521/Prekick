import pandas as pd

from prekick.elo import (
    fit_draw_parameter,
    three_way_probabilities,
)


# ============================================================
# 1. LOAD PROCESSED MATCH DATA
# ============================================================

data = pd.read_csv(
    "data/processed/model_data.csv",
    parse_dates=["Date"],
)

elo_history = pd.read_csv(
    "data/processed/elo_history.csv",
    parse_dates=["Date"],
)

elo_columns = [
    "match_id",
    "home_elo_before",
    "away_elo_before",
]

data = data.merge(
    elo_history[elo_columns],
    on="match_id",
    how="left",
    validate="one_to_one",
)

if data[
    [
        "home_elo_before",
        "away_elo_before",
    ]
].isna().any().any():
    raise ValueError(
        "Some matches are missing Elo ratings."
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
# 4. WALK-FORWARD BASE-RATE AND ELO PREDICTIONS
# ============================================================

prediction_blocks = []

for prediction_date in prediction_dates:
    train_data = data[
        data["Date"] < prediction_date
    ]

    predict_data = data[
        data["Date"] == prediction_date
    ].copy()

    # Base-rate probabilities
    home_rate = (
        (train_data["FTR"] == "H").mean()
    )

    draw_rate = (
        (train_data["FTR"] == "D").mean()
    )

    away_rate = (
        (train_data["FTR"] == "A").mean()
    )

    predict_data["base_rate_home_prob"] = home_rate
    predict_data["base_rate_draw_prob"] = draw_rate
    predict_data["base_rate_away_prob"] = away_rate

    # Fit the Elo draw parameter using past matches only
    draw_parameter = fit_draw_parameter(
        train_data["home_elo_before"],
        train_data["away_elo_before"],
        train_data["FTR"],
    )

    predict_data["elo_draw_parameter"] = (
        draw_parameter
    )

    elo_probabilities = [
        three_way_probabilities(
            row.home_elo_before,
            row.away_elo_before,
            draw_parameter,
        )
        for row in predict_data.itertuples()
    ]

    (
        predict_data["elo_home_prob"],
        predict_data["elo_draw_prob"],
        predict_data["elo_away_prob"],
    ) = zip(*elo_probabilities)

    prediction_blocks.append(
        predict_data
    )

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
    "elo_draw_parameter",
    "elo_home_prob",
    "elo_draw_prob",
    "elo_away_prob",
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
# 10. VALIDATE ELO PROBABILITIES
# ============================================================

elo_probability_columns = [
    "elo_home_prob",
    "elo_draw_prob",
    "elo_away_prob",
]

if backtest_output[
    elo_probability_columns
].isna().any().any():
    raise ValueError(
        "Some backtest matches have missing Elo probabilities."
    )

if not backtest_output[
    elo_probability_columns
].apply(
    lambda column: column.between(0, 1).all()
).all():
    raise ValueError(
        "Some Elo probabilities are outside 0 to 1."
    )

elo_prob_sum = (
    backtest_output["elo_home_prob"]
    + backtest_output["elo_draw_prob"]
    + backtest_output["elo_away_prob"]
)

if (elo_prob_sum - 1).abs().gt(1e-12).any():
    raise ValueError(
        "Some Elo probabilities do not sum to 1."
    )

if (
    backtest_output["elo_draw_parameter"]
    <= 0
).any():
    raise ValueError(
        "Some Elo draw parameters are not positive."
    )


# ============================================================
# 11. SAVE BACKTEST PREDICTIONS
# ============================================================

backtest_output.to_csv(
    "predictions/backtest_predictions.csv",
    index=False,
)

print(
    "Saved backtest predictions:",
    backtest_output.shape,
)