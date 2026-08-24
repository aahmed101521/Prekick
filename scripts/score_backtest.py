import pandas as pd

from prekick.scoring import score_prediction


backtest = pd.read_csv(
    "predictions/backtest_predictions.csv"
)


def score_row(
    row,
    home_column,
    draw_column,
    away_column,
):
    return score_prediction(
        row[home_column],
        row[draw_column],
        row[away_column],
        row["FTR"],
    )


base_rate_scores = backtest.apply(
    lambda row: score_row(
        row,
        "base_rate_home_prob",
        "base_rate_draw_prob",
        "base_rate_away_prob",
    ),
    axis=1,
)

market_scores = backtest.apply(
    lambda row: score_row(
        row,
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
    ),
    axis=1,
)


backtest["base_rate_rps"] = [
    scores["rps"]
    for scores in base_rate_scores
]

backtest["base_rate_log_loss"] = [
    scores["log_loss"]
    for scores in base_rate_scores
]

backtest["base_rate_brier"] = [
    scores["brier"]
    for scores in base_rate_scores
]


backtest["market_rps"] = [
    scores["rps"]
    for scores in market_scores
]

backtest["market_log_loss"] = [
    scores["log_loss"]
    for scores in market_scores
]

backtest["market_brier"] = [
    scores["brier"]
    for scores in market_scores
]


print("Scored matches:", len(backtest))

print(
    backtest[
        [
            "base_rate_rps",
            "base_rate_log_loss",
            "base_rate_brier",
            "market_rps",
            "market_log_loss",
            "market_brier",
        ]
    ].mean()
)

# ============================================================
# SAVE SCORED BACKTEST
# ============================================================

score_columns = [
    "base_rate_rps",
    "base_rate_log_loss",
    "base_rate_brier",
    "market_rps",
    "market_log_loss",
    "market_brier",
]

if backtest[score_columns].isna().any().any():
    raise ValueError(
        "Some backtest scores are missing."
    )

if len(backtest) != 1520:
    raise ValueError(
        "Scored backtest does not contain the expected 1520 matches."
    )

if backtest["match_id"].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in scored backtest."
    )

backtest.to_csv(
    "predictions/backtest_scored.csv",
    index=False,
)

print(
    "Saved scored backtest:",
    backtest.shape,
)
