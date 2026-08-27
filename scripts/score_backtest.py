import pandas as pd

from prekick.ensemble import blend_probabilities
from prekick.scoring import score_prediction


# ============================================================
# 1. LOAD BACKTEST PREDICTIONS
# ============================================================

backtest = pd.read_csv(
    "predictions/backtest_predictions.csv"
)

poisson = pd.read_csv(
    "predictions/poisson_backtest_predictions.csv"
)

dixon_coles = pd.read_csv(
    "predictions/dixon_coles_backtest_predictions.csv"
)


# ============================================================
# 2. MERGE POISSON PREDICTIONS
# ============================================================

poisson_columns = [
    "match_id",
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_home_prob",
    "poisson_draw_prob",
    "poisson_away_prob",
]

backtest = backtest.merge(
    poisson[poisson_columns],
    on="match_id",
    how="left",
    validate="one_to_one",
)

if len(backtest) != 1520:
    raise ValueError(
        "Merged backtest does not contain the expected 1520 matches."
    )

if backtest[
    [
        "poisson_home_prob",
        "poisson_draw_prob",
        "poisson_away_prob",
    ]
].isna().any().any():
    raise ValueError(
        "Some matches are missing Poisson predictions."
    )


# ============================================================
# 3. MERGE DIXON-COLES PREDICTIONS
# ============================================================

dixon_coles_columns = [
    "match_id",
    "dixon_coles_home_xg",
    "dixon_coles_away_xg",
    "dixon_coles_rho",
    "dixon_coles_home_prob",
    "dixon_coles_draw_prob",
    "dixon_coles_away_prob",
]

backtest = backtest.merge(
    dixon_coles[dixon_coles_columns],
    on="match_id",
    how="left",
    validate="one_to_one",
)

if len(backtest) != 1520:
    raise ValueError(
        "Merged backtest does not contain the expected 1520 matches."
    )

if backtest[
    [
        "dixon_coles_home_prob",
        "dixon_coles_draw_prob",
        "dixon_coles_away_prob",
    ]
].isna().any().any():
    raise ValueError(
        "Some matches are missing Dixon-Coles predictions."
    )


# ============================================================
# 4. CREATE 50/50 ELO + POISSON ENSEMBLE
# ============================================================

ensemble_probabilities = backtest.apply(
    lambda row: blend_probabilities(
        first_probabilities=(
            row["elo_home_prob"],
            row["elo_draw_prob"],
            row["elo_away_prob"],
        ),
        second_probabilities=(
            row["poisson_home_prob"],
            row["poisson_draw_prob"],
            row["poisson_away_prob"],
        ),
        first_weight=0.5,
    ),
    axis=1,
)

(
    backtest["ensemble_home_prob"],
    backtest["ensemble_draw_prob"],
    backtest["ensemble_away_prob"],
) = zip(*ensemble_probabilities)


# ============================================================
# 5. DEFINE SHARED ROW-SCORING FUNCTION
# ============================================================

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


# ============================================================
# 6. SCORE BASE-RATE PREDICTIONS
# ============================================================

base_rate_scores = backtest.apply(
    lambda row: score_row(
        row,
        "base_rate_home_prob",
        "base_rate_draw_prob",
        "base_rate_away_prob",
    ),
    axis=1,
)


# ============================================================
# 7. SCORE ELO PREDICTIONS
# ============================================================

elo_scores = backtest.apply(
    lambda row: score_row(
        row,
        "elo_home_prob",
        "elo_draw_prob",
        "elo_away_prob",
    ),
    axis=1,
)


# ============================================================
# 8. SCORE POISSON PREDICTIONS
# ============================================================

poisson_scores = backtest.apply(
    lambda row: score_row(
        row,
        "poisson_home_prob",
        "poisson_draw_prob",
        "poisson_away_prob",
    ),
    axis=1,
)


# ============================================================
# 9. SCORE DIXON-COLES PREDICTIONS
# ============================================================

dixon_coles_scores = backtest.apply(
    lambda row: score_row(
        row,
        "dixon_coles_home_prob",
        "dixon_coles_draw_prob",
        "dixon_coles_away_prob",
    ),
    axis=1,
)


# ============================================================
# 10. SCORE 50/50 ELO + POISSON ENSEMBLE
# ============================================================

ensemble_scores = backtest.apply(
    lambda row: score_row(
        row,
        "ensemble_home_prob",
        "ensemble_draw_prob",
        "ensemble_away_prob",
    ),
    axis=1,
)


# ============================================================
# 11. SCORE MARKET PREDICTIONS
# ============================================================

market_scores = backtest.apply(
    lambda row: score_row(
        row,
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
    ),
    axis=1,
)


# ============================================================
# 12. ADD BASE-RATE SCORES
# ============================================================

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


# ============================================================
# 13. ADD ELO SCORES
# ============================================================

backtest["elo_rps"] = [
    scores["rps"]
    for scores in elo_scores
]

backtest["elo_log_loss"] = [
    scores["log_loss"]
    for scores in elo_scores
]

backtest["elo_brier"] = [
    scores["brier"]
    for scores in elo_scores
]


# ============================================================
# 14. ADD POISSON SCORES
# ============================================================

backtest["poisson_rps"] = [
    scores["rps"]
    for scores in poisson_scores
]

backtest["poisson_log_loss"] = [
    scores["log_loss"]
    for scores in poisson_scores
]

backtest["poisson_brier"] = [
    scores["brier"]
    for scores in poisson_scores
]


# ============================================================
# 15. ADD DIXON-COLES SCORES
# ============================================================

backtest["dixon_coles_rps"] = [
    scores["rps"]
    for scores in dixon_coles_scores
]

backtest["dixon_coles_log_loss"] = [
    scores["log_loss"]
    for scores in dixon_coles_scores
]

backtest["dixon_coles_brier"] = [
    scores["brier"]
    for scores in dixon_coles_scores
]


# ============================================================
# 16. ADD ENSEMBLE SCORES
# ============================================================

backtest["ensemble_rps"] = [
    scores["rps"]
    for scores in ensemble_scores
]

backtest["ensemble_log_loss"] = [
    scores["log_loss"]
    for scores in ensemble_scores
]

backtest["ensemble_brier"] = [
    scores["brier"]
    for scores in ensemble_scores
]


# ============================================================
# 17. ADD MARKET SCORES
# ============================================================

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


# ============================================================
# 18. SUMMARY
# ============================================================

print(
    "Scored matches:",
    len(backtest),
)

summary_columns = [
    "base_rate_rps",
    "base_rate_log_loss",
    "base_rate_brier",
    "elo_rps",
    "elo_log_loss",
    "elo_brier",
    "poisson_rps",
    "poisson_log_loss",
    "poisson_brier",
    "dixon_coles_rps",
    "dixon_coles_log_loss",
    "dixon_coles_brier",
    "ensemble_rps",
    "ensemble_log_loss",
    "ensemble_brier",
    "market_rps",
    "market_log_loss",
    "market_brier",
]

print(
    backtest[
        summary_columns
    ].mean()
)


# ============================================================
# 19. VALIDATE SCORED BACKTEST
# ============================================================

if backtest[
    summary_columns
].isna().any().any():
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


# ============================================================
# 20. VALIDATE ENSEMBLE PROBABILITIES
# ============================================================

ensemble_probability_columns = [
    "ensemble_home_prob",
    "ensemble_draw_prob",
    "ensemble_away_prob",
]

if backtest[
    ensemble_probability_columns
].isna().any().any():
    raise ValueError(
        "Some ensemble probabilities are missing."
    )

if not backtest[
    ensemble_probability_columns
].apply(
    lambda column: column.between(0, 1).all()
).all():
    raise ValueError(
        "Some ensemble probabilities are outside 0 to 1."
    )

ensemble_probability_sum = (
    backtest["ensemble_home_prob"]
    + backtest["ensemble_draw_prob"]
    + backtest["ensemble_away_prob"]
)

if (
    ensemble_probability_sum - 1
).abs().gt(1e-12).any():
    raise ValueError(
        "Some ensemble probabilities do not sum to 1."
    )


# ============================================================
# 21. SAVE SCORED BACKTEST
# ============================================================

backtest.to_csv(
    "predictions/backtest_scored.csv",
    index=False,
)

print(
    "Saved scored backtest:",
    backtest.shape,
)
