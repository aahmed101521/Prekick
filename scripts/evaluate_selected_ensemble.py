import pandas as pd

from prekick.ensemble import blend_probabilities
from prekick.scoring import score_prediction


# ============================================================
# 1. LOAD HELD-OUT BACKTEST
# ============================================================

backtest = pd.read_csv(
    "predictions/backtest_scored.csv"
)

if len(backtest) != 1520:
    raise ValueError(
        "Backtest does not contain the expected 1520 matches."
    )


# ============================================================
# 2. FIX VALIDATION-SELECTED WEIGHT
# ============================================================

elo_weight = 0.25
poisson_weight = 0.75

print("Elo weight:", elo_weight)
print("Poisson weight:", poisson_weight)


# ============================================================
# 3. CREATE SELECTED ENSEMBLE PROBABILITIES
# ============================================================

selected_probabilities = backtest.apply(
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
        first_weight=elo_weight,
    ),
    axis=1,
)

(
    backtest["selected_home_prob"],
    backtest["selected_draw_prob"],
    backtest["selected_away_prob"],
) = zip(*selected_probabilities)


# ============================================================
# 4. VALIDATE PROBABILITIES
# ============================================================

probability_columns = [
    "selected_home_prob",
    "selected_draw_prob",
    "selected_away_prob",
]

if backtest[
    probability_columns
].isna().any().any():
    raise ValueError(
        "Some selected ensemble probabilities are missing."
    )

if not backtest[
    probability_columns
].apply(
    lambda column: column.between(0, 1).all()
).all():
    raise ValueError(
        "Some selected ensemble probabilities are outside 0 to 1."
    )

probability_sum = (
    backtest["selected_home_prob"]
    + backtest["selected_draw_prob"]
    + backtest["selected_away_prob"]
)

if (
    probability_sum - 1
).abs().gt(1e-12).any():
    raise ValueError(
        "Some selected ensemble probabilities do not sum to 1."
    )


# ============================================================
# 5. SCORE SELECTED ENSEMBLE
# ============================================================

scores = backtest.apply(
    lambda row: score_prediction(
        row["selected_home_prob"],
        row["selected_draw_prob"],
        row["selected_away_prob"],
        row["FTR"],
    ),
    axis=1,
)

selected_rps = sum(
    score["rps"]
    for score in scores
) / len(scores)

selected_log_loss = sum(
    score["log_loss"]
    for score in scores
) / len(scores)

selected_brier = sum(
    score["brier"]
    for score in scores
) / len(scores)


# ============================================================
# 6. PRINT HELD-OUT COMPARISON
# ============================================================

print()
print("Held-out backtest results:")
print()

print(
    "Selected 25/75 ensemble RPS:",
    selected_rps,
)
print(
    "Selected 25/75 ensemble log loss:",
    selected_log_loss,
)
print(
    "Selected 25/75 ensemble Brier:",
    selected_brier,
)

print()
print(
    "Existing 50/50 ensemble RPS:",
    backtest["ensemble_rps"].mean(),
)
print(
    "Existing 50/50 ensemble log loss:",
    backtest["ensemble_log_loss"].mean(),
)
print(
    "Existing 50/50 ensemble Brier:",
    backtest["ensemble_brier"].mean(),
)

print()
print(
    "Elo RPS:",
    backtest["elo_rps"].mean(),
)
print(
    "Elo log loss:",
    backtest["elo_log_loss"].mean(),
)
print(
    "Elo Brier:",
    backtest["elo_brier"].mean(),
)

print()
print(
    "Poisson RPS:",
    backtest["poisson_rps"].mean(),
)
print(
    "Poisson log loss:",
    backtest["poisson_log_loss"].mean(),
)
print(
    "Poisson Brier:",
    backtest["poisson_brier"].mean(),
)

print()
print(
    "Market RPS:",
    backtest["market_rps"].mean(),
)
print(
    "Market log loss:",
    backtest["market_log_loss"].mean(),
)
print(
    "Market Brier:",
    backtest["market_brier"].mean(),
)
