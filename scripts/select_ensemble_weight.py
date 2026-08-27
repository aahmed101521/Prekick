import pandas as pd

from prekick.ensemble import blend_probabilities
from prekick.scoring import score_prediction


# ============================================================
# 1. LOAD PRE-BACKTEST VALIDATION PREDICTIONS
# ============================================================

validation = pd.read_csv(
    "predictions/ensemble_validation_predictions.csv"
)

if len(validation) != 124:
    raise ValueError(
        "Validation data does not contain the expected 124 matches."
    )


# ============================================================
# 2. DEFINE PRE-SPECIFIED ELO WEIGHTS
# ============================================================

elo_weights = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


# ============================================================
# 3. SCORE EACH WEIGHT
# ============================================================

results = []

for elo_weight in elo_weights:
    scores = []

    for row in validation.itertuples():
        (
            home_probability,
            draw_probability,
            away_probability,
        ) = blend_probabilities(
            first_probabilities=(
                row.elo_home_prob,
                row.elo_draw_prob,
                row.elo_away_prob,
            ),
            second_probabilities=(
                row.poisson_home_prob,
                row.poisson_draw_prob,
                row.poisson_away_prob,
            ),
            first_weight=elo_weight,
        )

        score = score_prediction(
            home_probability,
            draw_probability,
            away_probability,
            row.FTR,
        )

        scores.append(score)

    mean_rps = sum(
        score["rps"]
        for score in scores
    ) / len(scores)

    mean_log_loss = sum(
        score["log_loss"]
        for score in scores
    ) / len(scores)

    mean_brier = sum(
        score["brier"]
        for score in scores
    ) / len(scores)

    results.append(
        {
            "elo_weight": elo_weight,
            "poisson_weight": 1 - elo_weight,
            "rps": mean_rps,
            "log_loss": mean_log_loss,
            "brier": mean_brier,
        }
    )


# ============================================================
# 4. CREATE RESULTS TABLE
# ============================================================

results_table = pd.DataFrame(results)

results_table = results_table.sort_values(
    "elo_weight"
).reset_index(
    drop=True
)

print()
print("Validation weight results:")
print()

print(
    results_table.to_string(
        index=False
    )
)


# ============================================================
# 5. SELECT WEIGHT USING VALIDATION RPS
# ============================================================

best_row = results_table.loc[
    results_table["rps"].idxmin()
]

best_elo_weight = float(
    best_row["elo_weight"]
)

best_poisson_weight = float(
    best_row["poisson_weight"]
)

print()
print(
    "Selected Elo weight:",
    best_elo_weight,
)

print(
    "Selected Poisson weight:",
    best_poisson_weight,
)

print(
    "Selected validation RPS:",
    float(best_row["rps"]),
)

print(
    "Selected validation log loss:",
    float(best_row["log_loss"]),
)

print(
    "Selected validation Brier:",
    float(best_row["brier"]),
)
