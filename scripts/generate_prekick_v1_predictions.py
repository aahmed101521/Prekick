from datetime import datetime, timezone

import pandas as pd

from prekick.elo import (
    fit_draw_parameter,
    three_way_probabilities,
    update_match,
)
from prekick.goal_model import (
    build_team_index,
    expected_goals,
    fit_goal_model,
    get_team_parameters,
)
from prekick.poisson import (
    match_outcome_probabilities,
)


INITIAL_ELO_RATING = 1500
ELO_HOME_ADVANTAGE = 100
ELO_K_FACTOR = 20

ELO_WEIGHT = 0.50
POISSON_WEIGHT = 0.50

MODEL_VERSION = "prekick_v1"


# ============================================================
# 1. LOAD HISTORICAL DATA
# ============================================================

historical_data = pd.read_csv(
    "data/processed/model_data.csv",
    parse_dates=["Date"],
)

historical_data = historical_data.sort_values(
    ["Date", "match_id"]
).reset_index(drop=True)

print(
    "Historical matches:",
    len(historical_data),
)

print(
    "First historical match:",
    historical_data["Date"].min(),
)

print(
    "Training cutoff:",
    historical_data["Date"].max(),
)


# ============================================================
# 2. RECONSTRUCT CURRENT ELO STATE
# ============================================================

elo_ratings = {}

historical_home_elos = []
historical_away_elos = []
historical_results = []

for match in historical_data.itertuples():
    home_rating = elo_ratings.get(
        match.HomeTeam,
        INITIAL_ELO_RATING,
    )

    away_rating = elo_ratings.get(
        match.AwayTeam,
        INITIAL_ELO_RATING,
    )

    # Store the ratings as they stood BEFORE this match.
    # These are required for fitting the Elo draw parameter.
    historical_home_elos.append(
        home_rating
    )

    historical_away_elos.append(
        away_rating
    )

    historical_results.append(
        match.FTR
    )

    new_home_rating, new_away_rating = update_match(
        home_rating,
        away_rating,
        match.FTR,
        home_advantage=ELO_HOME_ADVANTAGE,
        k_factor=ELO_K_FACTOR,
    )

    elo_ratings[match.HomeTeam] = (
        new_home_rating
    )

    elo_ratings[match.AwayTeam] = (
        new_away_rating
    )


# Fit the draw parameter using the complete historical
# training period and the corresponding pre-match ratings.
elo_draw_parameter = fit_draw_parameter(
    historical_home_elos,
    historical_away_elos,
    historical_results,
    home_advantage=ELO_HOME_ADVANTAGE,
)


# Elo updates conserve the total number of rating points.
expected_rating_total = (
    len(elo_ratings)
    * INITIAL_ELO_RATING
)

actual_rating_total = sum(
    elo_ratings.values()
)

if abs(
    actual_rating_total
    - expected_rating_total
) > 1e-9:
    raise ValueError(
        "Elo rating points were not conserved."
    )


print()
print(
    "Teams with Elo ratings:",
    len(elo_ratings),
)

print(
    "Final Elo rating total:",
    actual_rating_total,
)

print(
    "Fitted Elo draw parameter:",
    elo_draw_parameter,
)
