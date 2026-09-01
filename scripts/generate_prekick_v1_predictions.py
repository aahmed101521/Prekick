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
    "Historical training cutoff:",
    historical_data["Date"].max(),
)


# ============================================================
# 2. ADD COMPLETED CURRENT-SEASON MATCHES
# ============================================================

current_season_data = pd.read_csv(
    "data/fixtures/completed_matches_2026_27.csv",
    parse_dates=["date"],
)

if len(current_season_data) != 20:
    raise ValueError(
        "Expected exactly 20 completed 2026/27 matches."
    )

current_season_data = current_season_data.rename(
    columns={
        "date": "Date",
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "home_goals": "FTHG",
        "away_goals": "FTAG",
        "result": "FTR",
    }
)

if not current_season_data["FTR"].isin(
    ["H", "D", "A"]
).all():
    raise ValueError(
        "Current-season results must be H, D, or A."
    )

next_match_id = (
    int(historical_data["match_id"].max())
    + 1
)

current_season_data["match_id"] = range(
    next_match_id,
    next_match_id + len(current_season_data),
)

current_season_model_data = current_season_data[
    [
        "match_id",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
    ]
].copy()

live_training_data = pd.concat(
    [
        historical_data[
            [
                "match_id",
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
                "FTR",
            ]
        ],
        current_season_model_data,
    ],
    ignore_index=True,
)

live_training_data = live_training_data.sort_values(
    ["Date", "match_id"]
).reset_index(drop=True)

if len(live_training_data) != 1920:
    raise ValueError(
        "Expected exactly 1920 live training matches."
    )

if live_training_data["match_id"].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in live training data."
    )

print()
print(
    "Current-season completed matches:",
    len(current_season_data),
)

print(
    "Live training matches:",
    len(live_training_data),
)

print(
    "Live training cutoff:",
    live_training_data["Date"].max(),
)


# ============================================================
# 3. RECONSTRUCT CURRENT ELO STATE
# ============================================================

elo_ratings = {}

historical_home_elos = []
historical_away_elos = []
historical_results = []

for match in live_training_data.itertuples():
    home_rating = elo_ratings.get(
        match.HomeTeam,
        INITIAL_ELO_RATING,
    )

    away_rating = elo_ratings.get(
        match.AwayTeam,
        INITIAL_ELO_RATING,
    )

    # Store ratings as they stood BEFORE the match.
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


# Fit the draw parameter using every match available
# before the upcoming fixtures.
elo_draw_parameter = fit_draw_parameter(
    historical_home_elos,
    historical_away_elos,
    historical_results,
    home_advantage=ELO_HOME_ADVANTAGE,
)


# Elo updates conserve the total rating points.
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


# ============================================================
# 4. VERIFY HISTORICAL ELO RECONSTRUCTION
# ============================================================

elo_history = pd.read_csv(
    "data/processed/elo_history.csv",
)

reconstructed_elo_history = pd.DataFrame(
    {
        "match_id": historical_data["match_id"],
        "reconstructed_home_elo": (
            historical_home_elos[
                :len(historical_data)
            ]
        ),
        "reconstructed_away_elo": (
            historical_away_elos[
                :len(historical_data)
            ]
        ),
    }
)

elo_check = elo_history[
    [
        "match_id",
        "home_elo_before",
        "away_elo_before",
    ]
].merge(
    reconstructed_elo_history,
    on="match_id",
    how="inner",
    validate="one_to_one",
)

if len(elo_check) != len(historical_data):
    raise ValueError(
        "Elo verification did not match every historical fixture."
    )

home_difference = (
    elo_check["home_elo_before"]
    - elo_check["reconstructed_home_elo"]
).abs()

away_difference = (
    elo_check["away_elo_before"]
    - elo_check["reconstructed_away_elo"]
).abs()

maximum_elo_difference = max(
    home_difference.max(),
    away_difference.max(),
)

if maximum_elo_difference > 1e-9:
    raise ValueError(
        "Live Elo reconstruction does not match Elo history."
    )

print()
print(
    "Elo history rows verified:",
    len(elo_check),
)

print(
    "Maximum Elo reconstruction difference:",
    maximum_elo_difference,
)


# ============================================================
# 5. FIT CURRENT POISSON STATE
# ============================================================

poisson_teams = sorted(
    set(live_training_data["HomeTeam"])
    | set(live_training_data["AwayTeam"])
)

poisson_matches = list(
    live_training_data[
        [
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
        ]
    ].itertuples(
        index=False,
        name=None,
    )
)

(
    poisson_attacks,
    poisson_defences,
    poisson_home_advantage,
) = fit_goal_model(
    poisson_matches,
    poisson_teams,
)

poisson_team_index = build_team_index(
    poisson_teams
)

if len(poisson_attacks) != len(poisson_teams):
    raise ValueError(
        "Poisson attack parameters do not match team count."
    )

if len(poisson_defences) != len(poisson_teams):
    raise ValueError(
        "Poisson defence parameters do not match team count."
    )

print()
print(
    "Poisson teams fitted:",
    len(poisson_teams),
)

print(
    "Poisson home advantage:",
    poisson_home_advantage,
)

print(
    "Mean Poisson attack:",
    sum(poisson_attacks)
    / len(poisson_attacks),
)

print(
    "Mean Poisson defence:",
    sum(poisson_defences)
    / len(poisson_defences),
)
