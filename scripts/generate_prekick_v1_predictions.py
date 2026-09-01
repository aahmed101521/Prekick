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

expected_live_training_matches = (
    len(historical_data)
    + len(current_season_model_data)
)

if len(live_training_data) != expected_live_training_matches:
    raise ValueError(
        "Live training row count does not equal "
        "historical plus current-season matches."
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


# ============================================================
# 6. LOAD AND VALIDATE UPCOMING FIXTURES
# ============================================================

fixtures = pd.read_csv(
    "data/fixtures/upcoming_fixtures.csv",
)

expected_fixture_columns = [
    "fixture_id",
    "season",
    "matchweek",
    "kickoff_utc",
    "home_team",
    "away_team",
]

if list(fixtures.columns) != expected_fixture_columns:
    raise ValueError(
        "Upcoming fixture columns do not match the expected schema."
    )

if fixtures.empty:
    raise ValueError(
        "No upcoming fixtures found."
    )

if fixtures["fixture_id"].duplicated().any():
    raise ValueError(
        "Duplicate fixture IDs found."
    )

known_live_teams = set(
    live_training_data["HomeTeam"]
) | set(
    live_training_data["AwayTeam"]
)

fixture_teams = set(
    fixtures["home_team"]
) | set(
    fixtures["away_team"]
)

unknown_fixture_teams = sorted(
    fixture_teams - known_live_teams
)

print()
print(
    "Upcoming fixtures:",
    len(fixtures),
)

print(
    "Unknown fixture teams:",
    unknown_fixture_teams,
)

print()
print(
    fixtures[
        [
            "kickoff_utc",
            "home_team",
            "away_team",
        ]
    ].to_string(
        index=False,
    )
)


# ============================================================
# 7. GENERATE ELO, POISSON, AND PREKICK PREDICTIONS
# ============================================================

prediction_rows = []

for fixture in fixtures.itertuples():
    # --------------------------------------------------------
    # ELO
    # --------------------------------------------------------

    home_elo = elo_ratings.get(
        fixture.home_team,
        INITIAL_ELO_RATING,
    )

    away_elo = elo_ratings.get(
        fixture.away_team,
        INITIAL_ELO_RATING,
    )

    (
        elo_home_prob,
        elo_draw_prob,
        elo_away_prob,
    ) = three_way_probabilities(
        home_elo,
        away_elo,
        elo_draw_parameter,
        home_advantage=ELO_HOME_ADVANTAGE,
    )

    # --------------------------------------------------------
    # POISSON
    # --------------------------------------------------------

    (
        home_attack,
        home_defence,
    ) = get_team_parameters(
        fixture.home_team,
        poisson_team_index,
        poisson_attacks,
        poisson_defences,
    )

    (
        away_attack,
        away_defence,
    ) = get_team_parameters(
        fixture.away_team,
        poisson_team_index,
        poisson_attacks,
        poisson_defences,
    )

    (
        poisson_home_xg,
        poisson_away_xg,
    ) = expected_goals(
        home_attack=home_attack,
        home_defence=home_defence,
        away_attack=away_attack,
        away_defence=away_defence,
        home_advantage=poisson_home_advantage,
    )

    (
        poisson_home_prob,
        poisson_draw_prob,
        poisson_away_prob,
    ) = match_outcome_probabilities(
        poisson_home_xg,
        poisson_away_xg,
    )

    # --------------------------------------------------------
    # 50/50 PREKICK V1 ENSEMBLE
    # --------------------------------------------------------

    p_home = (
        ELO_WEIGHT * elo_home_prob
        + POISSON_WEIGHT * poisson_home_prob
    )

    p_draw = (
        ELO_WEIGHT * elo_draw_prob
        + POISSON_WEIGHT * poisson_draw_prob
    )

    p_away = (
        ELO_WEIGHT * elo_away_prob
        + POISSON_WEIGHT * poisson_away_prob
    )

    prediction_rows.append(
        {
            "fixture_id": fixture.fixture_id,
            "season": fixture.season,
            "matchweek": fixture.matchweek,
            "kickoff_utc": fixture.kickoff_utc,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_home_prob": elo_home_prob,
            "elo_draw_prob": elo_draw_prob,
            "elo_away_prob": elo_away_prob,
            "poisson_home_xg": poisson_home_xg,
            "poisson_away_xg": poisson_away_xg,
            "poisson_home_prob": poisson_home_prob,
            "poisson_draw_prob": poisson_draw_prob,
            "poisson_away_prob": poisson_away_prob,
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
        }
    )


predictions = pd.DataFrame(
    prediction_rows
)


# ============================================================
# 8. VALIDATE PREKICK PREDICTIONS
# ============================================================

if len(predictions) != len(fixtures):
    raise ValueError(
        "Prediction row count does not match fixture count."
    )

if predictions["fixture_id"].duplicated().any():
    raise ValueError(
        "Duplicate fixture IDs found in Prekick predictions."
    )

probability_columns = [
    "p_home",
    "p_draw",
    "p_away",
]

if not predictions[
    probability_columns
].apply(
    lambda column: column.between(0, 1).all()
).all():
    raise ValueError(
        "Some Prekick probabilities are outside 0 to 1."
    )

probability_sum = (
    predictions["p_home"]
    + predictions["p_draw"]
    + predictions["p_away"]
)

if (
    probability_sum - 1.0
).abs().gt(1e-12).any():
    raise ValueError(
        "Some Prekick probabilities do not sum to 1."
    )


# ============================================================
# 9. DISPLAY PREKICK PREDICTIONS
# ============================================================

print()
print(
    "Prekick v1 predictions:",
    len(predictions),
)

print()
print(
    predictions[
        [
            "home_team",
            "away_team",
            "poisson_home_xg",
            "poisson_away_xg",
            "elo_home_prob",
            "elo_draw_prob",
            "elo_away_prob",
            "poisson_home_prob",
            "poisson_draw_prob",
            "poisson_away_prob",
            "p_home",
            "p_draw",
            "p_away",
        ]
    ].to_string(
        index=False,
    )
)


# ============================================================
# 10. PREPARE LEDGER ROWS
# ============================================================

ledger_path = "predictions/ledger.csv"

expected_ledger_columns = [
    "fixture_id",
    "season",
    "matchweek",
    "kickoff_utc",
    "home_team",
    "away_team",
    "model_version",
    "training_cutoff_utc",
    "predicted_at_utc",
    "p_home",
    "p_draw",
    "p_away",
    "home_goals",
    "away_goals",
    "result",
    "rps",
    "log_loss",
    "brier",
]

ledger = pd.read_csv(
    ledger_path,
    dtype=str,
    keep_default_na=False,
)

if list(ledger.columns) != expected_ledger_columns:
    raise ValueError(
        "Ledger columns do not match the expected schema."
    )

existing_fixture_ids = set(
    ledger["fixture_id"]
)

duplicate_fixture_ids = sorted(
    set(predictions["fixture_id"])
    & existing_fixture_ids
)

if duplicate_fixture_ids:
    raise ValueError(
        "These fixture IDs already exist in the ledger: "
        + ", ".join(duplicate_fixture_ids)
    )

training_cutoff_utc = (
    live_training_data["Date"].max()
    .strftime("%Y-%m-%dT23:59:59Z")
)

predicted_at_utc = (
    datetime.now(timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z")
)

ledger_rows = predictions[
    [
        "fixture_id",
        "season",
        "matchweek",
        "kickoff_utc",
        "home_team",
        "away_team",
        "p_home",
        "p_draw",
        "p_away",
    ]
].copy()

ledger_rows["model_version"] = (
    MODEL_VERSION
)

ledger_rows["training_cutoff_utc"] = (
    training_cutoff_utc
)

ledger_rows["predicted_at_utc"] = (
    predicted_at_utc
)

ledger_rows["home_goals"] = ""
ledger_rows["away_goals"] = ""
ledger_rows["result"] = ""
ledger_rows["rps"] = ""
ledger_rows["log_loss"] = ""
ledger_rows["brier"] = ""

ledger_rows = ledger_rows[
    expected_ledger_columns
]

if len(ledger_rows) != len(predictions):
    raise ValueError(
        "Ledger row count does not match prediction count."
    )

if ledger_rows["fixture_id"].duplicated().any():
    raise ValueError(
        "Duplicate fixture IDs found in ledger rows."
    )


# ============================================================
# 11. DISPLAY CANDIDATE LEDGER ROWS
# ============================================================

print()
print(
    "Candidate ledger rows:",
    len(ledger_rows),
)

print()
print(
    ledger_rows.to_string(
        index=False,
    )
)


# ============================================================
# 12. APPEND PREKICK V1 PREDICTIONS TO LEDGER
# ============================================================

updated_ledger = pd.concat(
    [
        ledger,
        ledger_rows,
    ],
    ignore_index=True,
)

if len(updated_ledger) != (
    len(ledger) + len(ledger_rows)
):
    raise ValueError(
        "Unexpected ledger row count after append."
    )

updated_ledger.to_csv(
    ledger_path,
    index=False,
)

print()
print(
    "Prekick v1 rows appended:",
    len(ledger_rows),
)

print(
    "Total ledger rows:",
    len(updated_ledger),
)
