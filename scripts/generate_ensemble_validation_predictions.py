import pandas as pd

from prekick.elo import (
    fit_draw_parameter,
    three_way_probabilities,
)
from prekick.goal_model import (
    build_team_index,
    expected_goals,
    fit_goal_model,
    get_team_parameters,
)
from prekick.poisson import match_outcome_probabilities


# ============================================================
# 1. LOAD DATA
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


# ============================================================
# 2. DEFINE PRE-BACKTEST VALIDATION PERIOD
# ============================================================

validation_start = pd.Timestamp("2022-03-01")
backtest_start = pd.Timestamp("2022-08-05")

validation_data = data[
    (data["Date"] >= validation_start)
    & (data["Date"] < backtest_start)
].copy()

print(
    "Validation matches:",
    len(validation_data),
)

print(
    "First validation date:",
    validation_data["Date"].min(),
)

print(
    "Last validation date:",
    validation_data["Date"].max(),
)


# ============================================================
# 3. CREATE PREDICTION-DATE BLOCKS
# ============================================================

prediction_dates = (
    validation_data["Date"]
    .drop_duplicates()
    .sort_values()
)

print(
    "Prediction-date blocks:",
    len(prediction_dates),
)


# ============================================================
# 4. WALK-FORWARD VALIDATION PREDICTIONS
# ============================================================

prediction_blocks = []

for prediction_date in prediction_dates:
    print(
        "Fitting validation date:",
        prediction_date.date(),
    )

    # Strictly historical training data.
    train_data = data[
        data["Date"] < prediction_date
    ]

    # Predict every match on this date together.
    predict_data = data[
        data["Date"] == prediction_date
    ].copy()

    # --------------------------------------------------------
    # ELO
    # --------------------------------------------------------

    draw_parameter = fit_draw_parameter(
        train_data["home_elo_before"],
        train_data["away_elo_before"],
        train_data["FTR"],
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

    # --------------------------------------------------------
    # POISSON
    # --------------------------------------------------------

    teams = sorted(
        set(train_data["HomeTeam"])
        | set(train_data["AwayTeam"])
    )

    matches = list(
        train_data[
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

    attacks, defences, home_advantage = fit_goal_model(
        matches,
        teams,
    )

    team_index = build_team_index(teams)

    poisson_probabilities = []

    for row in predict_data.itertuples():
        home_attack, home_defence = get_team_parameters(
            row.HomeTeam,
            team_index,
            attacks,
            defences,
        )

        away_attack, away_defence = get_team_parameters(
            row.AwayTeam,
            team_index,
            attacks,
            defences,
        )

        (
            home_expected_goals,
            away_expected_goals,
        ) = expected_goals(
            home_attack=home_attack,
            home_defence=home_defence,
            away_attack=away_attack,
            away_defence=away_defence,
            home_advantage=home_advantage,
        )

        probabilities = match_outcome_probabilities(
            home_expected_goals,
            away_expected_goals,
        )

        poisson_probabilities.append(
            probabilities
        )

    (
        predict_data["poisson_home_prob"],
        predict_data["poisson_draw_prob"],
        predict_data["poisson_away_prob"],
    ) = zip(*poisson_probabilities)

    prediction_blocks.append(
        predict_data
    )


# ============================================================
# 5. COMBINE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = pd.concat(
    prediction_blocks,
    ignore_index=True,
)


# ============================================================
# 6. CREATE CLEAN OUTPUT
# ============================================================

output_columns = [
    "match_id",
    "Date",
    "season",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "elo_home_prob",
    "elo_draw_prob",
    "elo_away_prob",
    "poisson_home_prob",
    "poisson_draw_prob",
    "poisson_away_prob",
]

validation_output = validation_predictions[
    output_columns
].copy()


# ============================================================
# 7. VALIDATE OUTPUT
# ============================================================

if len(validation_output) != 124:
    raise ValueError(
        "Validation output does not contain the expected 124 matches."
    )

if validation_output["match_id"].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in validation predictions."
    )

probability_columns = [
    "elo_home_prob",
    "elo_draw_prob",
    "elo_away_prob",
    "poisson_home_prob",
    "poisson_draw_prob",
    "poisson_away_prob",
]

if validation_output[
    probability_columns
].isna().any().any():
    raise ValueError(
        "Some validation probabilities are missing."
    )

elo_sum = (
    validation_output["elo_home_prob"]
    + validation_output["elo_draw_prob"]
    + validation_output["elo_away_prob"]
)

poisson_sum = (
    validation_output["poisson_home_prob"]
    + validation_output["poisson_draw_prob"]
    + validation_output["poisson_away_prob"]
)

if (
    elo_sum - 1
).abs().gt(1e-12).any():
    raise ValueError(
        "Some Elo validation probabilities do not sum to 1."
    )

if (
    poisson_sum - 1
).abs().gt(1e-12).any():
    raise ValueError(
        "Some Poisson validation probabilities do not sum to 1."
    )


# ============================================================
# 8. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_output.to_csv(
    "predictions/ensemble_validation_predictions.csv",
    index=False,
)

print(
    "Saved ensemble validation predictions:",
    validation_output.shape,
)

print(
    validation_output.tail(10).to_string(
        index=False,
    )
)
