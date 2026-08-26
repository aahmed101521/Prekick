import pandas as pd

from prekick.goal_model import (
    build_team_index,
    expected_goals,
    fit_goal_model,
    get_team_parameters,
)
from prekick.poisson import match_outcome_probabilities


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
print("First prediction date:", prediction_dates.iloc[0])
print("Last prediction date:", prediction_dates.iloc[-1])


# ============================================================
# 4. WALK-FORWARD POISSON PREDICTIONS
# ============================================================

prediction_blocks = []

for prediction_date in prediction_dates:
    print(
        "Fitting prediction date:",
        prediction_date.date(),
    )

    # Only matches strictly before the prediction date
    # may be used for fitting.
    train_data = data[
        data["Date"] < prediction_date
    ]

    # All matches on the same date are predicted together.
    predict_data = data[
        data["Date"] == prediction_date
    ].copy()

    # Teams known from the historical training data.
    teams = sorted(
        set(train_data["HomeTeam"])
        | set(train_data["AwayTeam"])
    )

    # Historical scorelines used to fit the model.
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

    # Fit attack, defence, and home-advantage parameters.
    attacks, defences, home_advantage = fit_goal_model(
        matches,
        teams,
    )

    team_index = build_team_index(teams)

    poisson_predictions = []

    for row in predict_data.itertuples():
        # Known teams receive fitted parameters.
        # Previously unseen teams receive the neutral fallback.
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

        # Convert team strengths into expected goals.
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

        # Convert expected goals into H/D/A probabilities.
        (
            home_probability,
            draw_probability,
            away_probability,
        ) = match_outcome_probabilities(
            home_expected_goals,
            away_expected_goals,
        )

        poisson_predictions.append(
            (
                home_expected_goals,
                away_expected_goals,
                home_probability,
                draw_probability,
                away_probability,
            )
        )

    (
        predict_data["poisson_home_xg"],
        predict_data["poisson_away_xg"],
        predict_data["poisson_home_prob"],
        predict_data["poisson_draw_prob"],
        predict_data["poisson_away_prob"],
    ) = zip(*poisson_predictions)

    prediction_blocks.append(
        predict_data
    )


# ============================================================
# 5. COMBINE ALL WALK-FORWARD PREDICTIONS
# ============================================================

poisson_backtest = pd.concat(
    prediction_blocks,
    ignore_index=True,
)


# ============================================================
# 6. CREATE CLEAN POISSON BACKTEST OUTPUT
# ============================================================

output_columns = [
    "match_id",
    "Date",
    "season",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "poisson_home_xg",
    "poisson_away_xg",
    "poisson_home_prob",
    "poisson_draw_prob",
    "poisson_away_prob",
]

poisson_output = poisson_backtest[
    output_columns
].copy()


# ============================================================
# 7. VALIDATE BACKTEST ROWS
# ============================================================

if len(poisson_output) != 1520:
    raise ValueError(
        "Poisson backtest does not contain the expected 1520 matches."
    )

if poisson_output["match_id"].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in Poisson backtest."
    )


# ============================================================
# 8. VALIDATE EXPECTED GOALS
# ============================================================

xg_columns = [
    "poisson_home_xg",
    "poisson_away_xg",
]

if poisson_output[
    xg_columns
].isna().any().any():
    raise ValueError(
        "Some Poisson expected-goal values are missing."
    )

if (
    poisson_output[
        xg_columns
    ] <= 0
).any().any():
    raise ValueError(
        "Some Poisson expected-goal values are not positive."
    )


# ============================================================
# 9. VALIDATE POISSON PROBABILITIES
# ============================================================

poisson_probability_columns = [
    "poisson_home_prob",
    "poisson_draw_prob",
    "poisson_away_prob",
]

if poisson_output[
    poisson_probability_columns
].isna().any().any():
    raise ValueError(
        "Some Poisson probabilities are missing."
    )

if not poisson_output[
    poisson_probability_columns
].apply(
    lambda column: column.between(0, 1).all()
).all():
    raise ValueError(
        "Some Poisson probabilities are outside 0 to 1."
    )

poisson_prob_sum = (
    poisson_output["poisson_home_prob"]
    + poisson_output["poisson_draw_prob"]
    + poisson_output["poisson_away_prob"]
)

if (
    poisson_prob_sum - 1
).abs().gt(1e-12).any():
    raise ValueError(
        "Some Poisson probabilities do not sum to 1."
    )


# ============================================================
# 10. SAVE POISSON BACKTEST PREDICTIONS
# ============================================================

poisson_output.to_csv(
    "predictions/poisson_backtest_predictions.csv",
    index=False,
)

print(
    "Saved Poisson backtest predictions:",
    poisson_output.shape,
)

print(
    poisson_output.tail(10).to_string(
        index=False,
    )
)
