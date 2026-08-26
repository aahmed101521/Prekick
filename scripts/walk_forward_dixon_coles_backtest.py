import pandas as pd

from prekick.dixon_coles import (
    dixon_coles_match_outcome_probabilities,
    fit_dixon_coles_model,
)
from prekick.goal_model import (
    build_team_index,
    expected_goals,
    get_team_parameters,
)


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
# 4. WALK-FORWARD DIXON-COLES PREDICTIONS
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

    # Jointly fit:
    # - attack parameters
    # - defence parameters
    # - home advantage
    # - Dixon-Coles rho
    (
        attacks,
        defences,
        home_advantage,
        rho,
    ) = fit_dixon_coles_model(
        matches,
        teams,
    )

    team_index = build_team_index(teams)

    dixon_coles_predictions = []

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

        # Convert expected goals into Dixon-Coles
        # home/draw/away probabilities.
        (
            home_probability,
            draw_probability,
            away_probability,
        ) = dixon_coles_match_outcome_probabilities(
            home_expected_goals,
            away_expected_goals,
            rho,
        )

        dixon_coles_predictions.append(
            (
                home_expected_goals,
                away_expected_goals,
                rho,
                home_probability,
                draw_probability,
                away_probability,
            )
        )

    (
        predict_data["dixon_coles_home_xg"],
        predict_data["dixon_coles_away_xg"],
        predict_data["dixon_coles_rho"],
        predict_data["dixon_coles_home_prob"],
        predict_data["dixon_coles_draw_prob"],
        predict_data["dixon_coles_away_prob"],
    ) = zip(*dixon_coles_predictions)

    prediction_blocks.append(
        predict_data
    )


# ============================================================
# 5. COMBINE ALL WALK-FORWARD PREDICTIONS
# ============================================================

dixon_coles_backtest = pd.concat(
    prediction_blocks,
    ignore_index=True,
)


# ============================================================
# 6. CREATE CLEAN DIXON-COLES BACKTEST OUTPUT
# ============================================================

output_columns = [
    "match_id",
    "Date",
    "season",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "dixon_coles_home_xg",
    "dixon_coles_away_xg",
    "dixon_coles_rho",
    "dixon_coles_home_prob",
    "dixon_coles_draw_prob",
    "dixon_coles_away_prob",
]

dixon_coles_output = dixon_coles_backtest[
    output_columns
].copy()


# ============================================================
# 7. VALIDATE BACKTEST ROWS
# ============================================================

if len(dixon_coles_output) != 1520:
    raise ValueError(
        "Dixon-Coles backtest does not contain the expected 1520 matches."
    )

if dixon_coles_output["match_id"].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in Dixon-Coles backtest."
    )


# ============================================================
# 8. VALIDATE EXPECTED GOALS
# ============================================================

xg_columns = [
    "dixon_coles_home_xg",
    "dixon_coles_away_xg",
]

if dixon_coles_output[
    xg_columns
].isna().any().any():
    raise ValueError(
        "Some Dixon-Coles expected-goal values are missing."
    )

if (
    dixon_coles_output[
        xg_columns
    ] <= 0
).any().any():
    raise ValueError(
        "Some Dixon-Coles expected-goal values are not positive."
    )


# ============================================================
# 9. VALIDATE RHO
# ============================================================

if dixon_coles_output[
    "dixon_coles_rho"
].isna().any():
    raise ValueError(
        "Some Dixon-Coles rho values are missing."
    )

if not dixon_coles_output[
    "dixon_coles_rho"
].between(
    -0.2,
    0.2,
).all():
    raise ValueError(
        "Some Dixon-Coles rho values are outside the fitted bounds."
    )


# ============================================================
# 10. VALIDATE DIXON-COLES PROBABILITIES
# ============================================================

dixon_coles_probability_columns = [
    "dixon_coles_home_prob",
    "dixon_coles_draw_prob",
    "dixon_coles_away_prob",
]

if dixon_coles_output[
    dixon_coles_probability_columns
].isna().any().any():
    raise ValueError(
        "Some Dixon-Coles probabilities are missing."
    )

if not dixon_coles_output[
    dixon_coles_probability_columns
].apply(
    lambda column: column.between(0, 1).all()
).all():
    raise ValueError(
        "Some Dixon-Coles probabilities are outside 0 to 1."
    )

dixon_coles_prob_sum = (
    dixon_coles_output["dixon_coles_home_prob"]
    + dixon_coles_output["dixon_coles_draw_prob"]
    + dixon_coles_output["dixon_coles_away_prob"]
)

if (
    dixon_coles_prob_sum - 1
).abs().gt(1e-12).any():
    raise ValueError(
        "Some Dixon-Coles probabilities do not sum to 1."
    )


# ============================================================
# 11. SAVE DIXON-COLES BACKTEST PREDICTIONS
# ============================================================

dixon_coles_output.to_csv(
    "predictions/dixon_coles_backtest_predictions.csv",
    index=False,
)

print(
    "Saved Dixon-Coles backtest predictions:",
    dixon_coles_output.shape,
)

print()
print("Dixon-Coles rho summary:")

print(
    dixon_coles_output[
        "dixon_coles_rho"
    ].describe()
)

print()

print(
    dixon_coles_output.tail(10).to_string(
        index=False,
    )
)
