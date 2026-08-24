import pandas as pd

from prekick.scoring import score_prediction


ledger_path = "predictions/ledger.csv"

ledger = pd.read_csv(
    ledger_path,
    dtype=str,
    keep_default_na=False,
)

immutable_columns = [
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
]

immutable_before = ledger[
    immutable_columns
].copy()


completed_matches = ledger["result"].isin(
    ["H", "D", "A"]
)

for index in ledger.index[completed_matches]:
    scores = score_prediction(
        float(ledger.at[index, "p_home"]),
        float(ledger.at[index, "p_draw"]),
        float(ledger.at[index, "p_away"]),
        ledger.at[index, "result"],
    )

    ledger.at[index, "rps"] = str(
        scores["rps"]
    )
    ledger.at[index, "log_loss"] = str(
        scores["log_loss"]
    )
    ledger.at[index, "brier"] = str(
        scores["brier"]
    )


if not ledger[
    immutable_columns
].equals(immutable_before):
    raise ValueError(
        "Immutable prediction fields were changed."
    )


unfinished_matches = ~completed_matches

if (
    ledger.loc[
        unfinished_matches,
        ["rps", "log_loss", "brier"],
    ]
    != ""
).any().any():
    raise ValueError(
        "An unfinished fixture has scoring values."
    )


ledger.to_csv(
    ledger_path,
    index=False,
)

print(
    "Completed matches scored:",
    completed_matches.sum(),
)

print(
    "Unfinished matches left unscored:",
    unfinished_matches.sum(),
)
