import pandas as pd

from prekick.elo import (
    expected_score,
    update_match,
)


INITIAL_RATING = 1500
HOME_ADVANTAGE = 100
K_FACTOR = 20


# ============================================================
# 1. LOAD MATCH DATA
# ============================================================

data = pd.read_csv(
    "data/processed/model_data.csv",
    parse_dates=["Date"],
)

data = data.sort_values(
    ["Date", "match_id"]
).reset_index(drop=True)


# ============================================================
# 2. CHECK SAME-DAY TEAM DUPLICATES
# ============================================================

team_dates = pd.concat(
    [
        data[
            ["Date", "HomeTeam"]
        ].rename(
            columns={"HomeTeam": "team"}
        ),
        data[
            ["Date", "AwayTeam"]
        ].rename(
            columns={"AwayTeam": "team"}
        ),
    ],
    ignore_index=True,
)

if team_dates.duplicated(
    ["Date", "team"]
).any():
    raise ValueError(
        "A team appears more than once on the same date."
    )


# ============================================================
# 3. BUILD CHRONOLOGICAL ELO HISTORY
# ============================================================

ratings = {}
elo_rows = []

for _, match in data.iterrows():
    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    home_rating = ratings.get(
        home_team,
        INITIAL_RATING,
    )

    away_rating = ratings.get(
        away_team,
        INITIAL_RATING,
    )

    expected_home = expected_score(
        home_rating + HOME_ADVANTAGE,
        away_rating,
    )

    elo_rows.append(
        {
            "match_id": match["match_id"],
            "Date": match["Date"],
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "FTR": match["FTR"],
            "home_elo_before": home_rating,
            "away_elo_before": away_rating,
            "elo_expected_home": expected_home,
        }
    )

    new_home_rating, new_away_rating = update_match(
        home_rating,
        away_rating,
        match["FTR"],
        home_advantage=HOME_ADVANTAGE,
        k_factor=K_FACTOR,
    )

    ratings[home_team] = new_home_rating
    ratings[away_team] = new_away_rating


elo_history = pd.DataFrame(
    elo_rows
)


# ============================================================
# 4. VALIDATE ELO HISTORY
# ============================================================

if len(elo_history) != len(data):
    raise ValueError(
        "Elo history does not contain one row per match."
    )

if elo_history["match_id"].duplicated().any():
    raise ValueError(
        "Duplicate match IDs found in Elo history."
    )

if not elo_history[
    "elo_expected_home"
].between(0, 1).all():
    raise ValueError(
        "Some Elo expected scores are outside 0 to 1."
    )

expected_rating_total = (
    len(ratings) * INITIAL_RATING
)

actual_rating_total = sum(
    ratings.values()
)

if abs(
    actual_rating_total
    - expected_rating_total
) > 1e-9:
    raise ValueError(
        "Elo rating points were not conserved."
    )


# ============================================================
# 5. SAVE ELO HISTORY
# ============================================================

elo_history.to_csv(
    "data/processed/elo_history.csv",
    index=False,
)


# ============================================================
# 6. SUMMARY
# ============================================================

print(
    "Matches processed:",
    len(elo_history),
)

print(
    "Teams rated:",
    len(ratings),
)

print(
    "Final rating total:",
    actual_rating_total,
)

print(
    "Saved Elo history:",
    elo_history.shape,
)
