import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

required_columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
]

history_stats = [
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
]

season_files = [
    ("2021_22", "data/raw/epl_2021_22.csv"),
    ("2022_23", "data/raw/epl_2022_23.csv"),
    ("2023_24", "data/raw/epl_2023_24.csv"),
    ("2024_25", "data/raw/epl_2024_25.csv"),
    ("2025_26", "data/raw/epl_2025_26.csv"),
]




# ============================================================
# 2. LOAD AND STANDARDIZE EACH SEASON
# ============================================================

def load_season(filepath, season_label):
    raw = pd.read_csv(filepath)

    missing_columns = set(required_columns) - set(raw.columns)

    if missing_columns:
        raise ValueError(
            f"{season_label} is missing required columns: "
            f"{missing_columns}"
        )

    season = raw[required_columns].copy()

    season["season"] = season_label

    season["Date"] = pd.to_datetime(
        season["Date"],
        dayfirst=True,
    )

    return season


seasons = []

for season_label, filepath in season_files:
    season = load_season(filepath, season_label)
    seasons.append(season)


# ============================================================
# 3. COMBINE ALL SEASONS INTO ONE MATCH TABLE
# ============================================================

df = pd.concat(
    seasons,
    ignore_index=True,
)

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

# Give every original fixture a unique ID.
df["match_id"] = df.index


# ============================================================
# 4. CREATE TEAM-CENTRIC MATCH VIEWS
# ============================================================

# Home-team perspective
home_team_view = df.copy()

home_team_view["team"] = home_team_view["HomeTeam"]
home_team_view["opponent"] = home_team_view["AwayTeam"]
home_team_view["venue"] = "Home"

home_team_view["goals_for"] = home_team_view["FTHG"]
home_team_view["goals_against"] = home_team_view["FTAG"]

home_team_view["shots_for"] = home_team_view["HS"]
home_team_view["shots_against"] = home_team_view["AS"]

home_team_view["sot_for"] = home_team_view["HST"]
home_team_view["sot_against"] = home_team_view["AST"]


# Away-team perspective
away_team_view = df.copy()

away_team_view["team"] = away_team_view["AwayTeam"]
away_team_view["opponent"] = away_team_view["HomeTeam"]
away_team_view["venue"] = "Away"

away_team_view["goals_for"] = away_team_view["FTAG"]
away_team_view["goals_against"] = away_team_view["FTHG"]

away_team_view["shots_for"] = away_team_view["AS"]
away_team_view["shots_against"] = away_team_view["HS"]

away_team_view["sot_for"] = away_team_view["AST"]
away_team_view["sot_against"] = away_team_view["HST"]


# Combine both perspectives.
# Each original match now appears twice.
team_match_table = pd.concat(
    [home_team_view, away_team_view],
    ignore_index=True,
)


# ============================================================
# 5. ORDER EACH TEAM'S HISTORY CHRONOLOGICALLY
# ============================================================

team_match_table = team_match_table.sort_values(
    ["team", "Date"]
)


# ============================================================
# 6. CREATE LAGGED HISTORICAL FEATURES
# ============================================================

for stat in history_stats:
    for lag in range(1, 3):
        team_match_table[f"{stat}_lag_{lag}"] = (
            team_match_table
            .groupby("team")[stat]
            .shift(lag)
            .astype("Int64")
        )


# ============================================================
# 7. CREATE LIST OF HISTORICAL FEATURE COLUMNS
# ============================================================

lag_columns = []

for stat in history_stats:
    for lag in range(1, 3):
        lag_columns.append(
            f"{stat}_lag_{lag}"
        )


# ============================================================
# 8. SPLIT HOME AND AWAY HISTORIES
# ============================================================

home_history = team_match_table[
    team_match_table["venue"] == "Home"
]

away_history = team_match_table[
    team_match_table["venue"] == "Away"
]


# Keep only match_id + historical information.
home_features = home_history[
    ["match_id"] + lag_columns
].copy()

away_features = away_history[
    ["match_id"] + lag_columns
].copy()


# ============================================================
# 9. RENAME HOME AND AWAY FEATURES
# ============================================================

home_features = home_features.rename(
    columns={
        column: f"home_{column}"
        for column in lag_columns
    }
)

away_features = away_features.rename(
    columns={
        column: f"away_{column}"
        for column in lag_columns
    }
)


# ============================================================
# 10. MERGE BOTH TEAMS' HISTORY INTO ONE MATCH ROW
# ============================================================

match_features = home_features.merge(
    away_features,
    on="match_id",
)


# ============================================================
# 11. ADD MATCH INFORMATION AND TARGET
# ============================================================

match_info = df[
    [
        "match_id",
        "Date",
        "season",
        "HomeTeam",
        "AwayTeam",
        "FTR",
    ]
].copy()

model_data = match_info.merge(
    match_features,
    on="match_id",
)


# ============================================================
# 12. DEFINE MODEL INPUTS X AND TARGET y
# ============================================================

feature_columns = (
    [f"home_{column}" for column in lag_columns]
    + [f"away_{column}" for column in lag_columns]
)

X = model_data[feature_columns]

y = model_data["FTR"]


# ============================================================
# 13. BASIC SANITY CHECKS
# ============================================================

print("Combined match data:", df.shape)
print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max(),
)

print("Team-match table:", team_match_table.shape)
print("Model data:", model_data.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

# print(X.isna().sum())

missing_lag_1 = model_data[
    model_data["home_goals_for_lag_1"].isna()
    | model_data["away_goals_for_lag_1"].isna()
]

"""
print(
    missing_lag_1[
        [
            "Date",
            "season",
            "HomeTeam",
            "AwayTeam",
            "home_goals_for_lag_1",
            "away_goals_for_lag_1",
        ]
    ].to_string(index=False)
)
"""

complete_rows = X.dropna()

print("Complete rows:", complete_rows.shape)

print(
    "Matches with at least one missing feature:",
    X.isna().any(axis=1).sum(),
)