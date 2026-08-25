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
    "PSCH",
    "PSCD",
    "PSCA",
    "AvgCH",
    "AvgCD",
    "AvgCA",
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
# 2. LOAD AND STANDARDIZE ONE SEASON
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


# ============================================================
# 3. LOAD ALL SEASONS
# ============================================================

seasons = []

for season_label, filepath in season_files:
    season = load_season(
        filepath,
        season_label,
    )

    seasons.append(season)


# ============================================================
# 4. COMBINE ALL SEASONS
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

# Give every original fixture a unique identifier.
df["match_id"] = df.index


# ============================================================
# 5. CREATE HOME-TEAM PERSPECTIVE
# ============================================================

home_team_view = df[
    [
        "match_id",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "HS",
        "AS",
        "HST",
        "AST",
    ]
].copy()

home_team_view["team"] = home_team_view["HomeTeam"]
home_team_view["opponent"] = home_team_view["AwayTeam"]
home_team_view["venue"] = "Home"

home_team_view["goals_for"] = home_team_view["FTHG"]
home_team_view["goals_against"] = home_team_view["FTAG"]

home_team_view["shots_for"] = home_team_view["HS"]
home_team_view["shots_against"] = home_team_view["AS"]

home_team_view["sot_for"] = home_team_view["HST"]
home_team_view["sot_against"] = home_team_view["AST"]


# ============================================================
# 6. CREATE AWAY-TEAM PERSPECTIVE
# ============================================================

away_team_view = df[
    [
        "match_id",
        "Date",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "HS",
        "AS",
        "HST",
        "AST",
    ]
].copy()

away_team_view["team"] = away_team_view["AwayTeam"]
away_team_view["opponent"] = away_team_view["HomeTeam"]
away_team_view["venue"] = "Away"

away_team_view["goals_for"] = away_team_view["FTAG"]
away_team_view["goals_against"] = away_team_view["FTHG"]

away_team_view["shots_for"] = away_team_view["AS"]
away_team_view["shots_against"] = away_team_view["HS"]

away_team_view["sot_for"] = away_team_view["AST"]
away_team_view["sot_against"] = away_team_view["HST"]


# ============================================================
# 7. CREATE TEAM-MATCH TABLE
# ============================================================

# Each original fixture now appears twice:
# once from the home team's perspective,
# once from the away team's perspective.

team_match_table = pd.concat(
    [
        home_team_view,
        away_team_view,
    ],
    ignore_index=True,
)


# ============================================================
# 8. ORDER EACH TEAM'S HISTORY CHRONOLOGICALLY
# ============================================================

team_match_table = team_match_table.sort_values(
    ["team", "Date"]
)

matches_before = (
    team_match_table
    .groupby("team")
    .cumcount()
)

team_match_table["has_lag_1_history"] = (
    matches_before >= 1
).astype("int8")

team_match_table["has_lag_2_history"] = (
    matches_before >= 2
).astype("int8")


# ============================================================
# 9. CREATE LAGGED HISTORICAL FEATURES
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
# 10. CREATE LIST OF LAG COLUMNS
# ============================================================

lag_columns = []

for stat in history_stats:
    for lag in range(1, 3):
        lag_columns.append(
            f"{stat}_lag_{lag}"
        )

history_indicator_columns = [
    "has_lag_1_history",
    "has_lag_2_history",
]


# ============================================================
# 11. SEPARATE HOME AND AWAY HISTORIES
# ============================================================

home_history = team_match_table[
    team_match_table["venue"] == "Home"
]

away_history = team_match_table[
    team_match_table["venue"] == "Away"
]


# ============================================================
# 12. KEEP ONLY HISTORICAL FEATURES
# ============================================================

home_features = home_history[
    ["match_id"] + lag_columns + history_indicator_columns
].copy()

away_features = away_history[
    ["match_id"] + lag_columns + history_indicator_columns
].copy()


# ============================================================
# 13. RENAME HOME AND AWAY FEATURES
# ============================================================

home_features = home_features.rename(
    columns={
        column: f"home_{column}"
        for column in lag_columns + history_indicator_columns
    }
)

away_features = away_features.rename(
    columns={
        column: f"away_{column}"
        for column in lag_columns + history_indicator_columns
    }
)


# ============================================================
# 14. MERGE BOTH TEAMS' HISTORY
# ============================================================

match_features = home_features.merge(
    away_features,
    on="match_id",
)


# ============================================================
# 15. CREATE MATCH INFORMATION TABLE
# ============================================================

match_info = df[
    [
        "match_id",
        "Date",
        "season",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "FTR",
        "PSCH",
        "PSCD",
        "PSCA",
        "AvgCH",
        "AvgCD",
        "AvgCA",
    ]
].copy()

pinnacle_complete = match_info[
    ["PSCH", "PSCD", "PSCA"]
].notna().all(axis=1)

average_complete = match_info[
    ["AvgCH", "AvgCD", "AvgCA"]
].notna().all(axis=1)

match_info["market_odds_source"] = pd.NA

match_info.loc[
    pinnacle_complete,
    "market_odds_source",
] = "Pinnacle"

match_info.loc[
    ~pinnacle_complete & average_complete,
    "market_odds_source",
] = "Average"

match_info["market_home_odds"] = match_info["PSCH"]
match_info["market_draw_odds"] = match_info["PSCD"]
match_info["market_away_odds"] = match_info["PSCA"]

fallback_mask = (
    match_info["market_odds_source"] == "Average"
)

match_info.loc[
    fallback_mask,
    "market_home_odds",
] = match_info.loc[fallback_mask, "AvgCH"]

match_info.loc[
    fallback_mask,
    "market_draw_odds",
] = match_info.loc[fallback_mask, "AvgCD"]

match_info.loc[
    fallback_mask,
    "market_away_odds",
] = match_info.loc[fallback_mask, "AvgCA"]

selected_market_odds = [
    "market_home_odds",
    "market_draw_odds",
    "market_away_odds",
]

if match_info[selected_market_odds].isna().any().any():
    raise ValueError(
        "Some matches do not have complete market odds."
    )

if (match_info[selected_market_odds] <= 1).any().any():
    raise ValueError(
        "Some selected market odds are less than or equal to 1."
    )

match_info["market_raw_home_prob"] = (
    1 / match_info["market_home_odds"]
)

match_info["market_raw_draw_prob"] = (
    1 / match_info["market_draw_odds"]
)

match_info["market_raw_away_prob"] = (
    1 / match_info["market_away_odds"]
)

match_info["market_raw_prob_sum"] = (
    match_info["market_raw_home_prob"]
    + match_info["market_raw_draw_prob"]
    + match_info["market_raw_away_prob"]
)

match_info["market_overround"] = (
    match_info["market_raw_prob_sum"] - 1
)

match_info["market_home_prob"] = (
    match_info["market_raw_home_prob"]
    / match_info["market_raw_prob_sum"]
)

match_info["market_draw_prob"] = (
    match_info["market_raw_draw_prob"]
    / match_info["market_raw_prob_sum"]
)

match_info["market_away_prob"] = (
    match_info["market_raw_away_prob"]
    / match_info["market_raw_prob_sum"]
)

market_prob_sum = (
    match_info["market_home_prob"]
    + match_info["market_draw_prob"]
    + match_info["market_away_prob"]
)

if (market_prob_sum - 1).abs().gt(1e-12).any():
    raise ValueError(
        "Market probabilities do not sum to 1."
    )


# ============================================================
# 16. CREATE FINAL MODEL DATASET
# ============================================================

model_data = match_info.merge(
    match_features,
    on="match_id",
)

model_data.to_csv(
    "data/processed/model_data.csv",
    index=False,
)


# ============================================================
# 17. DEFINE MODEL FEATURES X AND TARGET y
# ============================================================

feature_columns = (
    [f"home_{column}" for column in lag_columns]
    + [f"away_{column}" for column in lag_columns]
    + [f"home_{column}" for column in history_indicator_columns]
    + [f"away_{column}" for column in history_indicator_columns]
)

X = model_data[feature_columns]

y = model_data["FTR"]


# ============================================================
# 18. SANITY CHECKS
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

print(
    "Matches with at least one missing feature:",
    X.isna().any(axis=1).sum(),
)

saved_data = pd.read_csv(
    "data/processed/model_data.csv"
)

print("Saved model data:", saved_data.shape)
print(saved_data.head())
