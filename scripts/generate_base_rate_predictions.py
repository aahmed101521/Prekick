import pandas as pd
from datetime import datetime, timezone

# ============================================================
# 1. LOAD HISTORICAL RESULTS
# ============================================================

historical_data = pd.read_csv(
    "data/processed/model_data.csv"
)

print(historical_data.shape)
print(historical_data["FTR"].value_counts())

# ============================================================
# 2. CALCULATE LEAGUE BASE RATES
# ============================================================

result_counts = historical_data["FTR"].value_counts()

total_matches = len(historical_data)

p_home = result_counts["H"] / total_matches
p_draw = result_counts["D"] / total_matches
p_away = result_counts["A"] / total_matches

print("P(Home):", p_home)
print("P(Draw):", p_draw)
print("P(Away):", p_away)

print(
    "Probability sum:",
    p_home + p_draw + p_away,
)

# ============================================================
# 3. DEFINE MATCHWEEK 1 FIXTURES
# ============================================================

fixtures = [
    {
        "fixture_id": "2026-27_mw01_arsenal_coventry-city",
        "kickoff_utc": "2026-08-21T19:00:00Z",
        "home_team": "Arsenal",
        "away_team": "Coventry City",
    },
    {
        "fixture_id": "2026-27_mw01_hull-city_manchester-united",
        "kickoff_utc": "2026-08-22T11:30:00Z",
        "home_team": "Hull City",
        "away_team": "Manchester United",
    },
    {
        "fixture_id": "2026-27_mw01_everton_crystal-palace",
        "kickoff_utc": "2026-08-22T14:00:00Z",
        "home_team": "Everton",
        "away_team": "Crystal Palace",
    },
    {
        "fixture_id": "2026-27_mw01_ipswich-town_sunderland",
        "kickoff_utc": "2026-08-22T14:00:00Z",
        "home_team": "Ipswich Town",
        "away_team": "Sunderland",
    },
    {
        "fixture_id": "2026-27_mw01_nottingham-forest_leeds-united",
        "kickoff_utc": "2026-08-22T14:00:00Z",
        "home_team": "Nottingham Forest",
        "away_team": "Leeds United",
    },
    {
        "fixture_id": "2026-27_mw01_brentford_tottenham-hotspur",
        "kickoff_utc": "2026-08-22T16:30:00Z",
        "home_team": "Brentford",
        "away_team": "Tottenham Hotspur",
    },
    {
        "fixture_id": "2026-27_mw01_brighton_aston-villa",
        "kickoff_utc": "2026-08-23T13:00:00Z",
        "home_team": "Brighton & Hove Albion",
        "away_team": "Aston Villa",
    },
    {
        "fixture_id": "2026-27_mw01_manchester-city_bournemouth",
        "kickoff_utc": "2026-08-23T13:00:00Z",
        "home_team": "Manchester City",
        "away_team": "AFC Bournemouth",
    },
    {
        "fixture_id": "2026-27_mw01_newcastle-united_liverpool",
        "kickoff_utc": "2026-08-23T15:30:00Z",
        "home_team": "Newcastle United",
        "away_team": "Liverpool",
    },
    {
        "fixture_id": "2026-27_mw01_fulham_chelsea",
        "kickoff_utc": "2026-08-24T19:00:00Z",
        "home_team": "Fulham",
        "away_team": "Chelsea",
    },
]

fixtures_df = pd.DataFrame(fixtures)

# ============================================================
# 4. ATTACH BASE-RATE PREDICTIONS
# ============================================================

fixtures_df["season"] = "2026_27"
fixtures_df["matchweek"] = 1

fixtures_df["model_version"] = "base_rate_v1"

fixtures_df["training_cutoff_utc"] = (
    "2026-05-24T23:59:59Z"
)

fixtures_df["p_home"] = p_home
fixtures_df["p_draw"] = p_draw
fixtures_df["p_away"] = p_away

print(
    fixtures_df[
        [
            "home_team",
            "away_team",
            "model_version",
            "p_home",
            "p_draw",
            "p_away",
        ]
    ]
)

print(
    "Number of fixtures:",
    len(fixtures_df),
)

# ============================================================
# 5. VALIDATE PREDICTIONS
# ============================================================

probability_sums = (
    fixtures_df["p_home"]
    + fixtures_df["p_draw"]
    + fixtures_df["p_away"]
)

print(probability_sums)

if not probability_sums.between(
    0.999999,
    1.000001,
).all():
    raise ValueError(
        "Prediction probabilities do not sum to 1."
    )
    
if fixtures_df["fixture_id"].duplicated().any():
    raise ValueError(
        "Duplicate fixture IDs found."
    )    
    
    
# ============================================================
# 6. CHECK AGAINST EXISTING LEDGER
# ============================================================

ledger_path = "predictions/ledger.csv"

existing_ledger = pd.read_csv(ledger_path)

existing_fixture_ids = set(
    existing_ledger["fixture_id"]
)

new_fixture_ids = set(
    fixtures_df["fixture_id"]
)

duplicate_fixture_ids = (
    existing_fixture_ids
    .intersection(new_fixture_ids)
)

print(
    "Existing ledger rows:",
    len(existing_ledger),
)

print(
    "Duplicate fixture IDs:",
    duplicate_fixture_ids,
)

if duplicate_fixture_ids:
    raise ValueError(
        "Some fixtures already exist in the prediction ledger."
    )    
    
    
# ============================================================
# 7. PREPARE LEDGER ROWS
# ============================================================

prediction_time = (
    datetime
    .now(timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z")
)

fixtures_df["predicted_at_utc"] = prediction_time    


fixtures_df["home_goals"] = pd.NA
fixtures_df["away_goals"] = pd.NA
fixtures_df["result"] = pd.NA

fixtures_df["rps"] = pd.NA
fixtures_df["log_loss"] = pd.NA
fixtures_df["brier"] = pd.NA


ledger_columns = [
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

ledger_rows = fixtures_df[
    ledger_columns
].copy()

print(ledger_rows.to_string(index=False))

if list(existing_ledger.columns) != ledger_columns:
    raise ValueError(
        "Ledger columns do not match the expected schema."
    )

ledger_rows.to_csv(
    ledger_path,
    mode="a",
    header=False,
    index=False,
)