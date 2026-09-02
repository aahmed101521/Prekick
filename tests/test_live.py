import pandas as pd
import pytest

from prekick.live import (
    find_duplicate_fixture_ids,
    reconcile_completed_results,
    reject_duplicate_fixture_ids,
    validate_fixture_count,
    validate_live_training_count,
    validate_prediction_count,
)


def test_validate_live_training_count_accepts_dynamic_count():
    validate_live_training_count(
        historical_count=1900,
        current_season_count=30,
        live_training_count=1930,
    )


def test_validate_live_training_count_rejects_mismatch():
    with pytest.raises(ValueError):
        validate_live_training_count(
            historical_count=1900,
            current_season_count=30,
            live_training_count=1920,
        )


def test_validate_fixture_count_accepts_nonzero_count():
    validate_fixture_count(
        fixture_count=7,
    )


def test_validate_fixture_count_rejects_zero():
    with pytest.raises(ValueError):
        validate_fixture_count(
            fixture_count=0,
        )


def test_validate_prediction_count_accepts_matching_count():
    validate_prediction_count(
        prediction_count=7,
        fixture_count=7,
    )


def test_validate_prediction_count_rejects_mismatch():
    with pytest.raises(ValueError):
        validate_prediction_count(
            prediction_count=6,
            fixture_count=7,
        )


def test_find_duplicate_fixture_ids():
    duplicate_fixture_ids = (
        find_duplicate_fixture_ids(
            prediction_fixture_ids=[
                "fixture_1",
                "fixture_2",
                "fixture_3",
            ],
            existing_fixture_ids=[
                "fixture_2",
                "fixture_4",
            ],
        )
    )

    assert duplicate_fixture_ids == [
        "fixture_2",
    ]


def test_reject_duplicate_fixture_ids_accepts_new_fixtures():
    reject_duplicate_fixture_ids(
        prediction_fixture_ids=[
            "fixture_1",
            "fixture_2",
        ],
        existing_fixture_ids=[
            "fixture_3",
            "fixture_4",
        ],
    )


def test_reject_duplicate_fixture_ids_rejects_existing_fixture():
    with pytest.raises(
        ValueError,
        match="fixture_2",
    ):
        reject_duplicate_fixture_ids(
            prediction_fixture_ids=[
                "fixture_1",
                "fixture_2",
            ],
            existing_fixture_ids=[
                "fixture_2",
                "fixture_3",
            ],
        )


def _ledger_row(
    fixture_id,
    season,
    matchweek,
    home_team,
    away_team,
    home_goals="",
    away_goals="",
    result="",
):
    return {
        "fixture_id": fixture_id,
        "season": season,
        "matchweek": str(matchweek),
        "kickoff_utc": "2026-08-21T19:00:00Z",
        "home_team": home_team,
        "away_team": away_team,
        "model_version": "prekick_v1",
        "training_cutoff_utc": "2026-08-20T23:59:59Z",
        "predicted_at_utc": "2026-08-20T12:00:00Z",
        "p_home": "0.50",
        "p_draw": "0.25",
        "p_away": "0.25",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "rps": "",
        "log_loss": "",
        "brier": "",
    }


def test_reconcile_completed_results_fills_blank_result():
    ledger = pd.DataFrame(
        [
            _ledger_row(
                fixture_id=(
                    "2026-27_mw03_"
                    "everton_manchester-united"
                ),
                season="2026_27",
                matchweek=3,
                home_team="Everton",
                away_team="Man United",
            )
        ]
    )

    completed_results = pd.DataFrame(
        [
            {
                "season": "2026_27",
                "matchweek": 3,
                "home_team": "Everton",
                "away_team": "Man United",
                "home_goals": 2,
                "away_goals": 1,
                "result": "H",
            }
        ]
    )

    updated, matched, changed = (
        reconcile_completed_results(
            ledger,
            completed_results,
        )
    )

    assert matched == 1
    assert changed == 1
    assert updated.iloc[0]["home_goals"] == "2"
    assert updated.iloc[0]["away_goals"] == "1"
    assert updated.iloc[0]["result"] == "H"


def test_reconcile_completed_results_supports_old_team_names():
    ledger = pd.DataFrame(
        [
            _ledger_row(
                fixture_id=(
                    "2026-27_mw01_"
                    "ipswich-town_sunderland"
                ),
                season="2026_27",
                matchweek=1,
                home_team="Ipswich Town",
                away_team="Sunderland",
            )
        ]
    )

    completed_results = pd.DataFrame(
        [
            {
                "season": "2026_27",
                "matchweek": 1,
                "home_team": "Ipswich",
                "away_team": "Sunderland",
                "home_goals": 2,
                "away_goals": 1,
                "result": "H",
            }
        ]
    )

    updated, matched, changed = (
        reconcile_completed_results(
            ledger,
            completed_results,
        )
    )

    assert matched == 1
    assert changed == 1
    assert updated.iloc[0]["result"] == "H"


def test_reconcile_completed_results_is_idempotent():
    ledger = pd.DataFrame(
        [
            _ledger_row(
                fixture_id=(
                    "2026-27_mw01_"
                    "arsenal_coventry-city"
                ),
                season="2026_27",
                matchweek=1,
                home_team="Arsenal",
                away_team="Coventry City",
                home_goals="3",
                away_goals="0",
                result="H",
            )
        ]
    )

    completed_results = pd.DataFrame(
        [
            {
                "season": "2026_27",
                "matchweek": 1,
                "home_team": "Arsenal",
                "away_team": "Coventry City",
                "home_goals": 3,
                "away_goals": 0,
                "result": "H",
            }
        ]
    )

    updated, matched, changed = (
        reconcile_completed_results(
            ledger,
            completed_results,
        )
    )

    assert matched == 1
    assert changed == 0
    assert updated.iloc[0]["result"] == "H"


def test_reconcile_completed_results_rejects_conflict():
    ledger = pd.DataFrame(
        [
            _ledger_row(
                fixture_id=(
                    "2026-27_mw01_"
                    "arsenal_coventry-city"
                ),
                season="2026_27",
                matchweek=1,
                home_team="Arsenal",
                away_team="Coventry City",
                home_goals="3",
                away_goals="0",
                result="H",
            )
        ]
    )

    completed_results = pd.DataFrame(
        [
            {
                "season": "2026_27",
                "matchweek": 1,
                "home_team": "Arsenal",
                "away_team": "Coventry City",
                "home_goals": 1,
                "away_goals": 1,
                "result": "D",
            }
        ]
    )

    with pytest.raises(ValueError):
        reconcile_completed_results(
            ledger,
            completed_results,
        )


def test_reconcile_completed_results_preserves_predictions():
    ledger = pd.DataFrame(
        [
            _ledger_row(
                fixture_id=(
                    "2026-27_mw03_"
                    "arsenal_chelsea"
                ),
                season="2026_27",
                matchweek=3,
                home_team="Arsenal",
                away_team="Chelsea",
            )
        ]
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

    completed_results = pd.DataFrame(
        [
            {
                "season": "2026_27",
                "matchweek": 3,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_goals": 2,
                "away_goals": 0,
                "result": "H",
            }
        ]
    )

    updated, _, _ = (
        reconcile_completed_results(
            ledger,
            completed_results,
        )
    )

    assert updated[
        immutable_columns
    ].equals(
        immutable_before
    )
