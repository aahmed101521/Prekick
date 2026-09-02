import pandas as pd

from prekick.data_sources import (
    normalise_team_name,
)


IMMUTABLE_PREDICTION_COLUMNS = [
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


def validate_live_training_count(
    historical_count,
    current_season_count,
    live_training_count,
):
    expected_count = (
        historical_count
        + current_season_count
    )

    if live_training_count != expected_count:
        raise ValueError(
            "Live training row count does not equal "
            "historical plus current-season matches."
        )


def validate_fixture_count(fixture_count):
    if fixture_count == 0:
        raise ValueError(
            "No upcoming fixtures found."
        )


def validate_prediction_count(
    prediction_count,
    fixture_count,
):
    if prediction_count != fixture_count:
        raise ValueError(
            "Prediction row count does not match fixture count."
        )


def validate_prediction_timing(
    kickoff_times,
    predicted_at_utc,
):
    predicted_at = pd.to_datetime(
        predicted_at_utc,
        utc=True,
        errors="coerce",
    )

    if pd.isna(predicted_at):
        raise ValueError(
            "Prediction timestamp is invalid."
        )

    parsed_kickoffs = pd.to_datetime(
        pd.Series(kickoff_times),
        utc=True,
        errors="coerce",
    )

    if parsed_kickoffs.isna().any():
        raise ValueError(
            "Fixture kickoff timestamps are invalid."
        )

    if (parsed_kickoffs <= predicted_at).any():
        raise ValueError(
            "Cannot generate prospective predictions "
            "at or after fixture kickoff."
        )


def find_duplicate_fixture_ids(
    prediction_fixture_ids,
    existing_fixture_ids,
):
    return sorted(
        set(prediction_fixture_ids)
        & set(existing_fixture_ids)
    )


def reject_duplicate_fixture_ids(
    prediction_fixture_ids,
    existing_fixture_ids,
):
    duplicate_fixture_ids = (
        find_duplicate_fixture_ids(
            prediction_fixture_ids,
            existing_fixture_ids,
        )
    )

    if duplicate_fixture_ids:
        raise ValueError(
            "These fixture IDs already exist in the ledger: "
            + ", ".join(duplicate_fixture_ids)
        )


def _match_key(
    season,
    matchweek,
    home_team,
    away_team,
):
    return (
        str(season),
        int(matchweek),
        normalise_team_name(
            str(home_team)
        ),
        normalise_team_name(
            str(away_team)
        ),
    )


def _result_from_goals(
    home_goals,
    away_goals,
):
    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def reconcile_completed_results(
    ledger,
    completed_results,
):
    required_ledger_columns = (
        IMMUTABLE_PREDICTION_COLUMNS
        + [
            "home_goals",
            "away_goals",
            "result",
        ]
    )

    missing_ledger_columns = [
        column
        for column in required_ledger_columns
        if column not in ledger.columns
    ]

    if missing_ledger_columns:
        raise ValueError(
            "Ledger is missing required columns: "
            + ", ".join(
                missing_ledger_columns
            )
        )

    required_result_columns = [
        "season",
        "matchweek",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    ]

    missing_result_columns = [
        column
        for column in required_result_columns
        if column not in completed_results.columns
    ]

    if missing_result_columns:
        raise ValueError(
            "Completed results are missing "
            "required columns: "
            + ", ".join(
                missing_result_columns
            )
        )

    updated_ledger = ledger.copy()

    immutable_before = (
        updated_ledger[
            IMMUTABLE_PREDICTION_COLUMNS
        ].copy()
    )

    result_lookup = {}

    for result_row in (
        completed_results.itertuples(
            index=False
        )
    ):
        home_goals = int(
            result_row.home_goals
        )

        away_goals = int(
            result_row.away_goals
        )

        expected_result = (
            _result_from_goals(
                home_goals,
                away_goals,
            )
        )

        if result_row.result not in {
            "H",
            "D",
            "A",
        }:
            raise ValueError(
                "Completed match result "
                "must be H, D, or A."
            )

        if (
            result_row.result
            != expected_result
        ):
            raise ValueError(
                "Completed match result "
                "does not agree with goals."
            )

        key = _match_key(
            season=result_row.season,
            matchweek=result_row.matchweek,
            home_team=(
                result_row.home_team
            ),
            away_team=(
                result_row.away_team
            ),
        )

        if key in result_lookup:
            raise ValueError(
                "Duplicate completed result "
                "found during reconciliation."
            )

        result_lookup[key] = {
            "home_goals": str(
                home_goals
            ),
            "away_goals": str(
                away_goals
            ),
            "result": (
                result_row.result
            ),
        }

    matched_rows = 0
    updated_rows = 0

    for index in updated_ledger.index:
        key = _match_key(
            season=updated_ledger.at[
                index,
                "season",
            ],
            matchweek=updated_ledger.at[
                index,
                "matchweek",
            ],
            home_team=updated_ledger.at[
                index,
                "home_team",
            ],
            away_team=updated_ledger.at[
                index,
                "away_team",
            ],
        )

        if key not in result_lookup:
            continue

        matched_rows += 1

        expected_values = (
            result_lookup[key]
        )

        row_changed = False

        for column in [
            "home_goals",
            "away_goals",
            "result",
        ]:
            existing_value = (
                updated_ledger.at[
                    index,
                    column,
                ]
            )

            if pd.isna(existing_value):
                existing_value = ""
            else:
                existing_value = str(
                    existing_value
                )

            expected_value = (
                expected_values[column]
            )

            if (
                existing_value
                not in {
                    "",
                    expected_value,
                }
            ):
                raise ValueError(
                    "Existing ledger result "
                    "conflicts with completed "
                    "match data for fixture "
                    + str(
                        updated_ledger.at[
                            index,
                            "fixture_id",
                        ]
                    )
                )

            if existing_value == "":
                updated_ledger.at[
                    index,
                    column,
                ] = expected_value

                row_changed = True

        if row_changed:
            updated_rows += 1

    if not updated_ledger[
        IMMUTABLE_PREDICTION_COLUMNS
    ].equals(
        immutable_before
    ):
        raise ValueError(
            "Immutable prediction fields "
            "were changed during "
            "result reconciliation."
        )

    return (
        updated_ledger,
        matched_rows,
        updated_rows,
    )


import pandas as pd

from prekick.live import (
    reconcile_completed_results,
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

    immutable_before = ledger[
        [
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
        immutable_before.columns
    ].equals(
        immutable_before
    )


def classify_fixture_batch(
    upcoming_fixture_ids,
    ledger_fixture_ids,
):
    upcoming_fixture_ids = set(
        upcoming_fixture_ids
    )

    ledger_fixture_ids = set(
        ledger_fixture_ids
    )

    existing_fixture_ids = (
        upcoming_fixture_ids
        & ledger_fixture_ids
    )

    new_fixture_ids = (
        upcoming_fixture_ids
        - ledger_fixture_ids
    )

    if not upcoming_fixture_ids:
        return "empty"

    if not existing_fixture_ids:
        return "new"

    if not new_fixture_ids:
        return "predicted"

    return "mixed"
