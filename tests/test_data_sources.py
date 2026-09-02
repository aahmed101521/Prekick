import pytest

from prekick.data_sources import (
    completed_matches_dataframe,
    normalise_team,
    season_label,
    season_slug,
    upcoming_fixtures_dataframe,
)


def test_season_label():
    assert season_label(2026) == "2026_27"


def test_season_slug():
    assert season_slug(2026) == "2026-27"


def test_normalise_team():
    assert normalise_team(
        {
            "name": "Manchester United FC",
            "shortName": "Man United",
        }
    ) == (
        "Man United",
        "manchester-united",
    )


def test_normalise_team_rejects_unknown_team():
    with pytest.raises(ValueError):
        normalise_team(
            {
                "name": "Unknown FC",
                "shortName": "Unknown",
            }
        )


def test_completed_matches_dataframe():
    matches = [
        {
            "status": "FINISHED",
            "utcDate": "2026-08-21T19:00:00Z",
            "homeTeam": {
                "name": "Arsenal FC",
            },
            "awayTeam": {
                "name": "Coventry City FC",
            },
            "score": {
                "fullTime": {
                    "home": 3,
                    "away": 0,
                }
            },
        },
        {
            "status": "TIMED",
            "utcDate": "2026-09-04T19:00:00Z",
            "homeTeam": {
                "name": "Ipswich Town FC",
            },
            "awayTeam": {
                "name": "Liverpool FC",
            },
            "score": {
                "fullTime": {
                    "home": None,
                    "away": None,
                }
            },
        },
    ]

    completed = completed_matches_dataframe(
        matches
    )

    assert len(completed) == 1

    row = completed.iloc[0]

    assert row["date"] == "2026-08-21"
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Coventry City"
    assert row["home_goals"] == 3
    assert row["away_goals"] == 0
    assert row["result"] == "H"


def test_completed_matches_dataframe_draw():
    matches = [
        {
            "status": "FINISHED",
            "utcDate": "2026-08-22T14:00:00Z",
            "homeTeam": {
                "name": "Chelsea FC",
            },
            "awayTeam": {
                "name": "Liverpool FC",
            },
            "score": {
                "fullTime": {
                    "home": 1,
                    "away": 1,
                }
            },
        }
    ]

    completed = completed_matches_dataframe(
        matches
    )

    assert completed.iloc[0]["result"] == "D"


def test_upcoming_fixtures_dataframe_uses_next_matchweek_only():
    matches = [
        {
            "status": "FINISHED",
            "matchday": 2,
            "utcDate": "2026-08-31T19:00:00Z",
            "homeTeam": {
                "name": "Aston Villa FC",
            },
            "awayTeam": {
                "name": "Arsenal FC",
            },
        },
        {
            "status": "TIMED",
            "matchday": 3,
            "utcDate": "2026-09-04T19:00:00Z",
            "homeTeam": {
                "name": "Ipswich Town FC",
            },
            "awayTeam": {
                "name": "Liverpool FC",
            },
        },
        {
            "status": "SCHEDULED",
            "matchday": 4,
            "utcDate": "2026-09-12T14:00:00Z",
            "homeTeam": {
                "name": "Aston Villa FC",
            },
            "awayTeam": {
                "name": "Nottingham Forest FC",
            },
        },
    ]

    fixtures = upcoming_fixtures_dataframe(
        matches,
        season_start=2026,
    )

    assert len(fixtures) == 1

    row = fixtures.iloc[0]

    assert (
        row["fixture_id"]
        == "2026-27_mw03_ipswich_liverpool"
    )
    assert row["season"] == "2026_27"
    assert row["matchweek"] == 3
    assert (
        row["kickoff_utc"]
        == "2026-09-04T19:00:00Z"
    )
    assert row["home_team"] == "Ipswich"
    assert row["away_team"] == "Liverpool"


def test_upcoming_fixture_id_preserves_existing_slug_convention():
    matches = [
        {
            "status": "TIMED",
            "matchday": 3,
            "utcDate": "2026-09-06T13:00:00Z",
            "homeTeam": {
                "name": "Everton FC",
            },
            "awayTeam": {
                "name": "Manchester United FC",
            },
        }
    ]

    fixtures = upcoming_fixtures_dataframe(
        matches,
        season_start=2026,
    )

    assert (
        fixtures.iloc[0]["fixture_id"]
        == (
            "2026-27_mw03_"
            "everton_manchester-united"
        )
    )


def test_upcoming_fixture_id_preserves_nottingham_slug():
    matches = [
        {
            "status": "TIMED",
            "matchday": 3,
            "utcDate": "2026-09-05T14:00:00Z",
            "homeTeam": {
                "name": "Nottingham Forest FC",
            },
            "awayTeam": {
                "name": "Tottenham Hotspur FC",
            },
        }
    ]

    fixtures = upcoming_fixtures_dataframe(
        matches,
        season_start=2026,
    )

    assert (
        fixtures.iloc[0]["fixture_id"]
        == (
            "2026-27_mw03_"
            "nottingham-forest_"
            "tottenham-hotspur"
        )
    )
