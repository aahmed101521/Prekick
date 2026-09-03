import pytest

from prekick.data_sources import (
    build_live_data,
    completed_matches_dataframe,
    completed_results_dataframe,
    normalise_team,
    normalise_team_name,
    season_label,
    season_slug,
    upcoming_fixtures_dataframe,
    validate_premier_league_schedule,
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


def test_normalise_team_name_supports_old_ledger_names():
    assert (
        normalise_team_name(
            "Manchester United"
        )
        == "Man United"
    )

    assert (
        normalise_team_name(
            "Ipswich Town"
        )
        == "Ipswich"
    )

    assert (
        normalise_team_name(
            "Leeds United"
        )
        == "Leeds"
    )

    assert (
        normalise_team_name(
            "Brighton & Hove Albion"
        )
        == "Brighton"
    )


def test_completed_results_dataframe_includes_matchweek():
    matches = [
        {
            "status": "FINISHED",
            "matchday": 1,
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
        }
    ]

    results = completed_results_dataframe(
        matches,
        season_start=2026,
    )

    row = results.iloc[0]

    assert row["season"] == "2026_27"
    assert row["matchweek"] == 1
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Coventry City"
    assert row["home_goals"] == 3
    assert row["away_goals"] == 0
    assert row["result"] == "H"

SCHEDULE_TEST_TEAMS = [
    "Arsenal FC",
    "AFC Bournemouth",
    "Aston Villa FC",
    "Brentford FC",
    "Brighton & Hove Albion FC",
    "Chelsea FC",
    "Coventry City FC",
    "Crystal Palace FC",
    "Everton FC",
    "Fulham FC",
    "Hull City AFC",
    "Ipswich Town FC",
    "Leeds United FC",
    "Liverpool FC",
    "Manchester City FC",
    "Manchester United FC",
    "Newcastle United FC",
    "Nottingham Forest FC",
    "Sunderland AFC",
    "Tottenham Hotspur FC",
]


def _complete_test_schedule():
    teams = SCHEDULE_TEST_TEAMS.copy()
    fixed_team = teams[0]
    rotating = teams[1:]
    first_half = []

    for matchweek in range(1, 20):
        order = [
            fixed_team,
            *rotating,
        ]
        round_fixtures = []

        for index in range(10):
            home_team = order[index]
            away_team = order[-(index + 1)]

            if (matchweek + index) % 2 == 0:
                home_team, away_team = (
                    away_team,
                    home_team,
                )

            round_fixtures.append(
                (
                    home_team,
                    away_team,
                )
            )

        first_half.append(round_fixtures)
        rotating = [
            rotating[-1],
            *rotating[:-1],
        ]

    matches = []

    for first_half_index, round_fixtures in enumerate(
        first_half,
        start=1,
    ):
        for home_team, away_team in round_fixtures:
            matches.append(
                {
                    "status": "SCHEDULED",
                    "matchday": first_half_index,
                    "utcDate": "2026-09-01T15:00:00Z",
                    "homeTeam": {
                        "name": home_team,
                    },
                    "awayTeam": {
                        "name": away_team,
                    },
                }
            )

        for home_team, away_team in round_fixtures:
            matches.append(
                {
                    "status": "SCHEDULED",
                    "matchday": first_half_index + 19,
                    "utcDate": "2027-01-01T15:00:00Z",
                    "homeTeam": {
                        "name": away_team,
                    },
                    "awayTeam": {
                        "name": home_team,
                    },
                }
            )

    return matches


def test_validate_premier_league_schedule_accepts_complete_schedule():
    matches = _complete_test_schedule()

    validate_premier_league_schedule(
        matches
    )


def test_validate_premier_league_schedule_rejects_missing_fixture():
    matches = _complete_test_schedule()
    matches.pop()

    with pytest.raises(
        ValueError,
        match="must contain 380 fixtures",
    ):
        validate_premier_league_schedule(
            matches
        )


def test_validate_premier_league_schedule_rejects_duplicate_pairing():
    matches = _complete_test_schedule()

    matches[-1] = {
        **matches[-1],
        "homeTeam": matches[0]["homeTeam"].copy(),
        "awayTeam": matches[0]["awayTeam"].copy(),
    }

    with pytest.raises(
        ValueError,
        match="Duplicate Premier League home-away pairing",
    ):
        validate_premier_league_schedule(
            matches
        )


def test_build_live_data_rejects_incomplete_schedule(
    monkeypatch,
):
    matches = _complete_test_schedule()
    matches.pop()

    monkeypatch.setattr(
        "prekick.data_sources.fetch_premier_league_matches",
        lambda **kwargs: matches,
    )

    with pytest.raises(
        ValueError,
        match="must contain 380 fixtures",
    ):
        build_live_data(
            api_key="test-key",
            season_start=2026,
        )
