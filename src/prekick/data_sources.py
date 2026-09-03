import os

import pandas as pd
import requests


FOOTBALL_DATA_API_URL = (
    "https://api.football-data.org/v4/"
    "competitions/PL/matches"
)

DEFAULT_SEASON_START = 2026
DEFAULT_TIMEOUT_SECONDS = 30

EXPECTED_EPL_TEAMS = 20
EXPECTED_EPL_MATCHWEEKS = 38
EXPECTED_EPL_FIXTURES_PER_MATCHWEEK = 10
EXPECTED_EPL_FIXTURES = 380

UPCOMING_STATUSES = {
    "SCHEDULED",
    "TIMED",
}


# ------------------------------------------------------------
# Football-data.org name -> Prekick name + stable fixture slug
#
# The slug is deliberately separate from the model-facing
# team name because the existing ledger uses identifiers such
# as "manchester-united" while the model uses "Man United".
# ------------------------------------------------------------

TEAM_NAME_MAP = {
    "Arsenal FC": (
        "Arsenal",
        "arsenal",
    ),
    "Arsenal": (
        "Arsenal",
        "arsenal",
    ),
    "AFC Bournemouth": (
        "Bournemouth",
        "bournemouth",
    ),
    "Bournemouth": (
        "Bournemouth",
        "bournemouth",
    ),
    "Aston Villa FC": (
        "Aston Villa",
        "aston-villa",
    ),
    "Aston Villa": (
        "Aston Villa",
        "aston-villa",
    ),
    "Brentford FC": (
        "Brentford",
        "brentford",
    ),
    "Brentford": (
        "Brentford",
        "brentford",
    ),
    "Brighton & Hove Albion FC": (
        "Brighton",
        "brighton",
    ),
    "Brighton & Hove Albion": (
        "Brighton",
        "brighton",
    ),
    "Brighton": (
        "Brighton",
        "brighton",
    ),
    "Chelsea FC": (
        "Chelsea",
        "chelsea",
    ),
    "Chelsea": (
        "Chelsea",
        "chelsea",
    ),
    "Coventry City FC": (
        "Coventry City",
        "coventry-city",
    ),
    "Coventry City": (
        "Coventry City",
        "coventry-city",
    ),
    "Crystal Palace FC": (
        "Crystal Palace",
        "crystal-palace",
    ),
    "Crystal Palace": (
        "Crystal Palace",
        "crystal-palace",
    ),
    "Everton FC": (
        "Everton",
        "everton",
    ),
    "Everton": (
        "Everton",
        "everton",
    ),
    "Fulham FC": (
        "Fulham",
        "fulham",
    ),
    "Fulham": (
        "Fulham",
        "fulham",
    ),
    "Hull City AFC": (
        "Hull City",
        "hull-city",
    ),
    "Hull City FC": (
        "Hull City",
        "hull-city",
    ),
    "Hull City": (
        "Hull City",
        "hull-city",
    ),
    "Ipswich Town FC": (
        "Ipswich",
        "ipswich",
    ),
    "Ipswich Town": (
        "Ipswich",
        "ipswich",
    ),
    "Ipswich": (
        "Ipswich",
        "ipswich",
    ),
    "Leeds United FC": (
        "Leeds",
        "leeds",
    ),
    "Leeds United": (
        "Leeds",
        "leeds",
    ),
    "Leeds": (
        "Leeds",
        "leeds",
    ),
    "Liverpool FC": (
        "Liverpool",
        "liverpool",
    ),
    "Liverpool": (
        "Liverpool",
        "liverpool",
    ),
    "Manchester City FC": (
        "Man City",
        "manchester-city",
    ),
    "Manchester City": (
        "Man City",
        "manchester-city",
    ),
    "Man City": (
        "Man City",
        "manchester-city",
    ),
    "Manchester United FC": (
        "Man United",
        "manchester-united",
    ),
    "Manchester United": (
        "Man United",
        "manchester-united",
    ),
    "Man United": (
        "Man United",
        "manchester-united",
    ),
    "Newcastle United FC": (
        "Newcastle",
        "newcastle-united",
    ),
    "Newcastle United": (
        "Newcastle",
        "newcastle-united",
    ),
    "Newcastle": (
        "Newcastle",
        "newcastle-united",
    ),
    "Nottingham Forest FC": (
        "Nott'm Forest",
        "nottingham-forest",
    ),
    "Nottingham Forest": (
        "Nott'm Forest",
        "nottingham-forest",
    ),
    "Nott'm Forest": (
        "Nott'm Forest",
        "nottingham-forest",
    ),
    "Sunderland AFC": (
        "Sunderland",
        "sunderland",
    ),
    "Sunderland": (
        "Sunderland",
        "sunderland",
    ),
    "Tottenham Hotspur FC": (
        "Tottenham",
        "tottenham-hotspur",
    ),
    "Tottenham Hotspur": (
        "Tottenham",
        "tottenham-hotspur",
    ),
    "Tottenham": (
        "Tottenham",
        "tottenham-hotspur",
    ),
}


COMPLETED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]


UPCOMING_COLUMNS = [
    "fixture_id",
    "season",
    "matchweek",
    "kickoff_utc",
    "home_team",
    "away_team",
]


COMPLETED_RESULT_COLUMNS = [
    "season",
    "matchweek",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
]


def season_label(season_start):
    next_year = (
        season_start + 1
    ) % 100

    return (
        f"{season_start}_"
        f"{next_year:02d}"
    )


def season_slug(season_start):
    next_year = (
        season_start + 1
    ) % 100

    return (
        f"{season_start}-"
        f"{next_year:02d}"
    )


def normalise_team(team):
    candidates = [
        team.get("name"),
        team.get("shortName"),
    ]

    for candidate in candidates:
        if candidate in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[
                candidate
            ]

    available_names = [
        candidate
        for candidate in candidates
        if candidate
    ]

    raise ValueError(
        "Unknown football-data.org team: "
        + " / ".join(available_names)
    )



def validate_premier_league_schedule(
    matches,
):
    if len(matches) != EXPECTED_EPL_FIXTURES:
        raise ValueError(
            "Premier League season schedule "
            f"must contain {EXPECTED_EPL_FIXTURES} fixtures; "
            f"found {len(matches)}."
        )

    teams = set()
    ordered_pairings = set()
    teams_by_matchweek = {
        matchweek: set()
        for matchweek in range(
            1,
            EXPECTED_EPL_MATCHWEEKS + 1,
        )
    }
    fixtures_by_matchweek = {
        matchweek: 0
        for matchweek in range(
            1,
            EXPECTED_EPL_MATCHWEEKS + 1,
        )
    }

    for match in matches:
        matchday = match.get("matchday")
        if matchday is None:
            raise ValueError(
                "Premier League fixture has no matchday."
            )

        matchweek = int(matchday)
        if matchweek not in fixtures_by_matchweek:
            raise ValueError(
                "Premier League fixture has invalid matchday: "
                f"{matchweek}."
            )

        home_team, _ = normalise_team(
            match["homeTeam"]
        )
        away_team, _ = normalise_team(
            match["awayTeam"]
        )

        if home_team == away_team:
            raise ValueError(
                "Premier League fixture has the same "
                "home and away team."
            )

        pairing = (
            home_team,
            away_team,
        )
        if pairing in ordered_pairings:
            raise ValueError(
                "Duplicate Premier League home-away "
                "pairing found in season schedule."
            )

        ordered_pairings.add(pairing)
        teams.update(
            [
                home_team,
                away_team,
            ]
        )

        fixtures_by_matchweek[matchweek] += 1

        matchweek_teams = teams_by_matchweek[matchweek]
        if (
            home_team in matchweek_teams
            or away_team in matchweek_teams
        ):
            raise ValueError(
                "Premier League team appears more than "
                "once in the same matchweek."
            )

        matchweek_teams.update(
            [
                home_team,
                away_team,
            ]
        )

    if len(teams) != EXPECTED_EPL_TEAMS:
        raise ValueError(
            "Premier League season schedule "
            f"must contain {EXPECTED_EPL_TEAMS} teams; "
            f"found {len(teams)}."
        )

    for matchweek in range(
        1,
        EXPECTED_EPL_MATCHWEEKS + 1,
    ):
        fixture_count = fixtures_by_matchweek[matchweek]
        if fixture_count != EXPECTED_EPL_FIXTURES_PER_MATCHWEEK:
            raise ValueError(
                "Premier League matchweek "
                f"{matchweek} must contain "
                f"{EXPECTED_EPL_FIXTURES_PER_MATCHWEEK} "
                f"fixtures; found {fixture_count}."
            )

        team_count = len(teams_by_matchweek[matchweek])
        if team_count != EXPECTED_EPL_TEAMS:
            raise ValueError(
                "Premier League matchweek "
                f"{matchweek} must contain "
                f"{EXPECTED_EPL_TEAMS} teams; "
                f"found {team_count}."
            )


def fetch_premier_league_matches(
    api_key=None,
    season_start=DEFAULT_SEASON_START,
    timeout=DEFAULT_TIMEOUT_SECONDS,
):
    if api_key is None:
        api_key = os.getenv(
            "FOOTBALL_DATA_API_KEY"
        )

    if not api_key:
        raise ValueError(
            "FOOTBALL_DATA_API_KEY is not set."
        )

    response = requests.get(
        FOOTBALL_DATA_API_URL,
        headers={
            "X-Auth-Token": api_key,
        },
        params={
            "season": season_start,
        },
        timeout=timeout,
    )

    response.raise_for_status()

    payload = response.json()

    matches = payload.get(
        "matches"
    )

    if not isinstance(matches, list):
        raise ValueError(
            "Football-data.org response "
            "does not contain a match list."
        )

    return matches


def _normalise_utc_timestamp(
    value,
):
    timestamp = pd.to_datetime(
        value,
        utc=True,
    )

    return timestamp.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def normalise_team_name(
    team_name,
):
    if team_name not in TEAM_NAME_MAP:
        raise ValueError(
            "Unknown team name: "
            + str(team_name)
        )

    return TEAM_NAME_MAP[
        team_name
    ][0]


def _result_from_goals(
    home_goals,
    away_goals,
):
    if home_goals > away_goals:
        return "H"

    if home_goals < away_goals:
        return "A"

    return "D"


def completed_matches_dataframe(
    matches,
):
    rows = []

    for match in matches:
        if match.get("status") != "FINISHED":
            continue

        score = match.get(
            "score",
            {},
        ).get(
            "fullTime",
            {},
        )

        home_goals = score.get(
            "home"
        )

        away_goals = score.get(
            "away"
        )

        if (
            home_goals is None
            or away_goals is None
        ):
            raise ValueError(
                "Finished match has missing "
                "full-time score."
            )

        (
            home_team,
            _,
        ) = normalise_team(
            match["homeTeam"]
        )

        (
            away_team,
            _,
        ) = normalise_team(
            match["awayTeam"]
        )

        kickoff_utc = (
            _normalise_utc_timestamp(
                match["utcDate"]
            )
        )

        rows.append(
            {
                "_kickoff_utc": (
                    kickoff_utc
                ),
                "date": (
                    kickoff_utc[:10]
                ),
                "home_team": (
                    home_team
                ),
                "away_team": (
                    away_team
                ),
                "home_goals": int(
                    home_goals
                ),
                "away_goals": int(
                    away_goals
                ),
                "result": (
                    _result_from_goals(
                        home_goals,
                        away_goals,
                    )
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=COMPLETED_COLUMNS
        )

    completed = pd.DataFrame(
        rows
    )

    completed = completed.sort_values(
        [
            "_kickoff_utc",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    completed = completed[
        COMPLETED_COLUMNS
    ]

    return completed


def completed_results_dataframe(
    matches,
    season_start=DEFAULT_SEASON_START,
):
    rows = []

    for match in matches:
        if match.get("status") != "FINISHED":
            continue

        matchday = match.get(
            "matchday"
        )

        if matchday is None:
            raise ValueError(
                "Finished match has no matchday."
            )

        score = match.get(
            "score",
            {},
        ).get(
            "fullTime",
            {},
        )

        home_goals = score.get(
            "home"
        )

        away_goals = score.get(
            "away"
        )

        if (
            home_goals is None
            or away_goals is None
        ):
            raise ValueError(
                "Finished match has missing "
                "full-time score."
            )

        (
            home_team,
            _,
        ) = normalise_team(
            match["homeTeam"]
        )

        (
            away_team,
            _,
        ) = normalise_team(
            match["awayTeam"]
        )

        rows.append(
            {
                "season": (
                    season_label(
                        season_start
                    )
                ),
                "matchweek": int(
                    matchday
                ),
                "home_team": (
                    home_team
                ),
                "away_team": (
                    away_team
                ),
                "home_goals": int(
                    home_goals
                ),
                "away_goals": int(
                    away_goals
                ),
                "result": (
                    _result_from_goals(
                        home_goals,
                        away_goals,
                    )
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=COMPLETED_RESULT_COLUMNS
        )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        [
            "matchweek",
            "home_team",
            "away_team",
        ]
    ).reset_index(
        drop=True
    )

    results = results[
        COMPLETED_RESULT_COLUMNS
    ]

    duplicated_matches = results.duplicated(
        subset=[
            "season",
            "matchweek",
            "home_team",
            "away_team",
        ]
    )

    if duplicated_matches.any():
        raise ValueError(
            "Duplicate completed matches "
            "found in API data."
        )

    return results


def upcoming_fixtures_dataframe(
    matches,
    season_start=DEFAULT_SEASON_START,
):
    upcoming_matches = [
        match
        for match in matches
        if (
            match.get("status")
            in UPCOMING_STATUSES
            and match.get("matchday")
            is not None
        )
    ]

    if not upcoming_matches:
        return pd.DataFrame(
            columns=UPCOMING_COLUMNS
        )

    next_matchweek = min(
        int(match["matchday"])
        for match in upcoming_matches
    )

    next_matches = [
        match
        for match in upcoming_matches
        if int(
            match["matchday"]
        ) == next_matchweek
    ]

    rows = []

    for match in next_matches:
        (
            home_team,
            home_slug,
        ) = normalise_team(
            match["homeTeam"]
        )

        (
            away_team,
            away_slug,
        ) = normalise_team(
            match["awayTeam"]
        )

        matchweek = int(
            match["matchday"]
        )

        fixture_id = (
            f"{season_slug(season_start)}"
            f"_mw{matchweek:02d}"
            f"_{home_slug}"
            f"_{away_slug}"
        )

        rows.append(
            {
                "fixture_id": (
                    fixture_id
                ),
                "season": (
                    season_label(
                        season_start
                    )
                ),
                "matchweek": (
                    matchweek
                ),
                "kickoff_utc": (
                    _normalise_utc_timestamp(
                        match["utcDate"]
                    )
                ),
                "home_team": (
                    home_team
                ),
                "away_team": (
                    away_team
                ),
            }
        )

    fixtures = pd.DataFrame(
        rows
    )

    fixtures = fixtures.sort_values(
        [
            "kickoff_utc",
            "fixture_id",
        ]
    ).reset_index(
        drop=True
    )

    fixtures = fixtures[
        UPCOMING_COLUMNS
    ]

    if fixtures[
        "fixture_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate fixture IDs generated "
            "from football-data.org."
        )

    return fixtures


def build_live_data(
    api_key=None,
    season_start=DEFAULT_SEASON_START,
):
    matches = (
        fetch_premier_league_matches(
            api_key=api_key,
            season_start=season_start,
        )
    )

    completed = (
        completed_matches_dataframe(
            matches
        )
    )

    upcoming = (
        upcoming_fixtures_dataframe(
            matches,
            season_start=season_start,
        )
    )

    return completed, upcoming
