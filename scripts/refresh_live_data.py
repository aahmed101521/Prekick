from pathlib import Path

from prekick.data_sources import (
    build_live_data,
)


COMPLETED_PATH = Path(
    "data/fixtures/"
    "completed_matches_2026_27.csv"
)

UPCOMING_PATH = Path(
    "data/fixtures/"
    "upcoming_fixtures.csv"
)


def main():
    completed, upcoming = (
        build_live_data()
    )

    if completed.empty:
        raise ValueError(
            "No completed Premier League "
            "matches were returned."
        )

    if upcoming.empty:
        raise ValueError(
            "No upcoming Premier League "
            "fixtures were returned."
        )

    COMPLETED_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed.to_csv(
        COMPLETED_PATH,
        index=False,
    )

    upcoming.to_csv(
        UPCOMING_PATH,
        index=False,
    )

    print(
        "Completed matches written:",
        len(completed),
    )

    print(
        "Upcoming fixtures written:",
        len(upcoming),
    )

    print(
        "Completed matches file:",
        COMPLETED_PATH,
    )

    print(
        "Upcoming fixtures file:",
        UPCOMING_PATH,
    )


if __name__ == "__main__":
    main()
