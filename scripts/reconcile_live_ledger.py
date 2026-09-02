import argparse

import pandas as pd

from prekick.data_sources import (
    completed_results_dataframe,
    fetch_premier_league_matches,
)
from prekick.live import (
    reconcile_completed_results,
)


LEDGER_PATH = "predictions/ledger.csv"
SEASON_START = 2026

RESULT_COLUMNS = [
    "home_goals",
    "away_goals",
    "result",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile completed Premier League "
            "results with the Prekick prediction ledger."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write reconciled results to the ledger. "
            "Without this flag the script performs "
            "a dry run only."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    ledger = pd.read_csv(
        LEDGER_PATH,
        dtype=str,
        keep_default_na=False,
    )

    original_ledger = ledger.copy()

    matches = fetch_premier_league_matches(
        season_start=SEASON_START,
    )

    completed_results = (
        completed_results_dataframe(
            matches,
            season_start=SEASON_START,
        )
    )

    (
        updated_ledger,
        matched_rows,
        updated_rows,
    ) = reconcile_completed_results(
        ledger,
        completed_results,
    )

    changed_mask = (
        original_ledger[
            RESULT_COLUMNS
        ]
        != updated_ledger[
            RESULT_COLUMNS
        ]
    ).any(axis=1)

    changed_rows = updated_ledger.loc[
        changed_mask,
        [
            "fixture_id",
            "model_version",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
        ],
    ]

    print(
        "Completed API matches:",
        len(completed_results),
    )

    print(
        "Ledger rows matched:",
        matched_rows,
    )

    print(
        "Ledger rows requiring updates:",
        updated_rows,
    )

    if changed_rows.empty:
        print()
        print(
            "No ledger result updates are required."
        )
    else:
        print()
        print(
            "Proposed ledger result updates:"
        )

        print()
        print(
            changed_rows.to_string(
                index=False,
            )
        )

    if not args.write:
        print()
        print(
            "DRY RUN ONLY - ledger was not written."
        )

        print(
            "Run again with --write "
            "to apply these result updates."
        )

        return

    if updated_rows == 0:
        print()
        print(
            "Nothing to write."
        )

        return

    updated_ledger.to_csv(
        LEDGER_PATH,
        index=False,
    )

    print()
    print(
        "Ledger result rows updated:",
        updated_rows,
    )

    print(
        "Ledger written:",
        LEDGER_PATH,
    )


if __name__ == "__main__":
    main()
