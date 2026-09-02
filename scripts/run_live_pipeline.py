import subprocess
import sys

import pandas as pd

from prekick.live import (
    classify_fixture_batch,
)


UPCOMING_PATH = (
    "data/fixtures/upcoming_fixtures.csv"
)

LEDGER_PATH = (
    "predictions/ledger.csv"
)


def run_script(*arguments):
    command = [
        sys.executable,
        *arguments,
    ]

    print()
    print(
        "=" * 60
    )

    print(
        "Running:",
        " ".join(command),
    )

    print(
        "=" * 60
    )
    print()

    subprocess.run(
        command,
        check=True,
    )


def main():
    # --------------------------------------------------------
    # 1. REFRESH RESULTS AND UPCOMING FIXTURES
    # --------------------------------------------------------

    run_script(
        "scripts/refresh_live_data.py"
    )

    # --------------------------------------------------------
    # 2. RECONCILE FINISHED MATCHES WITH THE LEDGER
    # --------------------------------------------------------

    run_script(
        "scripts/reconcile_live_ledger.py",
        "--write",
    )

    # --------------------------------------------------------
    # 3. SCORE ALL COMPLETED PREDICTIONS
    # --------------------------------------------------------

    run_script(
        "scripts/score_live_ledger.py"
    )

    # --------------------------------------------------------
    # 4. CHECK WHETHER UPCOMING FIXTURES NEED PREDICTIONS
    # --------------------------------------------------------

    fixtures = pd.read_csv(
        UPCOMING_PATH,
        dtype=str,
        keep_default_na=False,
    )

    ledger = pd.read_csv(
        LEDGER_PATH,
        dtype=str,
        keep_default_na=False,
    )

    if fixtures[
        "fixture_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate fixture IDs found "
            "in upcoming fixtures."
        )

    batch_state = classify_fixture_batch(
        upcoming_fixture_ids=fixtures[
            "fixture_id"
        ],
        ledger_fixture_ids=ledger[
            "fixture_id"
        ],
    )

    already_predicted = len(
        set(
            fixtures["fixture_id"]
        )
        & set(
            ledger["fixture_id"]
        )
    )

    new_predictions = len(
        set(
            fixtures["fixture_id"]
        )
        - set(
            ledger["fixture_id"]
        )
    )

    print()
    print(
        "Upcoming fixtures:",
        len(fixtures),
    )

    print(
        "Already predicted:",
        already_predicted,
    )

    print(
        "New fixtures requiring predictions:",
        new_predictions,
    )

    # --------------------------------------------------------
    # 5. GENERATE PREDICTIONS WHEN THE WHOLE BATCH IS NEW
    # --------------------------------------------------------

    if batch_state == "predicted":
        print()
        print(
            "All upcoming fixtures already "
            "have ledger predictions."
        )

        print(
            "Prediction generation skipped."
        )

    elif batch_state == "new":
        run_script(
            "scripts/"
            "generate_prekick_v1_predictions.py"
        )

    elif batch_state == "mixed":
        raise ValueError(
            "Upcoming fixture batch is in a mixed state: "
            "some fixtures already have predictions "
            "and some do not. Refusing to continue "
            "automatically."
        )

    else:
        raise ValueError(
            "No upcoming fixtures are available."
        )

    # --------------------------------------------------------
    # 6. FINAL SUMMARY
    # --------------------------------------------------------

    final_ledger = pd.read_csv(
        LEDGER_PATH,
        dtype=str,
        keep_default_na=False,
    )

    completed_predictions = (
        final_ledger["result"].isin(
            ["H", "D", "A"]
        ).sum()
    )

    unfinished_predictions = (
        ~final_ledger["result"].isin(
            ["H", "D", "A"]
        )
    ).sum()

    print()
    print(
        "=" * 60
    )

    print(
        "PREKICK LIVE PIPELINE COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "Ledger rows:",
        len(final_ledger),
    )

    print(
        "Completed predictions:",
        completed_predictions,
    )

    print(
        "Unfinished predictions:",
        unfinished_predictions,
    )


if __name__ == "__main__":
    main()
