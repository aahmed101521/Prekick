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
