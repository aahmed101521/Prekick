import pytest

from prekick.live import (
    find_duplicate_fixture_ids,
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
